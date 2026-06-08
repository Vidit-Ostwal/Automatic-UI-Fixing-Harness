"""
Trajectory runner — executes one BFS trajectory against a fresh Docker instance.

For each step in the trajectory:
  1. Capture state BEFORE the action.
  2. Attempt the action (click selector; fall back to LLM if selector fails).
  3. Capture state AFTER the action.
  4. Fire all three oracles: visual, logic (crash + count), diff.
  5. Collect any findings into the FindingCollector.

After all steps, replay the workflow at 3 viewport sizes (375 / 768 / 1280 px)
and run the visual oracle again at each size.

The runner never raises — all exceptions are caught and logged so that one
bad trajectory doesn't kill the rest of the parallel batch.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from browser.session import BrowserSession
from models import BugType, DetectedBy, Finding, Severity
from oracles.diff import DiffOracle
from oracles.logic import LogicOracle
from oracles.visual import VisualOracle
from reporter.collector import FindingCollector

logger = logging.getLogger(__name__)

# Viewport sizes for multi-viewport regression checks.
VIEWPORTS = [(375, 812), (768, 1024), (1280, 800)]


@dataclass
class TrajectoryStep:
    action: str
    steps: list[dict]   # [{"type": "fill"|"click"|"press"|"select", "selector": str, "value": str}]
    from_hash: str = ""
    to_hash: str = ""
    description: str = ""


@dataclass
class RunnerConfig:
    trajectory_id: str = ""
    app_url: str = "http://localhost:5230"
    expect_change_actions: set[str] = field(default_factory=lambda: {
        "create_memo", "edit_memo", "pin_memo", "unpin_memo",
        "archive_memo", "delete_memo", "submit", "save", "toggle",
    })


class TrajectoryRunner:
    """
    Executes one trajectory and returns a populated FindingCollector.

    Parameters
    ----------
    steps            List of TrajectoryStep objects from the BFS graph.
    config           RunnerConfig (trajectory id, app URL, etc.).
    visual_oracle    VisualOracle instance (shared, stateless).
    logic_oracle     LogicOracle instance (shared, stateless).
    diff_oracle      DiffOracle instance (may hold an LLM client).
    llm_oracle       Optional LLMOracle for action resolution fallback.
    """

    def __init__(
        self,
        steps: list[TrajectoryStep],
        config: RunnerConfig,
        visual_oracle: Optional[VisualOracle] = None,
        logic_oracle: Optional[LogicOracle] = None,
        diff_oracle: Optional[DiffOracle] = None,
        llm_oracle=None,
    ):
        self._steps = steps
        self._config = config
        self._visual = visual_oracle or VisualOracle()
        self._logic  = logic_oracle  or LogicOracle()
        self._diff   = diff_oracle   or DiffOracle()
        self._llm    = llm_oracle
        self._collector = FindingCollector(trajectory_id=config.trajectory_id)

    async def run(self) -> FindingCollector:
        """Execute the trajectory and return the finding collector."""
        async with BrowserSession.create(
            viewport_width=1280, viewport_height=800
        ) as session:
            try:
                await self._run_with_session(session)
            except Exception as e:
                logger.exception(
                    "Runner %s: unhandled error: %s", self._config.trajectory_id, e
                )
        return self._collector

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run_with_session(self, session: BrowserSession) -> None:
        # Navigate to the app — the trajectory steps handle everything from here.
        await session.navigate(self._config.app_url)
        await session._wait_stable()

        # Execute trajectory steps.
        for step in self._steps:
            await self._execute_step(session, step)

        # Multi-viewport visual regression.
        await self._run_viewport_checks(session)

    async def _execute_step(
        self, session: BrowserSession, step: TrajectoryStep
    ) -> None:
        context = f"[{self._config.trajectory_id}] {step.action}"
        logger.info("Runner: %s", context)

        # Capture BEFORE state.
        before = await session.capture_state()

        # Run visual oracle on the before state.
        visual_findings = await self._visual.check(session, step_context=context)
        for f in visual_findings:
            self._collector.add(f)

        # Try action.
        action_succeeded = await self._try_action(session, step, context)
        if not action_succeeded:
            return

        # Capture AFTER state.
        after = await session.capture_state()

        # Diff oracle.
        expect_change = any(
            kw in step.action.lower()
            for kw in self._config.expect_change_actions
        )
        diff_finding = await self._diff.check(before, after, step.action, expect_change)
        if diff_finding:
            self._collector.add(diff_finding)

        # Logic oracle — crash check after every action.
        crash = self._logic.check_no_crash(after, step.action)
        if crash:
            self._collector.add(crash)

        # Visual oracle on the after state.
        post_visual = await self._visual.check(session, step_context=f"After: {context}")
        for f in post_visual:
            self._collector.add(f)

    async def _try_action(
        self, session: BrowserSession, step: TrajectoryStep, context: str
    ) -> bool:
        """
        Execute all steps of the workflow in sequence.

        If any step fails, falls back to LLM to resolve the action.
        Reports a finding if the action cannot be completed at all.
        """
        # Primary: execute the multi-step workflow from the BFS graph.
        if step.steps:
            if await self._execute_steps(session, step.steps):
                return True
            logger.debug("Runner: workflow steps failed for %s", step.action)

        # Fallback: ask LLM to find the right element to click.
        if self._llm:
            state    = await session.capture_state()
            elements = await session.get_interactive_elements()
            try:
                if await self._llm_resolve_action(session, step, state, elements):
                    return True
            except Exception as e:
                logger.debug("Runner: LLM resolution failed for %s: %s", step.action, e)

        # Report as a finding — element/workflow missing or inaccessible.
        self._collector.add(Finding(
            id=f"MISS-{step.action[:8]}",
            title=f"UI element not found for action: '{step.action}'",
            bug_type=BugType.LOGIC,
            severity=Severity.HIGH,
            steps=[context],
            detected_by=DetectedBy.HEURISTIC,
            reasoning=(
                f"Could not execute workflow '{step.action}' "
                f"({len(step.steps)} step(s)). "
                f"Elements may be missing or inaccessible."
            ),
        ))
        return False

    async def _execute_steps(
        self, session: BrowserSession, steps: list[dict]
    ) -> bool:
        """Execute a sequence of interaction steps. Returns False on first failure."""
        for s in steps:
            try:
                t   = s.get("type", "click")
                sel = s.get("selector", "")
                val = s.get("value", "")
                if t == "fill":
                    await session.fill(sel, val)
                elif t == "click":
                    await session.click(sel)
                elif t == "press":
                    await session.press(sel, val)
                elif t == "select":
                    await session.fill(sel, val)
            except Exception as e:
                logger.debug("Runner: step %s/%s failed: %s", t, sel, e)
                return False
        return True

    async def _llm_resolve_action(
        self, session, step, state, elements
    ) -> bool:
        """Fall back to LLMOracle to find a single element to click for this action."""
        selector = await self._llm.resolve_action(
            state.screenshot, elements, step.action
        )
        if selector:
            try:
                await session.click(selector)
                return True
            except Exception:
                pass
        return False

    async def _run_viewport_checks(self, session: BrowserSession) -> None:
        """Re-run visual checks at each viewport size."""
        for width, height in VIEWPORTS:
            await session.set_viewport(width, height)
            context = f"Viewport {width}x{height}"
            findings = await self._visual.check(session, step_context=context)
            for f in findings:
                self._collector.add(f)
        # Restore default viewport.
        await session.set_viewport(1280, 800)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def steps_from_trajectory(trajectory: dict) -> list[TrajectoryStep]:
    """Convert a trajectory dict (from trajectory_extractor) to TrajectoryStep list."""
    result = []
    for s in trajectory.get("steps", []):
        raw_steps = s.get("steps", [])
        # Backwards compat: old trajectories store a single selector string.
        if not raw_steps and s.get("selector"):
            raw_steps = [{"type": "click", "selector": s["selector"], "value": ""}]
        result.append(TrajectoryStep(
            action=s["action"],
            steps=raw_steps,
            from_hash=s.get("from_hash", ""),
            to_hash=s.get("to_hash", ""),
            description=s.get("description", ""),
        ))
    return result
