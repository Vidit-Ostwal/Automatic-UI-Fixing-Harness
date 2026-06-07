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
from executor.workflows.auth import signup
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
    selector: str
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
        # Step 1 — sign up.
        auth = await signup(session, self._config.app_url)
        if not auth.success:
            self._collector.add(Finding(
                id=f"AUTH-{self._config.trajectory_id}",
                title="Sign-up workflow failed",
                bug_type=BugType.LOGIC,
                severity=Severity.CRITICAL,
                steps=["Navigate to app", "Fill sign-up form"],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=auth.error or "Still on auth page after submit.",
            ))
            return

        # Step 2 — execute trajectory steps.
        for step in self._steps:
            await self._execute_step(session, step)

        # Step 3 — multi-viewport visual regression.
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
        Attempt the action. Falls back to LLM element resolution if the
        primary selector fails.  Returns True if the click succeeded.
        """
        # Primary: use the selector from the BFS graph.
        if step.selector:
            try:
                await session.click(step.selector)
                return True
            except Exception:
                logger.debug("Runner: selector failed for %s: %s", step.action, step.selector)

        # Fallback: ask LLM to find the right element.
        if self._llm:
            state = await session.capture_state()
            elements = await session.get_interactive_elements()
            try:
                resolution = await self._llm_resolve_action(
                    session, step, state, elements
                )
                if resolution:
                    return True
            except Exception as e:
                logger.debug("Runner: LLM resolution failed for %s: %s", step.action, e)

        # Report missing element as a finding.
        self._collector.add(Finding(
            id=f"MISS-{step.action[:8]}",
            title=f"UI element not found for action: '{step.action}'",
            bug_type=BugType.LOGIC,
            severity=Severity.HIGH,
            steps=[context],
            detected_by=DetectedBy.HEURISTIC,
            reasoning=(
                f"Could not locate selector '{step.selector}' for action "
                f"'{step.action}'. The element may be missing or inaccessible."
            ),
        ))
        return False

    async def _llm_resolve_action(
        self, session, step, state, elements
    ) -> bool:
        """Use LLMOracle to identify and click the correct element for an action."""
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
    return [
        TrajectoryStep(
            action=s["action"],
            selector=s.get("selector", ""),
            from_hash=s.get("from_hash", ""),
            to_hash=s.get("to_hash", ""),
            description=s.get("description", ""),
        )
        for s in trajectory.get("steps", [])
    ]
