"""
GoalExecutor — executes one test goal from trajectories_goal.json.

Input is ONLY the goal dict (id, goal, instructions, success_criteria).
No trajectories.json is consulted — the executor is fully LLM-driven:
for each plain-English instruction it looks at the current page, asks the
LLM to resolve the instruction to concrete UI steps, and executes them.

Output layout
-------------
output/
  executor_runs/
    T-001_<run_id>/
      run.json
      step_00_before.png
      step_00_after.png
      step_01_before.png
      step_01_after.png
      ...

run.json fields
---------------
  test_case_id, run_id, goal, description, instructions, success_criteria
  steps[]
    step_index, instruction, resolved_steps, screenshot_before/after,
    url_before/after, success, error
  steps_executed, steps_succeeded, final_state, completed, error
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from browser.session import BrowserSession
from docker_manager import DockerInstance
from executor.step_message import QUEUE_DONE_SENTINEL, StepMessage
from utils.step_runner import dismiss_overlays, execute_steps, wait_for_navigation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ExecutorResult:
    test_case_id: str
    run_id: str
    completed: bool
    steps_executed: int
    steps_succeeded: int
    final_state: str        # "success" | "partial" | "failed"
    run_dir: Path
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# GoalExecutor
# ---------------------------------------------------------------------------

class GoalExecutor:
    """
    Executes one goal against a fresh browser instance.

    Parameters
    ----------
    goal        Entry from trajectories_goal.json.
                Required keys: id, instructions.
                Optional: goal, description, success_criteria.
    step_queue  Shared asyncio.Queue[StepMessage].  One message per instruction,
                plus a QUEUE_DONE_SENTINEL when the executor finishes.
    output_dir  Root output directory.  A run subdir is created here.
    llm_oracle  LLM oracle used to resolve plain-English instructions to UI
                steps.  When None, every instruction is skipped (no steps).
    """

    def __init__(
        self,
        goal: dict,
        step_queue: "asyncio.Queue[StepMessage]",
        output_dir: Path,
        llm_oracle=None,
    ) -> None:
        self._goal         = goal
        self._queue        = step_queue
        self._output_dir   = output_dir
        self._llm          = llm_oracle
        self._run_id       = uuid.uuid4().hex[:10]
        self._test_case_id = goal.get("id", "T-???")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> ExecutorResult:
        run_dir = (
            self._output_dir / "executor_runs"
            / f"{self._test_case_id}_{self._run_id}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        instructions: list[str] = self._goal.get("instructions", [])
        logger.info(
            "GoalExecutor  %s / run=%s  starting  (%d instruction(s))",
            self._test_case_id, self._run_id, len(instructions),
        )

        step_records:   list[dict] = []
        steps_executed  = 0
        steps_succeeded = 0
        final_state     = "failed"
        exec_error: Optional[str] = None

        try:
            async with DockerInstance.start() as docker:
                async with BrowserSession.create() as session:
                    # Navigate to the app — same starting point as BFS explorer.
                    await session.navigate(docker.url)
                    await session._wait_stable()

                    # Execute each instruction against the live page.
                    for step_index, instruction in enumerate(instructions):
                        record, msg = await self._execute_instruction(
                            session, step_index, instruction, run_dir
                        )
                        step_records.append(record)
                        await self._queue.put(msg)

                        steps_executed += 1
                        if record["success"]:
                            steps_succeeded += 1
                        else:
                            logger.warning(
                                "GoalExecutor %s step %d failed — continuing  (%s)",
                                self._test_case_id, step_index,
                                (record.get("error") or "")[:80],
                            )

                    if steps_succeeded == steps_executed and steps_executed > 0:
                        final_state = "success"
                    elif steps_succeeded > 0:
                        final_state = "partial"

        except Exception as exc:
            exec_error = str(exc)
            logger.exception(
                "GoalExecutor %s: unhandled error: %s", self._test_case_id, exc
            )
        finally:
            await self._queue.put(
                StepMessage(
                    test_case_id=self._test_case_id,
                    run_id=self._run_id,
                    step_index=-1,
                    action_name=QUEUE_DONE_SENTINEL,
                    action_description="",
                    screenshot_before=b"",
                    screenshot_after=b"",
                    url_before="",
                    url_after="",
                    a11y_before={},
                    a11y_after={},
                    success=final_state == "success",
                    error=exec_error,
                )
            )

        # ------------------------------------------------------------------
        # Write run.json
        # ------------------------------------------------------------------
        run_record = {
            "test_case_id":     self._test_case_id,
            "run_id":           self._run_id,
            "goal":             self._goal.get("goal", ""),
            "description":      self._goal.get("description", ""),
            "instructions":     instructions,
            "success_criteria": self._goal.get("success_criteria", []),
            "steps":            step_records,
            "steps_executed":   steps_executed,
            "steps_succeeded":  steps_succeeded,
            "final_state":      final_state,
            "completed":        final_state == "success",
            "error":            exec_error,
        }
        (run_dir / "run.json").write_text(json.dumps(run_record, indent=2))

        logger.info(
            "GoalExecutor  %s / run=%s  done — %s  (%d/%d steps succeeded)",
            self._test_case_id, self._run_id, final_state,
            steps_succeeded, steps_executed,
        )

        return ExecutorResult(
            test_case_id=self._test_case_id,
            run_id=self._run_id,
            completed=final_state == "success",
            steps_executed=steps_executed,
            steps_succeeded=steps_succeeded,
            final_state=final_state,
            run_dir=run_dir,
            error=exec_error,
        )

    # ------------------------------------------------------------------
    # Instruction execution
    # ------------------------------------------------------------------

    async def _execute_instruction(
        self,
        session: BrowserSession,
        step_index: int,
        instruction: str,
        run_dir: Path,
    ) -> tuple[dict, StepMessage]:
        logger.info(
            "GoalExecutor %s step %d: %s",
            self._test_case_id, step_index, instruction[:80],
        )

        # ── dismiss any open overlays (same as BFS explorer) ────────
        await dismiss_overlays(session)

        # ── before screenshot + state ────────────────────────────────
        before_state = await session.capture_state()
        before_png   = before_state.screenshot

        # ── resolve plain-English instruction → UI steps ────────────
        resolved_steps: list[dict] = []
        success = False
        error: Optional[str] = None

        if self._llm is None:
            error = "no LLM oracle — cannot resolve instruction"
        else:
            elements = await session.get_interactive_elements()
            from executor.prompts import resolve_instruction

            resolved_steps = await resolve_instruction(
                self._llm,
                before_png, elements, instruction
            ) or []
            if not resolved_steps:
                error = "LLM could not resolve instruction to any UI steps"

        # ── execute via shared step_runner (identical to BFS explorer)
        # fill verification + LLM fill-value fallback + exception handling
        if resolved_steps:
            success, error, _ = await execute_steps(
                session, resolved_steps, self._llm
            )

        # ── wait for SPA navigation to settle (same as BFS explorer) ─
        await wait_for_navigation(session, before_state.url)

        # ── after screenshot + state ─────────────────────────────────
        after_state = await session.capture_state()
        after_png   = after_state.screenshot

        # ── save PNGs ────────────────────────────────────────────────
        before_path = run_dir / f"step_{step_index:02d}_before.png"
        after_path  = run_dir / f"step_{step_index:02d}_after.png"
        before_path.write_bytes(before_png)
        after_path.write_bytes(after_png)

        before_rel = str(before_path.relative_to(self._output_dir))
        after_rel  = str(after_path.relative_to(self._output_dir))

        # ── build record + queue message ─────────────────────────────
        step_record = {
            "step_index":        step_index,
            "instruction":       instruction,
            "resolved_steps":    resolved_steps,
            "screenshot_before": before_rel,
            "screenshot_after":  after_rel,
            "url_before":        before_state.url,
            "url_after":         after_state.url,
            "success":           success,
            "error":             error,
        }

        step_msg = StepMessage(
            test_case_id=self._test_case_id,
            run_id=self._run_id,
            step_index=step_index,
            action_name=instruction[:60],
            action_description=instruction,
            screenshot_before=before_png,
            screenshot_after=after_png,
            url_before=before_state.url,
            url_after=after_state.url,
            a11y_before=before_state.a11y_tree,
            a11y_after=after_state.a11y_tree,
            success=success,
            error=error,
            screenshot_before_path=before_rel,
            screenshot_after_path=after_rel,
        )

        return step_record, step_msg
