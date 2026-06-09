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
    step_index, instruction, resolution_reason, resolved_steps,
    resolve_failure_kind, resolve_failure_detail, last_llm_preview,
    verify_explanation, retry_count, screenshot_before/after,
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

from harness.browser import BrowserSession
from harness.config import env_int
from harness.docker import DockerInstance
from harness.executor.prompts import resolve_instruction, verify_instruction_outcome
from harness.executor.step_message import QUEUE_DONE_SENTINEL, StepMessage
from harness.utils.step_runner import dismiss_overlays, execute_steps, wait_for_settle

logger = logging.getLogger("executor")


def _clip(text: str, n: int = 72) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _log_step_header(
    test_case_id: str,
    run_id: str,
    step_index: int,
    total_steps: int,
    phase: str,
    *,
    attempt: int | None = None,
    max_attempts: int | None = None,
) -> None:
    attempt_s = ""
    if attempt is not None and max_attempts is not None:
        attempt_s = f"  attempt {attempt}/{max_attempts}"
    logger.info(
        "[%s|%s] step %d/%d%s  ── %s",
        test_case_id, run_id, step_index + 1, total_steps, attempt_s, phase,
    )


def _log_step_detail(key: str, value: str) -> None:
    logger.info("  %-12s %s", key + ":", value)


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
            "[%s|%s] START  %d instruction(s)",
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
                                "[%s|%s] step %d/%d  FAILED (continuing)  →  %s",
                                self._test_case_id, self._run_id,
                                step_index + 1, len(instructions),
                                record.get("error") or "unknown error",
                            )
                            if record.get("resolve_failure_detail"):
                                _log_step_detail(
                                    "resolve",
                                    f"{record.get('resolve_failure_kind', '')}: "
                                    f"{record['resolve_failure_detail']}",
                                )
                            if record.get("last_llm_preview"):
                                _log_step_detail("llm", record["last_llm_preview"])

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
            "[%s|%s] DONE  %s  (%d/%d steps ok)",
            self._test_case_id, self._run_id, final_state.upper(),
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
        total_steps = len(self._goal.get("instructions", []))
        max_retries = env_int("EXECUTOR_INSTRUCTION_RETRIES")
        max_attempts = max_retries + 1

        _log_step_header(
            self._test_case_id, self._run_id, step_index, total_steps, "INSTRUCTION",
        )
        _log_step_detail("text", _clip(instruction, 100))

        # ── dismiss any open overlays (same as BFS explorer) ────────
        await dismiss_overlays(session)

        # ── before screenshot + state ────────────────────────────────
        before_state = await session.capture_state()
        before_png   = before_state.screenshot

        # ── resolve → execute → verify, with retries on failure ─────
        resolved_steps: list[dict] = []
        resolution_reason = ""
        resolve_failure_kind = ""
        resolve_failure_detail = ""
        last_llm_preview = ""
        verify_explanation = ""
        retry_count = 0
        success = False
        error: Optional[str] = None

        if self._llm is None:
            error = "no LLM oracle — cannot resolve instruction"
        else:
            previous_attempts: list[dict] = []

            for attempt in range(max_attempts):
                attempt_num = attempt + 1
                await dismiss_overlays(session)
                attempt_before = await session.capture_state()
                elements = await session.get_interactive_elements()

                _log_step_header(
                    self._test_case_id, self._run_id, step_index, total_steps,
                    "RESOLVE", attempt=attempt_num, max_attempts=max_attempts,
                )
                _log_step_detail("url", attempt_before.url)
                _log_step_detail("elements", str(len(elements)))

                outcome = await resolve_instruction(
                    self._llm,
                    attempt_before.screenshot,
                    elements,
                    instruction,
                    a11y_tree=attempt_before.a11y_tree,
                    previous_attempts=previous_attempts,
                )
                last_llm_preview = outcome.raw_preview

                if not outcome.ok:
                    resolve_failure_kind = outcome.failure_kind
                    resolve_failure_detail = outcome.failure_detail
                    error = outcome.summary()
                    previous_attempts.append({
                        "reason": outcome.result.reason if outcome.result else "",
                        "steps": [],
                        "failure": error,
                    })
                    logger.warning(
                        "[%s|%s] step %d/%d  RESOLVE FAILED  →  %s",
                        self._test_case_id, self._run_id,
                        step_index + 1, total_steps, error,
                    )
                    if outcome.raw_preview:
                        _log_step_detail("llm", outcome.raw_preview)
                    if attempt < max_retries:
                        retry_count += 1
                        continue
                    break

                resolution_reason = outcome.result.reason
                resolved_steps = outcome.result.steps
                _log_step_detail("reason", _clip(resolution_reason or "(none)", 100))
                _log_step_detail("steps", str(len(resolved_steps)))

                _log_step_header(
                    self._test_case_id, self._run_id, step_index, total_steps,
                    "EXECUTE", attempt=attempt_num, max_attempts=max_attempts,
                )
                step_success, step_error, _ = await execute_steps(
                    session, resolved_steps, self._llm
                )
                await wait_for_settle(session, prev_url=attempt_before.url)
                attempt_after = await session.capture_state()

                if not step_success:
                    previous_attempts.append({
                        "reason": resolution_reason,
                        "steps": resolved_steps,
                        "failure": step_error,
                    })
                    error = step_error or "step execution failed"
                    logger.warning(
                        "[%s|%s] step %d/%d  EXECUTE FAILED  →  %s",
                        self._test_case_id, self._run_id,
                        step_index + 1, total_steps, error,
                    )
                    if attempt < max_retries:
                        retry_count += 1
                        continue
                    break

                _log_step_header(
                    self._test_case_id, self._run_id, step_index, total_steps,
                    "VERIFY OUTCOME", attempt=attempt_num, max_attempts=max_attempts,
                )
                achieved, verify_explanation = await verify_instruction_outcome(
                    self._llm,
                    instruction,
                    attempt_before.screenshot,
                    attempt_after.screenshot,
                    resolution_reason,
                    a11y_before=attempt_before.a11y_tree,
                    a11y_after=attempt_after.a11y_tree,
                    execution_error=step_error or None,
                )
                if achieved:
                    success = True
                    error = None
                    resolve_failure_kind = ""
                    resolve_failure_detail = ""
                    logger.info(
                        "[%s|%s] step %d/%d  OK  →  %s",
                        self._test_case_id, self._run_id,
                        step_index + 1, total_steps,
                        _clip(verify_explanation or "instruction achieved", 100),
                    )
                    break

                failure = f"outcome_not_achieved: {verify_explanation}"
                previous_attempts.append({
                    "reason": resolution_reason,
                    "steps": resolved_steps,
                    "failure": failure,
                })
                error = verify_explanation
                logger.warning(
                    "[%s|%s] step %d/%d  VERIFY FAILED  →  %s",
                    self._test_case_id, self._run_id,
                    step_index + 1, total_steps, verify_explanation,
                )
                if attempt < max_retries:
                    retry_count += 1
                    continue
                break

        # ── wait for UI to settle before final capture ───────────────
        await wait_for_settle(session, prev_url=before_state.url)

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
            "step_index":             step_index,
            "instruction":            instruction,
            "resolution_reason":      resolution_reason,
            "resolved_steps":         resolved_steps,
            "resolve_failure_kind":   resolve_failure_kind,
            "resolve_failure_detail": resolve_failure_detail,
            "last_llm_preview":       last_llm_preview,
            "verify_explanation":     verify_explanation,
            "retry_count":            retry_count,
            "screenshot_before":      before_rel,
            "screenshot_after":       after_rel,
            "url_before":             before_state.url,
            "url_after":              after_state.url,
            "success":                success,
            "error":                  error,
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
