"""
VerifierAgent — inspects each executor step for visual and logic bugs.

Called once per StepMessage from the executor queue.  The agent maintains
a growing context window of previous steps so each LLM call can reason
about the full trajectory, not just the isolated current screenshot.

Output
------
output/verifier_claims/<test_case_id>_<run_id>/
  claims.json          — all findings for this run, structured report
  step_<N>_before.png  — symlink / copy of the executor screenshot (for convenience)
  step_<N>_after.png

claims.json schema
------------------
{
  "test_case_id": "T-001",
  "run_id": "abc123def4",
  "goal": "Verify that a user can ...",
  "description": "create_account → create_new_memo",
  "total_steps_verified": 5,
  "total_findings": 2,
  "findings": [
    {
      "id": "V-T001-001",
      "step_index": 2,
      "instruction": "Click the Save button ...",
      "bug_type": "logic",
      "severity": "high",
      "title": "Memo not visible after save",
      "description": "...",
      "evidence": "...",
      "reproduction_steps": ["..."],
      "screenshot_before": "executor_runs/T-001_abc123/step_02_before.png",
      "screenshot_after":  "executor_runs/T-001_abc123/step_02_after.png"
    }
  ]
}
"""

import json
import logging
from pathlib import Path
from typing import Optional

from harness.executor.step_message import StepMessage

logger = logging.getLogger(__name__)


class VerifierAgent:
    """
    Stateful verifier for one executor run.

    One instance is created per (test_case_id, run_id) by the verifier
    consumer in run_harness.py.  `verify_step` is awaited for each
    StepMessage; `finalize` writes the report.

    Parameters
    ----------
    goal        Entry from trajectories_goal.json for this test case.
    output_dir  Root output directory (verifier_claims/ is created here).
    llm_oracle  LLMOracle instance — must not be None for real analysis.
    """

    def __init__(self, goal: dict, output_dir: Path, llm_oracle) -> None:
        self._goal       = goal
        self._output_dir = output_dir
        self._llm        = llm_oracle
        self._history:  list[dict] = []   # grows after each verified step
        self._findings: list[dict] = []
        self._run_id:   Optional[str] = None
        self._test_case_id = goal.get("id", "T-???")
        self._finding_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def verify_step(self, msg: StepMessage) -> list[dict]:
        """
        Verify one step.  Appends any findings to the internal list and
        returns them so the caller can log immediately.
        """
        if self._run_id is None:
            self._run_id = msg.run_id

        current = {
            "step_index":        msg.step_index,
            "instruction":       msg.action_description or msg.action_name,
            "screenshot_before": msg.screenshot_before,
            "screenshot_after":  msg.screenshot_after,
            "url_before":        msg.url_before,
            "url_after":         msg.url_after,
            "success":           msg.success,
            "error":             msg.error,
        }

        step_findings: list[dict] = []

        if self._llm is not None:
            try:
                from harness.verifier.prompts import verify_step as llm_verify_step

                raw = await llm_verify_step(
                    self._llm,
                    goal=self._goal,
                    history=self._history,
                    current_step=current,
                )
                for f in raw:
                    self._finding_counter += 1
                    finding = {
                        "id":                f"V-{self._test_case_id}-{self._finding_counter:03d}",
                        "step_index":        msg.step_index,
                        "instruction":       current["instruction"],
                        "bug_type":          f.get("bug_type", "logic"),
                        "severity":          f.get("severity", "medium"),
                        "title":             f.get("title", ""),
                        "description":       f.get("description", ""),
                        "evidence":          f.get("evidence", ""),
                        "reproduction_steps": f.get("reproduction_steps", []),
                        "screenshot_before": msg.screenshot_before_path,
                        "screenshot_after":  msg.screenshot_after_path,
                    }
                    step_findings.append(finding)
                    self._findings.append(finding)

                if step_findings:
                    logger.info(
                        "Verifier %s step %d: %d finding(s) — %s",
                        self._test_case_id, msg.step_index, len(step_findings),
                        ", ".join(f["title"] for f in step_findings),
                    )
                else:
                    logger.debug(
                        "Verifier %s step %d: clean", self._test_case_id, msg.step_index
                    )

            except Exception as exc:
                logger.warning(
                    "Verifier %s step %d: LLM error — %s",
                    self._test_case_id, msg.step_index, exc,
                )

        # Append to history (after-screenshot only to keep future prompts lean)
        self._history.append({
            "step_index":      msg.step_index,
            "instruction":     current["instruction"],
            "screenshot_after": msg.screenshot_after,
            "url_after":       msg.url_after,
            "success":         msg.success,
        })

        return step_findings

    def finalize(self) -> Path:
        """
        Write claims.json and return its path.
        Called once when the executor sends its DONE sentinel.
        """
        run_id = self._run_id or "unknown"
        claims_dir = (
            self._output_dir / "verifier_claims"
            / f"{self._test_case_id}_{run_id}"
        )
        claims_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "test_case_id":        self._test_case_id,
            "run_id":              run_id,
            "goal":                self._goal.get("goal", ""),
            "description":         self._goal.get("description", ""),
            "total_steps_verified": len(self._history),
            "total_findings":      len(self._findings),
            "findings":            self._findings,
        }

        claims_path = claims_dir / "claims.json"
        claims_path.write_text(json.dumps(report, indent=2))

        severity_counts = _count_severities(self._findings)
        logger.info(
            "Verifier %s/%s finalized — %d finding(s)  [crit=%d high=%d med=%d low=%d]  → %s",
            self._test_case_id, run_id, len(self._findings),
            severity_counts["critical"], severity_counts["high"],
            severity_counts["medium"],  severity_counts["low"],
            claims_path,
        )
        return claims_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_severities(findings: list[dict]) -> dict:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        sev = f.get("severity", "low")
        if sev in counts:
            counts[sev] += 1
    return counts
