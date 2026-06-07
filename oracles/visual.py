"""
Visual oracle — deterministic geometry-based defect detection.

Runs three checks via the browser's JS engine:
  overflow_x   — element's content is wider than its container
  overlap      — two interactive elements share screen space (> 5px)
  viewport_clip — interactive element is rendered outside the viewport

Each violation is retried twice (500 ms apart) before being reported, to
avoid flagging transient animation frames or layout shifts.

Optionally passes confirmed screenshots to an LLMOracle for a second-opinion
visual sanity check.
"""

import asyncio
import uuid
from dataclasses import dataclass
from typing import Optional

from browser.session import BrowserSession
from models import BugType, DetectedBy, Finding, PageState, Severity


# Severity mapping for geometry violation types.
_SEVERITY_MAP = {
    "overlap":       Severity.HIGH,
    "viewport_clip": Severity.MEDIUM,
    "overflow_x":    Severity.LOW,
}


@dataclass
class _RawViolation:
    type: str
    element: str
    detail: str


class VisualOracle:
    """
    Runs deterministic geometry checks on the current page state.

    Violations are confirmed with two retries before being escalated.
    Optionally defers to an LLMOracle for screenshot-level sanity checks.
    """

    RETRY_COUNT = 2
    RETRY_DELAY_S = 0.5

    def __init__(self, llm_oracle=None):
        self._llm = llm_oracle

    async def check(
        self,
        session: BrowserSession,
        step_context: str = "",
    ) -> list[Finding]:
        """
        Run all visual checks on the current page.
        Returns a list of confirmed Finding objects.
        """
        violations = await self._confirmed_violations(session)
        findings = [self._to_finding(v, step_context) for v in violations]

        # Optional LLM screenshot sanity pass.
        if self._llm and violations:
            state = await session.capture_state()
            verdict = await self._llm.judge_screenshot(
                state.screenshot,
                context=step_context or "General visual check",
            )
            if verdict.verdict == "bug":
                findings.append(
                    Finding(
                        id=f"VIS-LLM-{uuid.uuid4().hex[:6]}",
                        title=verdict.description,
                        bug_type=BugType.VISUAL,
                        severity=Severity(verdict.severity or "medium"),
                        steps=[step_context] if step_context else [],
                        detected_by=DetectedBy.LLM,
                        reasoning=verdict.reasoning,
                    )
                )

        return findings

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _confirmed_violations(
        self, session: BrowserSession
    ) -> list[_RawViolation]:
        """
        A violation must appear in ALL retry attempts to be confirmed.
        This filters out transient animation frames and layout shifts.
        """
        all_runs: list[list[dict]] = []
        for i in range(self.RETRY_COUNT + 1):
            if i > 0:
                await asyncio.sleep(self.RETRY_DELAY_S)
            run = await session.get_geometry_violations()
            all_runs.append(run)

        # Keep only violations that appeared in every run.
        def key(v: dict) -> str:
            return f"{v['type']}::{v['element']}"

        first_keys = {key(v): v for v in all_runs[0]}
        confirmed = []
        for k, v in first_keys.items():
            if all(any(key(r) == k for r in run) for run in all_runs[1:]):
                confirmed.append(_RawViolation(v["type"], v["element"], v["detail"]))

        return confirmed

    def _to_finding(self, v: _RawViolation, context: str) -> Finding:
        severity = _SEVERITY_MAP.get(v.type, Severity.LOW)
        title = {
            "overflow_x":    f"Element overflow: {v.element}",
            "overlap":       f"Overlapping elements: {v.element}",
            "viewport_clip": f"Element clipped by viewport: {v.element}",
        }.get(v.type, f"Visual issue ({v.type}): {v.element}")

        steps = []
        if context:
            steps.append(context)

        return Finding(
            id=f"VIS-{uuid.uuid4().hex[:6]}",
            title=title,
            bug_type=BugType.VISUAL,
            severity=severity,
            steps=steps,
            detected_by=DetectedBy.HEURISTIC,
            reasoning=v.detail,
        )
