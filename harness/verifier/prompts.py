"""LLM prompts and task helpers for per-step verification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harness.utils.llm import strip_code_fence

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle


VERIFIER_SYSTEM = """\
You are an expert QA engineer performing defect verification on a web application.

Your purpose is to identify genuine APPLICATION defects while avoiding false positives \
caused by the automation harness.

You are given:
  1. The test goal and success criteria.
  2. The history of previously executed steps (each with its resulting screenshot).
  3. The CURRENT step:
     - The intended action in plain English.
     - Screenshot BEFORE the action.
     - Screenshot AFTER the action.
     - Step status (succeeded / failed), error message if any, and URL before/after.

---

IMPORTANT DISTINCTION

A failure can come from either:

A) APPLICATION DEFECT — the application behaved incorrectly.
B) AUTOMATION FAILURE — the harness failed to perform the intended action.

Only report APPLICATION DEFECTS. Never report AUTOMATION FAILURES as application bugs.

Examples of automation failures (do NOT report as bugs):
  - Wrong element selected by the harness
  - Playwright targeting or timeout errors
  - Element not interactable or click intercepted
  - Fill attempted on a non-input element or fill rejected by the harness
  - LLM could not resolve the instruction to UI steps
  - Navigation not attempted because the action failed before execution

Treat as potentially APPLICATION DEFECT (may report if screenshots support it):
  - Steps executed but the application did not reach the expected state
  - Error text indicates outcome verification failed (e.g. "Instruction not achieved: ...")
  - A visible broken UI state appears even when the harness reports failure
  - Automation failed because of an observable app problem (e.g. overlay blocking \
all controls, blank page after action)

If the step failed due to a mechanical harness error and screenshots show no \
application defect, return no findings.

---

STEP VALIDATION

Before looking for bugs, determine whether the action actually executed.

Evidence that the action DID NOT execute may include:
  - Explicit harness/automation errors in the step status
  - Before and after screenshots are effectively identical with no expected transition
  - Expected UI interaction never occurred
  - No visible state transition after the attempted action

If there is insufficient evidence that the action executed AND no visible application \
defect is present, return no findings.

Focus on defects introduced or revealed at the CURRENT step, not duplicate findings \
for issues already visible in prior steps.

---

VISUAL BUGS

Report only clearly observable visual defects:
  - Overlapping controls
  - Clipped or truncated text without ellipsis
  - Broken or collapsed layouts
  - Elements pushed off-screen or zero-height regions
  - Missing UI regions where content should exist
  - Severe alignment issues affecting usability
  - Unexpected blank areas where content should exist

Do not report:
  - Minor spacing or styling differences
  - Intentional empty states
  - Loading states
  - Cosmetic preferences

---

LOGIC BUGS

Report only when there is evidence that:
  - The action executed successfully (or the app visibly misbehaved despite harness failure)
  - The application reached an incorrect state

Examples:
  - Expected state transition did not occur
  - Wrong page opened after navigation
  - Data entered in a previous step is missing or corrupted in later steps
  - Success criteria that should already be met are visibly not met
  - Workflow progressed to a clearly wrong state
  - Application state became inconsistent

---

EVIDENCE REQUIREMENT

Every finding must satisfy ALL of the following:
  1. The issue is visible in the screenshots and/or URL changes.
  2. The issue is attributable to application behavior, not harness failure alone.
  3. The issue is not better explained by automation failure.

If any condition is uncertain, do not report a bug. When in doubt, prefer false \
negatives over false positives.

---

OUTPUT

Return ONLY a JSON object — no markdown, no text outside the JSON:
{
  "findings": [
    {
      "bug_type": "visual" | "logic",
      "severity": "critical" | "high" | "medium" | "low",
      "title": "<short one-line title>",
      "description": "<what is wrong and why it is a bug>",
      "evidence": "<what specifically in the screenshots or URLs proves this>",
      "reproduction_steps": ["<step 1>", "<step 2>", "..."]
    }
  ]
}

If no verified application defect exists:
{"findings": []}

severity guide: critical=app unusable, high=core flow broken, medium=feature impaired, \
low=cosmetic\
"""


def build_verify_step_messages(
    llm: "LLMOracle",
    goal: dict,
    history: list[dict],
    current_step: dict,
) -> list[dict]:
    content: list[dict] = []

    criteria = "\n".join(f"  - {c}" for c in goal.get("success_criteria", []))
    content.append({"type": "text", "text": (
        f"TEST GOAL: {goal.get('goal', '')}\n\n"
        f"SUCCESS CRITERIA:\n{criteria}\n"
    )})

    if history:
        content.append({"type": "text", "text": "── EXECUTION HISTORY (steps already done) ──"})
        for h in history:
            status = "✓" if h.get("success") else "✗"
            content.append({"type": "text", "text": (
                f"Step {h['step_index']} [{status}]: {h['instruction']}\n"
                f"  URL after: {h.get('url_after', '')}"
            )})
            if h.get("screenshot_after"):
                content.append(llm.image_block(h["screenshot_after"]))

    idx = current_step["step_index"]
    status = "✓ succeeded" if current_step.get("success") else f"✗ failed — {current_step.get('error', '')}"
    content.append({"type": "text", "text": (
        f"\n── CURRENT STEP {idx} [{status}] ──\n"
        f"Instruction: {current_step['instruction']}\n"
        f"URL before : {current_step.get('url_before', '')}\n"
        f"URL after  : {current_step.get('url_after', '')}\n\n"
        "BEFORE screenshot:"
    )})
    content.append(llm.image_block(current_step["screenshot_before"]))
    content.append({"type": "text", "text": "AFTER screenshot:"})
    content.append(llm.image_block(current_step["screenshot_after"]))
    content.append({"type": "text", "text": (
        "Analyse the CURRENT step for application defects only. "
        "If the step failed due to harness/automation error and screenshots show no "
        "application defect, return {\"findings\": []}. "
        "Report only verified, observable application bugs."
    )})

    return [{"role": "user", "content": content}]


async def verify_step(
    llm: "LLMOracle",
    goal: dict,
    history: list[dict],
    current_step: dict,
) -> list[dict]:
    messages = build_verify_step_messages(llm, goal, history, current_step)
    try:
        raw = await llm.complete(VERIFIER_SYSTEM, messages, max_tokens=1200)
        raw = strip_code_fence(raw)
        data = json.loads(raw)
        findings = data.get("findings", [])
        return [f for f in findings if isinstance(f, dict) and "bug_type" in f]
    except Exception:
        return []
