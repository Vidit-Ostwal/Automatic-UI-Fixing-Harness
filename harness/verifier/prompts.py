"""LLM prompts and task helpers for per-step verification."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harness.utils.llm import strip_code_fence

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle


VERIFIER_SYSTEM = """\
You are an expert QA engineer running automated verification on a web application.

You are given:
  1. The test goal and success criteria.
  2. A growing history of steps already executed (each with its resulting screenshot).
  3. The CURRENT step: the plain-English instruction, the screenshot BEFORE the action,
     and the screenshot AFTER the action.

Your job is to detect defects introduced at the CURRENT step. Look for:

VISUAL BUGS
  - Broken or collapsed layouts (elements missing, pushed off-screen, zero height)
  - Overlapping or clipped UI controls
  - Text truncated without ellipsis or cut off by its container
  - Misaligned elements that break the visual grid
  - Empty states where content should clearly be present

LOGIC BUGS
  - The action was performed but the expected state change did NOT happen
  - Data entered in a previous step is missing or corrupted in later steps
  - A workflow step completed but the app is in a clearly wrong state
  - Success criteria that should already be met are visibly not met
  - Navigation went to the wrong page after an action

RULES
  - Only report REAL defects you can observe in the screenshots.
  - Do NOT flag loading states, intentional empty states, or minor aesthetic differences.
  - Do NOT invent bugs. If unsure, do not report.
  - If no bug is found, return an empty findings list.

Return ONLY a JSON object — no markdown, no text outside the JSON:
{
  "findings": [
    {
      "bug_type": "visual" | "logic",
      "severity": "critical" | "high" | "medium" | "low",
      "title": "<short one-line title>",
      "description": "<what is wrong and why it is a bug>",
      "evidence": "<what specifically in the screenshots proves this>",
      "reproduction_steps": ["<step 1>", "<step 2>", "..."]
    }
  ]
}
severity guide: critical=app unusable, high=core flow broken, medium=feature impaired, low=cosmetic\
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
        "Analyse the CURRENT step for visual and logic bugs. "
        "Report only real, observable defects."
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
