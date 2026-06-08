"""LLM prompts and task helpers for oracle visual/diff checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness.models import PageState
from harness.oracles.llm import OracleVerdict, parse_verdict
from harness.utils.llm import strip_code_fence

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle


VISUAL_SYSTEM = """\
You are a UI quality oracle analysing web application screenshots for defects.

Rules:
- IGNORE loading spinners, skeleton screens, and intentional empty states.
- IGNORE minor aesthetic differences that do not affect usability.
- ONLY flag clear defects: broken layouts, overlapping controls, clipped text,
  elements rendering outside their container, or missing elements that should
  clearly be present given the context.

Respond with ONLY a JSON object — no markdown, no explanation outside the JSON:
{
  "verdict":     "bug" | "ok" | "noise",
  "description": "<one sentence>",
  "severity":    "critical" | "high" | "medium" | "low" | null,
  "reasoning":   "<brief explanation>"
}
severity must be null when verdict is not "bug".\
"""

DIFF_SYSTEM = """\
You are a UI quality oracle. You will receive two screenshots labelled BEFORE
and AFTER, plus the action that was performed between them.

Determine whether the UI change is:
  "ok"    — the change is expected and correct for the given action
  "bug"   — the change is unexpected, missing, or wrong
  "noise" — a loading/transition state; not enough information to judge

Respond with ONLY a JSON object:
{
  "verdict":     "bug" | "ok" | "noise",
  "description": "<one sentence>",
  "severity":    "critical" | "high" | "medium" | "low" | null,
  "reasoning":   "<brief explanation>"
}
severity must be null when verdict is not "bug".\
"""


def build_visual_messages(llm: "LLMOracle", screenshot: bytes, context: str = "") -> list[dict]:
    text = "Analyse this screenshot for UI defects."
    if context:
        text += f" Context: {context}"
    return [
        {
            "role": "user",
            "content": [
                llm.image_block(screenshot),
                {"type": "text", "text": text},
            ],
        }
    ]


def build_diff_messages(
    llm: "LLMOracle",
    before: PageState,
    after: PageState,
    action: str,
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "BEFORE screenshot:"},
                llm.image_block(before.screenshot),
                {"type": "text", "text": "AFTER screenshot:"},
                llm.image_block(after.screenshot),
                {"type": "text", "text": f"Action performed: {action}"},
            ],
        }
    ]


async def judge_screenshot(
    llm: "LLMOracle",
    screenshot: bytes,
    context: str = "",
) -> OracleVerdict:
    messages = build_visual_messages(llm, screenshot, context)
    raw = await llm.complete(VISUAL_SYSTEM, messages)
    return parse_verdict(raw)


async def judge_diff(
    llm: "LLMOracle",
    before: PageState,
    after: PageState,
    action: str,
) -> OracleVerdict:
    messages = build_diff_messages(llm, before, after, action)
    raw = await llm.complete(DIFF_SYSTEM, messages)
    return parse_verdict(raw)
