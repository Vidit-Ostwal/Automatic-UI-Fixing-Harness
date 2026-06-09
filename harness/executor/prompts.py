"""LLM prompts and task helpers for goal-driven instruction resolution."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from harness.utils.llm import strip_code_fence
from harness.utils.page_context import format_a11y_tree, format_element_list

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle

logger = logging.getLogger("executor.resolve")

_RAW_PREVIEW_MAX = 240


@dataclass
class ResolveResult:
    reason: str
    steps: list[dict]


@dataclass
class ResolveOutcome:
    """Result of one resolve_instruction LLM call, with failure diagnostics."""

    result: ResolveResult | None = None
    failure_kind: str = ""
    failure_detail: str = ""
    raw_preview: str = ""
    element_count: int = 0

    @property
    def ok(self) -> bool:
        return self.result is not None and bool(self.result.steps)

    def summary(self) -> str:
        if self.ok:
            return f"resolved {len(self.result.steps)} step(s)"
        if self.failure_detail:
            return f"{self.failure_kind}: {self.failure_detail}"
        return self.failure_kind or "unknown failure"


def _preview(raw: str, max_len: int = _RAW_PREVIEW_MAX) -> str:
    text = " ".join(raw.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


RESOLVE_INSTRUCTION_SYSTEM = (
    "You are a precise UI automation assistant. "
    "You translate one plain-English instruction into the minimal "
    "sequence of browser interactions needed to execute it on the CURRENT page.\n\n"
    "You receive the same page context as the BFS exploration planner: "
    "a screenshot, accessibility tree, and interactive element list with real selectors.\n\n"
    "CRITICAL — selectors:\n"
    "- Every step's \"selector\" MUST be copied verbatim from the element list.\n"
    "- Do NOT invent, guess, or modify selectors. A wrong selector causes a timeout.\n\n"
    "Step types: fill | click | press | select\n"
    "For press, value is the key name (e.g. \"Enter\").\n"
    "For fill/select, infer realistic values from the instruction and field labels.\n\n"
    "First explain your reasoning, then return the steps. "
    "Reply with ONLY a JSON object."
)


VERIFY_INSTRUCTION_SYSTEM = (
    "You are a precise UI automation assistant. "
    "You judge whether a plain-English instruction was successfully "
    "carried out on a web page, using before/after screenshots and "
    "accessibility trees (same context as BFS exploration). "
    "Reply with ONLY a JSON object."
)


def _format_previous_attempts(previous_attempts: list[dict]) -> str:
    if not previous_attempts:
        return ""
    lines = ["Previous attempts that did NOT achieve the instruction:"]
    for i, attempt in enumerate(previous_attempts, 1):
        lines.append(f"\nAttempt {i}:")
        lines.append(f"  Reason: {attempt.get('reason', '')}")
        lines.append(f"  Steps: {json.dumps(attempt.get('steps', []))}")
        lines.append(f"  Failure: {attempt.get('failure', '')}")
    lines.append(
        "\nTry a different approach on the CURRENT page. "
        "Do not repeat the same steps that already failed."
    )
    return "\n".join(lines)


def build_resolve_instruction_messages(
    llm: "LLMOracle",
    screenshot: bytes,
    elements: list[dict],
    instruction: str,
    a11y_tree: dict | None = None,
    previous_attempts: list[dict] | None = None,
) -> list[dict]:
    tree_json = format_a11y_tree(a11y_tree)
    element_list = format_element_list(elements)
    retry_context = _format_previous_attempts(previous_attempts or [])
    prompt = (
        f"Instruction to execute: \"{instruction}\"\n\n"
        f"Accessibility tree:\n{tree_json}\n\n"
        f"Interactive elements with REAL selectors (use ONLY these):\n"
        f"{element_list}\n"
    )
    if retry_context:
        prompt += f"\n{retry_context}\n"
    prompt += (
        "\nReturn a JSON object with:\n"
        "  reason — why these steps will carry out the instruction on this page\n"
        "  steps  — the exact sequence of UI interactions\n\n"
        "Reply with ONLY a JSON object — no markdown, no explanation:\n"
        '{\n'
        '  "reason": "<why these steps achieve the instruction>",\n'
        '  "steps": [\n'
        '    {"type": "fill",  "selector": "<selector from element list>", "value": "<text to enter>"},\n'
        '    {"type": "click", "selector": "<selector from element list>"}\n'
        "  ]\n"
        "}\n"
        'If the instruction cannot be carried out on this page, return: {"reason": "...", "steps": []}'
    )
    return [
        {
            "role": "user",
            "content": [
                llm.image_block(screenshot),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_verify_instruction_messages(
    llm: "LLMOracle",
    instruction: str,
    screenshot_before: bytes,
    screenshot_after: bytes,
    resolution_reason: str,
    a11y_before: dict | None = None,
    a11y_after: dict | None = None,
    execution_error: str | None = None,
) -> list[dict]:
    prompt = (
        f"Instruction that was attempted: \"{instruction}\"\n\n"
        f"Why these steps were chosen:\n{resolution_reason}\n\n"
        f"Accessibility tree BEFORE:\n{format_a11y_tree(a11y_before)}\n\n"
        f"Accessibility tree AFTER:\n{format_a11y_tree(a11y_after)}\n"
    )
    if execution_error:
        prompt += f"\nExecution error (if any): {execution_error}\n"
    prompt += (
        "\nThe first image is BEFORE the steps. The second is AFTER.\n"
        "Was the instruction successfully carried out?\n\n"
        "Reply with ONLY a JSON object — no markdown:\n"
        '{\n'
        '  "achieved": true | false,\n'
        '  "explanation": "<what changed and whether it satisfies the instruction>"\n'
        "}"
    )
    return [
        {
            "role": "user",
            "content": [
                llm.image_block(screenshot_before),
                llm.image_block(screenshot_after),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _parse_resolve_response(raw: str) -> tuple[ResolveResult | None, str, str]:
    """
    Parse LLM resolve response.

    Returns (result, failure_kind, failure_detail).
    failure_kind/failure_detail are empty when result is not None.
    """
    try:
        cleaned = strip_code_fence(raw)
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, "json_parse_error", f"{exc.msg} at pos {exc.pos}"
    except Exception as exc:
        return None, "parse_error", str(exc)

    if isinstance(parsed, list):
        if parsed:
            return ResolveResult(reason="", steps=parsed), "", ""
        return None, "empty_steps", "LLM returned empty JSON array"

    if isinstance(parsed, dict):
        steps = parsed.get("steps", [])
        reason = str(parsed.get("reason", "")).strip()
        if not isinstance(steps, list):
            return None, "invalid_shape", f'"steps" is {type(steps).__name__}, expected array'
        if not steps:
            detail = reason or "LLM returned empty steps array"
            return None, "empty_steps", detail
        return ResolveResult(reason=reason, steps=steps), "", ""

    return None, "invalid_shape", f"expected object or array, got {type(parsed).__name__}"


async def resolve_instruction(
    llm: "LLMOracle",
    screenshot: bytes,
    elements: list[dict],
    instruction: str,
    a11y_tree: dict | None = None,
    previous_attempts: list[dict] | None = None,
) -> ResolveOutcome:
    element_count = len(elements)
    messages = build_resolve_instruction_messages(
        llm,
        screenshot,
        elements,
        instruction,
        a11y_tree=a11y_tree,
        previous_attempts=previous_attempts,
    )
    try:
        raw = await llm.complete(RESOLVE_INSTRUCTION_SYSTEM, messages, max_tokens=600)
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ResolveOutcome(
            failure_kind="llm_error",
            failure_detail=str(exc),
            element_count=element_count,
        )

    preview = _preview(raw)
    result, failure_kind, failure_detail = _parse_resolve_response(raw)
    if result is not None:
        return ResolveOutcome(
            result=result,
            raw_preview=preview,
            element_count=element_count,
        )
    return ResolveOutcome(
        failure_kind=failure_kind,
        failure_detail=failure_detail,
        raw_preview=preview,
        element_count=element_count,
    )


async def verify_instruction_outcome(
    llm: "LLMOracle",
    instruction: str,
    screenshot_before: bytes,
    screenshot_after: bytes,
    resolution_reason: str,
    a11y_before: dict | None = None,
    a11y_after: dict | None = None,
    execution_error: str | None = None,
) -> tuple[bool, str]:
    messages = build_verify_instruction_messages(
        llm,
        instruction,
        screenshot_before,
        screenshot_after,
        resolution_reason,
        a11y_before=a11y_before,
        a11y_after=a11y_after,
        execution_error=execution_error,
    )
    try:
        raw = await llm.complete(VERIFY_INSTRUCTION_SYSTEM, messages, max_tokens=300)
        raw = strip_code_fence(raw)
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            achieved = bool(parsed.get("achieved"))
            explanation = str(parsed.get("explanation", "")).strip()
            return achieved, explanation
    except Exception:
        logger.debug("verify_instruction_outcome: failed to parse LLM response", exc_info=True)
    return False, "could not verify instruction outcome"
