"""LLM prompts and task helpers for goal-driven instruction resolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from utils.elements import serialize_elements
from utils.llm import strip_code_fence

if TYPE_CHECKING:
    from oracles.llm import LLMOracle


RESOLVE_INSTRUCTION_SYSTEM = (
    "You are a precise UI automation assistant. "
    "You translate one plain-English instruction into the minimal "
    "sequence of browser interactions needed to execute it. "
    "Use ONLY selectors from the provided element list. "
    "Reply with ONLY a JSON array."
)


def build_resolve_instruction_messages(
    llm: "LLMOracle",
    screenshot: bytes,
    elements: list[dict],
    instruction: str,
) -> list[dict]:
    element_list = serialize_elements(elements)
    prompt = (
        f"Instruction to execute: \"{instruction}\"\n\n"
        f"Interactive elements on the current page (use ONLY these selectors):\n"
        f"{element_list}\n\n"
        "Return the exact sequence of UI steps needed to carry out this instruction.\n"
        "Reply with ONLY a JSON array — no markdown, no explanation:\n"
        '[\n'
        '  {"type": "fill",  "selector": "<selector>", "value": "<text to enter>"},\n'
        '  {"type": "click", "selector": "<selector>"}\n'
        ']\n'
        "Step types: fill | click | press | select\n"
        "For press, value is the key name (e.g. \"Enter\").\n"
        "If the instruction cannot be carried out on this page, return: []"
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


async def resolve_instruction(
    llm: "LLMOracle",
    screenshot: bytes,
    elements: list[dict],
    instruction: str,
) -> list[dict] | None:
    messages = build_resolve_instruction_messages(llm, screenshot, elements, instruction)
    try:
        raw = await llm.complete(RESOLVE_INSTRUCTION_SYSTEM, messages, max_tokens=400)
        raw = strip_code_fence(raw)
        steps = json.loads(raw)
        if isinstance(steps, list):
            return steps or None
        return None
    except Exception:
        return None
