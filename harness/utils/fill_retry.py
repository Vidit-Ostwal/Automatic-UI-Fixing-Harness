"""LLM fill-retry prompt — shared by planner BFS and goal executor via step_runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from harness.utils.llm import strip_code_fence

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle


FILL_RETRY_SYSTEM = """\
You are a UI automation assistant. A fill step failed because the value you \
provided was not accepted by the input field — the field ended up with a \
different value than what was written.

Reason about why the value was rejected (wrong format, too long, contains \
forbidden characters, wrong type, etc.) and suggest a corrected value that \
the field is likely to accept.

Reply with ONLY the raw replacement value — no quotes, no explanation, \
no markdown. Just the string to type into the field.\
"""


def build_fill_retry_messages(
    llm: "LLMOracle",
    selector: str,
    tried_value: str,
    actual_value: str,
    screenshot: bytes,
) -> list[dict]:
    prompt = (
        f"Input field selector: {selector}\n"
        f"Value we tried to fill: {tried_value!r}\n"
        f"Value the field actually contains after fill: {actual_value!r}\n\n"
        "The field rejected or modified our value. "
        "Suggest a corrected value that the field will accept."
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


async def suggest_fill_value(
    llm: "LLMOracle",
    selector: str,
    tried_value: str,
    actual_value: str,
    screenshot: bytes,
) -> str | None:
    messages = build_fill_retry_messages(llm, selector, tried_value, actual_value, screenshot)
    try:
        raw = await llm.complete(FILL_RETRY_SYSTEM, messages, max_tokens=100)
        value = raw.strip().strip("\"'")
        return value if value else None
    except Exception:
        return None
