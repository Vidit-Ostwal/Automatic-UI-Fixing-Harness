"""LLM prompts and task helpers for planner workflow discovery and goal writing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from harness.utils.page_context import format_a11y_tree, format_element_list
from harness.utils.llm import strip_code_fence

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle


PAGE_ACTIONS_SYSTEM = """\
You are analyzing a web application page to identify distinct user workflows \
that have NOT yet been explored.

You will receive an optional section titled "Already-explored states". \
Read it carefully and DO NOT re-suggest any workflow whose name already \
appears in the "tried" list for the current URL. Instead, reason about \
what remains unexplored — edge cases, alternative paths, or features that \
have not been exercised yet.

For EACH NEW workflow, provide the COMPLETE sequence of interactions needed to
accomplish one user intention — not just the final click, but every preceding
fill/select step too.

Rules:
- Identify SEPARATE workflows for each distinct form or intent on the page.
  A search bar and a create-memo form are two different workflows, not one.
- A standalone button or link → one step: click it.
- A form → fill each field with a contextually appropriate value, then submit.
- A "create" workflow → open the dialog/page, fill required fields, submit.
- Deduplicate: if there are 10 identical "Pin" buttons, that is ONE workflow.
- Do NOT include the same element in two different workflows.
- SKIP workflows that are purely navigational duplicates of other workflows.
- SKIP any workflow whose name matches one already in the explored memory.
- CRITICAL — do NOT enumerate repetitive instances of the same control type. \
  Calendar dates, list items, table rows, pagination numbers, and any set of \
  controls that differ only by a date/number/ordinal are ONE workflow — pick a \
  single representative (e.g. one date, one item, one page number). \
  Exploring every June date adds no new coverage.

For each input field, reason about an appropriate test value from its label,
placeholder, type attribute, and the surrounding UI context. Do NOT use generic
placeholder text — think about what a real user would actually type here:
- A password field in a sign-up form → a realistic strong password (mix of letters, digits, symbol)
- A username or display-name field → a plausible short username
- An email field → a plausible email address
- A search or filter field → a realistic search term relevant to the app's content
- A title field for a note/memo → a short descriptive title that makes sense for the app
- A content or body field → a short realistic paragraph relevant to the app's purpose
- A URL or link field → a plausible URL
- Any numeric field → a plausible number for the context (e.g. quantity, age, price)
- Unknown fields → infer the most realistic value from the field's label and context

CRITICAL — selectors:
- You will be given a list of interactive elements with their REAL selectors.
- Every step's "selector" field MUST be copied verbatim from that list.
- Do NOT invent, guess, or modify selectors. A wrong selector causes a timeout.

Step types: "fill" | "click" | "select" | "press"
For "press" the value is the key name (e.g. "Enter", "Escape").
For "select" the value is the option text to choose.

Return ONLY a JSON array — no markdown, no text outside the JSON:
[
  {
    "name": "snake_case_workflow_name",
    "description": "one sentence: what does this workflow accomplish?",
    "steps": [
      {"type": "fill",  "selector": "<selector from element list>", "value": "<inferred value>"},
      {"type": "fill",  "selector": "<selector from element list>", "value": "<inferred value>"},
      {"type": "click", "selector": "<selector from element list>"}
    ],
    "expected_outcome": "brief description of what should change on the page"
  }
]\
"""

GOAL_WRITER_SYSTEM = """\
You are a test documentation expert. Given a raw browser-automation trajectory \
(a sequence of UI interactions), produce a structured test goal document.

Return ONLY a JSON object — no markdown, no text outside the JSON:
{
  "goal": "<one sentence: what user scenario this test verifies>",
  "instructions": [
    "<step 1 in plain English>",
    "<step 2 in plain English>"
  ],
  "success_criteria": [
    "<verifiable condition 1>",
    "<verifiable condition 2>"
  ]
}

Guidelines:
- goal: read like a user story acceptance criterion ("Verify that a user can…")
- instructions: translate each automation step into clear plain English — mention \
  field names, button labels, and realistic values the user would enter
- success_criteria: 3-5 concrete, observable checks the verifier confirms at the end \
  (data persisted, UI updated, no errors shown, correct navigation, etc.)\
"""

ALTERNATIVE_FILLS_SYSTEM = """\
You are a backend-aware UI automation assistant. \
You reason about server-side form validation rules to generate fill values \
that are genuinely different strategies — not just variations of the same pattern. \
Reply with ONLY a JSON object — no markdown, no explanation, no chain-of-thought text.\
"""

_ALTERNATIVE_FILL_STRATEGIES = [
    "Try VERY SHORT, simple values: username 4-6 lowercase letters only (e.g. 'jsmith'), password exactly 8 chars with 1 uppercase + 1 digit + 1 special (e.g. 'Pass1!ab').",
    "Try values with NO special characters or underscores anywhere — some servers ban them: plain lowercase username (e.g. 'testuser99'), password using only letters+digits (e.g. 'TestPass99').",
    "Try a LONGER username (10-15 chars) and a longer password (16+ chars) with multiple special chars — maybe the server requires a minimum length you haven't met.",
    "Try a completely different NAME STYLE: firstname+lastname format for username (e.g. 'johndoe2025'), and a passphrase-style password with spaces if the field allows (e.g. 'Correct!Horse9Battery').",
]


def build_page_actions_messages(
    llm: "LLMOracle",
    a11y_tree: dict,
    screenshot: bytes,
    elements: list[dict] | None = None,
    explored_context: str = "",
) -> list[dict]:
    tree_json = format_a11y_tree(a11y_tree)
    element_list = format_element_list(elements)
    memory_section = f"\n\n{explored_context}" if explored_context else ""
    user_text = (
        f"Accessibility tree:\n{tree_json}\n\n"
        f"Interactive elements with REAL selectors (use ONLY these):\n{element_list}"
        f"{memory_section}"
    )
    return [
        {
            "role": "user",
            "content": [
                llm.image_block(screenshot),
                {"type": "text", "text": user_text},
            ],
        }
    ]


def build_goal_writer_messages(
    llm: "LLMOracle",
    trajectory: dict,
    screenshots: list[tuple[str, bytes]] | None = None,
) -> list[dict]:
    steps_text = ""
    for i, step in enumerate(trajectory.get("steps", []), start=1):
        action = step.get("action", "?")
        desc = step.get("description", "")
        sub_steps = step.get("steps", [])
        sub_line = "  ".join(
            f'{s.get("type")}({s.get("selector", "")!r}, {s.get("value", "")!r})'
            for s in sub_steps
        )
        steps_text += f"\nStep {i}: {action}"
        if desc:
            steps_text += f"\n  Description: {desc}"
        if sub_line:
            steps_text += f"\n  Interactions: {sub_line}"

    intro = (
        f"Trajectory ID: {trajectory.get('id', '')}\n"
        f"Description: {trajectory.get('description', '')}\n"
        f"Steps:{steps_text}\n\n"
    )

    content: list[dict] = [{"type": "text", "text": intro}]
    if screenshots:
        content.append({
            "type": "text",
            "text": (
                f"Below are {len(screenshots)} screenshot(s) showing the UI state "
                "at each point in the trajectory, in order. Use them to write "
                "specific, grounded instructions that reference visible UI elements "
                "(button labels, field placeholders, page headings, etc.).\n"
            ),
        })
        for label, png_bytes in screenshots:
            content.append({"type": "text", "text": f"— {label} —"})
            content.append(llm.image_block(png_bytes))

    content.append({"type": "text", "text": "Produce the test goal document as described."})
    return [{"role": "user", "content": content}]


def build_alternative_fills_messages(
    llm: "LLMOracle",
    action_name: str,
    steps: list[dict],
    screenshot: bytes,
    previous_attempts: list[dict[str, str]] | None = None,
) -> list[dict] | None:
    fills = [s for s in steps if s.get("type") == "fill"]
    if not fills:
        return None

    original_desc = "\n".join(
        f'  selector={s["selector"]!r}  original_value={s["value"]!r}'
        for s in fills
    )

    prior_section = ""
    if previous_attempts:
        lines = []
        for i, attempt in enumerate(previous_attempts, 1):
            pairs = "  ".join(f'{sel!r}: {val!r}' for sel, val in attempt.items())
            lines.append(f"  Attempt {i}: {pairs}")
        prior_section = (
            "\n\nPrevious retry attempts that ALSO FAILED — do not repeat these patterns:\n"
            + "\n".join(lines)
        )

    attempt_num = len(previous_attempts or []) + 1
    strategy_hint = _ALTERNATIVE_FILL_STRATEGIES[(attempt_num - 1) % len(_ALTERNATIVE_FILL_STRATEGIES)]

    prompt = (
        f"Action '{action_name}' ran but the page did not navigate away — "
        "the form rejected the input.\n\n"
        f"Original fill values:\n{original_desc}"
        f"{prior_section}\n\n"
        "STEP 1 — Read the screenshot carefully. What error message does the UI show? "
        "(e.g. 'Username already taken', 'Password too short', 'Invalid characters', etc.)\n\n"
        "STEP 2 — Based on the error, reason about the specific server-side validation rule "
        "that is failing. Common rules:\n"
        "  • Username: min/max length, alphanumeric+underscore only, must start with letter, "
        "no spaces, case-insensitive uniqueness check\n"
        "  • Password: min 8 chars, must have uppercase+lowercase+digit+special char, "
        "no spaces allowed, max length cap, common-password blacklist\n"
        "  • 'Username taken' → pick a completely unrelated name, not a variation\n\n"
        f"STEP 3 — For THIS attempt, use this specific strategy: {strategy_hint}\n\n"
        "Generate values that directly target the hypothesised rule. "
        "Do NOT just increment a number or add a suffix — that is not a different strategy.\n\n"
        "Reply with ONLY a JSON object mapping each selector to its new value:\n"
        '{"<selector>": "<new value>", ...}'
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


async def analyze_page_actions(
    llm: "LLMOracle",
    a11y_tree: dict,
    screenshot: bytes,
    elements: list[dict] | None = None,
    explored_context: str = "",
) -> list[dict] | None:
    messages = build_page_actions_messages(
        llm, a11y_tree, screenshot, elements=elements, explored_context=explored_context
    )
    try:
        raw = await llm.complete(PAGE_ACTIONS_SYSTEM, messages, max_tokens=1500)
        raw = strip_code_fence(raw)
        workflows = json.loads(raw)
        return [w for w in workflows if "name" in w and "steps" in w]
    except Exception:
        return None


async def write_trajectory_goal(
    llm: "LLMOracle",
    trajectory: dict,
    screenshots: list[tuple[str, bytes]] | None = None,
) -> dict | None:
    messages = build_goal_writer_messages(llm, trajectory, screenshots)
    try:
        raw = await llm.complete(GOAL_WRITER_SYSTEM, messages, max_tokens=1000)
        raw = strip_code_fence(raw)
        data = json.loads(raw)
        if "goal" in data and "instructions" in data and "success_criteria" in data:
            return data
        return None
    except Exception:
        return None


async def suggest_alternative_fills(
    llm: "LLMOracle",
    action_name: str,
    steps: list[dict],
    screenshot: bytes,
    previous_attempts: list[dict[str, str]] | None = None,
) -> dict[str, str] | None:
    messages = build_alternative_fills_messages(
        llm, action_name, steps, screenshot, previous_attempts=previous_attempts
    )
    if messages is None:
        return None
    try:
        raw = await llm.complete(ALTERNATIVE_FILLS_SYSTEM, messages, max_tokens=300)
        raw = strip_code_fence(raw)
        data = json.loads(raw)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else None
    except Exception:
        return None
