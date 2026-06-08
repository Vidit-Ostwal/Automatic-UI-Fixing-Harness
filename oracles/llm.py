"""
LLM oracle — asks a vision-capable model to judge screenshots and diffs.

Supports Anthropic (claude-sonnet-4-6), OpenAI (gpt-4o), and a local
OpenAI-compatible endpoint (Qwen/Qwen3.5-9B).  Provider priority:
  1. Anthropic  — explicit LLM_PROVIDER=anthropic, or ANTHROPIC_API_KEY set
  2. OpenAI     — explicit LLM_PROVIDER=openai,     or OPENAI_API_KEY set
  3. Local      — explicit LLM_PROVIDER=local,       or last-resort fallback
                  when neither cloud key is present

Environment variables
---------------------
LLM_PROVIDER        "anthropic" | "openai" | "local"  (optional, auto-detected)
ANTHROPIC_API_KEY   required when provider = anthropic
OPENAI_API_KEY      required when provider = openai
ANTHROPIC_MODEL     override default anthropic model  (optional)
OPENAI_MODEL        override default openai model      (optional)
LOCAL_LLM_URL       override local endpoint base URL   (default: http://20.150.215.227)
LOCAL_LLM_MODEL     override local model name          (default: Qwen/Qwen3.5-9B)
"""

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

from models import PageState


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OracleVerdict:
    verdict: str               # "bug" | "ok" | "noise"
    description: str
    severity: Optional[str]    # "critical" | "high" | "medium" | "low" | None
    reasoning: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_VISUAL_SYSTEM = """\
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

_PAGE_ACTIONS_SYSTEM = """\
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

_RESOLVE_SYSTEM = "You are a UI automation assistant. Reply with ONLY a CSS selector string or the word NONE."

_FILL_RETRY_SYSTEM = """\
You are a UI automation assistant. A fill step failed because the value you \
provided was not accepted by the input field — the field ended up with a \
different value than what was written.

Reason about why the value was rejected (wrong format, too long, contains \
forbidden characters, wrong type, etc.) and suggest a corrected value that \
the field is likely to accept.

Reply with ONLY the raw replacement value — no quotes, no explanation, \
no markdown. Just the string to type into the field.\
"""

_VERIFIER_SYSTEM = """\
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

_GOAL_WRITER_SYSTEM = """\
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

_DIFF_SYSTEM = """\
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


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class _AnthropicProvider:
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self):
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self._model = os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()

    @staticmethod
    def image_block(png_bytes: bytes) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(png_bytes).decode(),
            },
        }


class _OpenAIProvider:
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.environ.get("OPENAI_MODEL", self.DEFAULT_MODEL)

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        # Prepend system as a system message.
        full = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=full,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def image_block(png_bytes: bytes) -> dict:
        data_url = (
            "data:image/png;base64,"
            + base64.standard_b64encode(png_bytes).decode()
        )
        return {"type": "image_url", "image_url": {"url": data_url}}


class _LocalProvider:
    """
    OpenAI-compatible local inference endpoint (Qwen/Qwen3.5-9B by default).

    Requires no API key.  Supports both vision (image_url) and text-only calls.
    Uses httpx for async HTTP so there is no extra runtime dependency.
    """

    DEFAULT_URL   = "http://20.150.215.227"
    DEFAULT_MODEL = "Qwen/Qwen3.5-9B"

    def __init__(self):
        import httpx
        base = os.environ.get("LOCAL_LLM_URL", self.DEFAULT_URL).rstrip("/")
        self._endpoint = f"{base}/v1/chat/completions"
        self._model    = os.environ.get("LOCAL_LLM_MODEL", self.DEFAULT_MODEL)
        self._client   = httpx.AsyncClient(timeout=120.0)

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        full = [{"role": "system", "content": system}] + messages
        payload = {
            "model": self._model,
            "messages": full,
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        response = await self._client.post(
            self._endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def image_block(png_bytes: bytes) -> dict:
        data_url = (
            "data:image/png;base64,"
            + base64.standard_b64encode(png_bytes).decode()
        )
        return {"type": "image_url", "image_url": {"url": data_url}}


def _build_provider():
    """
    Select provider based on env vars.

    Priority:
      1. anthropic  (explicit or ANTHROPIC_API_KEY present)
      2. openai     (explicit or OPENAI_API_KEY present)
      3. local      (explicit LLM_PROVIDER=local, or last-resort fallback)
    """
    explicit = os.environ.get("LLM_PROVIDER", "").lower()

    if explicit == "anthropic" or (not explicit and "ANTHROPIC_API_KEY" in os.environ):
        return _AnthropicProvider()

    if explicit == "openai" or (not explicit and "OPENAI_API_KEY" in os.environ):
        return _OpenAIProvider()

    # Last resort: local endpoint — works with no API keys.
    return _LocalProvider()


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

_VALID_VERDICTS = {"bug", "ok", "noise"}
_VALID_SEVERITIES = {"critical", "high", "medium", "low", None}


def _parse_verdict(raw: str) -> OracleVerdict:
    """Parse LLM JSON response into OracleVerdict, with safe fallbacks."""
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        return OracleVerdict(
            verdict="noise",
            description="LLM returned unparseable response.",
            severity=None,
            reasoning=raw[:200],
        )

    verdict = data.get("verdict", "noise")
    if verdict not in _VALID_VERDICTS:
        verdict = "noise"

    severity = data.get("severity")
    if severity not in _VALID_SEVERITIES:
        severity = None
    if verdict != "bug":
        severity = None

    return OracleVerdict(
        verdict=verdict,
        description=data.get("description", ""),
        severity=severity,
        reasoning=data.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LLMOracle:
    """
    Vision-capable LLM oracle.

    Supports Anthropic, OpenAI, and a local Qwen endpoint. Provider is
    auto-detected from env vars or overridden via LLM_PROVIDER.

    Methods
    -------
    judge_screenshot(screenshot, context)
        Send a single screenshot for visual defect analysis.

    judge_diff(before, after, action)
        Send before/after screenshots to judge whether a state change is
        expected, buggy, or noise.
    """

    def __init__(self, provider=None):
        self._provider = provider or _build_provider()

    @classmethod
    def from_env(cls) -> "LLMOracle":
        return cls(provider=_build_provider())

    async def judge_screenshot(
        self,
        screenshot: bytes,
        context: str = "",
    ) -> OracleVerdict:
        """
        Analyse a single screenshot for visual defects.
        context is a short description of what the page is showing.
        """
        text = "Analyse this screenshot for UI defects."
        if context:
            text += f" Context: {context}"

        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": text},
                ],
            }
        ]
        raw = await self._provider.complete(_VISUAL_SYSTEM, messages)
        return _parse_verdict(raw)

    async def analyze_page_actions(
        self,
        a11y_tree: dict,
        screenshot: bytes,
        elements: list[dict] | None = None,
        explored_context: str = "",
    ) -> list[dict] | None:
        """
        Reason about what complete user workflows are possible on this page.

        Sends the screenshot, accessibility tree, and (crucially) the list of
        real DOM elements with their actual selectors. The LLM must use only
        selectors from that list — this prevents hallucinated selectors that
        time out at execution time.

        explored_context is the ExplorationMemory summary — a compact list of
        already-tried workflows per URL. The LLM uses it to focus on what is
        still unexplored rather than re-suggesting the same actions.

        Returns a list of workflow dicts, or None on any failure so the caller
        can fall back to DOM-only grouping.
        """
        tree_json = json.dumps(a11y_tree, indent=2)[:3000]  # guard token budget

        # Serialize only the fields the LLM needs: selector, role, label, type.
        element_list = json.dumps(
            [
                {
                    "selector": e.get("selector", ""),
                    "role":     e.get("role", ""),
                    "label":    e.get("label", ""),
                    "type":     e.get("type", ""),
                }
                for e in (elements or [])
                if e.get("selector")
            ],
            indent=2,
        )[:2000]

        memory_section = (
            f"\n\n{explored_context}" if explored_context else ""
        )
        user_text = (
            f"Accessibility tree:\n{tree_json}\n\n"
            f"Interactive elements with REAL selectors (use ONLY these):\n{element_list}"
            f"{memory_section}"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(
                _PAGE_ACTIONS_SYSTEM, messages, max_tokens=1500
            )
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            workflows = json.loads(raw)
            return [w for w in workflows if "name" in w and "steps" in w]
        except Exception:
            return None

    async def resolve_instruction(
        self,
        screenshot: bytes,
        elements: list[dict],
        instruction: str,
    ) -> list[dict] | None:
        """
        Map one plain-English instruction to a sequence of UI interaction steps.

        Returns a list like:
          [{"type": "fill",  "selector": "...", "value": "..."},
           {"type": "click", "selector": "..."}]
        or None if the instruction cannot be resolved against the current page.

        The returned selectors are always copied verbatim from `elements`.
        """
        element_list = json.dumps(
            [
                {
                    "selector": e.get("selector", ""),
                    "role":     e.get("role", ""),
                    "label":    e.get("label", ""),
                    "type":     e.get("type", ""),
                }
                for e in elements[:40]
                if e.get("selector")
            ],
            indent=2,
        )
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
        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(
                (
                    "You are a precise UI automation assistant. "
                    "You translate one plain-English instruction into the minimal "
                    "sequence of browser interactions needed to execute it. "
                    "Use ONLY selectors from the provided element list. "
                    "Reply with ONLY a JSON array."
                ),
                messages,
                max_tokens=400,
            )
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            steps = json.loads(raw)
            if isinstance(steps, list):
                return steps or None
            return None
        except Exception:
            return None

    async def resolve_action(
        self,
        screenshot: bytes,
        elements: list[dict],
        action_name: str,
    ) -> str | None:
        """
        Ask the LLM which selector to click to perform action_name.

        Returns a selector string, or None if no match is found.
        """
        element_list = json.dumps(
            [{"label": e.get("label"), "selector": e.get("selector"), "role": e.get("role")}
             for e in elements[:20]],
            indent=2,
        )
        prompt = (
            f"I need to perform the action: '{action_name}'.\n"
            f"Available interactive elements:\n{element_list}\n\n"
            f"Which selector should I click? Reply with ONLY the selector string. "
            f"If no element matches, reply with 'NONE'."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(_RESOLVE_SYSTEM, messages)
            selector = raw.strip().strip("\"'")
            return selector if selector and selector != "NONE" else None
        except Exception:
            return None

    async def suggest_alternative_fills(
        self,
        action_name: str,
        steps: list[dict],
        screenshot: bytes,
        previous_attempts: list[dict[str, str]] | None = None,
    ) -> dict[str, str] | None:
        """
        The action ran but the URL didn't change — likely a form error.
        Suggest a new set of fill values that are meaningfully different from
        all previous attempts and more likely to succeed.

        previous_attempts is a list of {selector: value} dicts, one per prior
        retry, so the LLM knows exactly what has already been tried.

        Returns {selector: new_value} or None on failure.
        """
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

        attempt_num = len(previous_attempts) + 1
        strategies = [
            "Try VERY SHORT, simple values: username 4-6 lowercase letters only (e.g. 'jsmith'), password exactly 8 chars with 1 uppercase + 1 digit + 1 special (e.g. 'Pass1!ab').",
            "Try values with NO special characters or underscores anywhere — some servers ban them: plain lowercase username (e.g. 'testuser99'), password using only letters+digits (e.g. 'TestPass99').",
            "Try a LONGER username (10-15 chars) and a longer password (16+ chars) with multiple special chars — maybe the server requires a minimum length you haven't met.",
            "Try a completely different NAME STYLE: firstname+lastname format for username (e.g. 'johndoe2025'), and a passphrase-style password with spaces if the field allows (e.g. 'Correct!Horse9Battery').",
        ]
        strategy_hint = strategies[(attempt_num - 1) % len(strategies)]

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
        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(
                (
                    "You are a backend-aware UI automation assistant. "
                    "You reason about server-side form validation rules to generate fill values "
                    "that are genuinely different strategies — not just variations of the same pattern. "
                    "Reply with ONLY a JSON object — no markdown, no explanation, no chain-of-thought text."
                ),
                messages,
                max_tokens=300,
            )
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else None
        except Exception:
            return None

    async def suggest_fill_value(
        self,
        selector: str,
        tried_value: str,
        actual_value: str,
        screenshot: bytes,
    ) -> str | None:
        """
        Called when a fill step's value was not accepted by the input field.
        Returns a replacement string to try, or None on failure.
        """
        prompt = (
            f"Input field selector: {selector}\n"
            f"Value we tried to fill: {tried_value!r}\n"
            f"Value the field actually contains after fill: {actual_value!r}\n\n"
            "The field rejected or modified our value. "
            "Suggest a corrected value that the field will accept."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(_FILL_RETRY_SYSTEM, messages, max_tokens=100)
            value = raw.strip().strip("\"'")
            return value if value else None
        except Exception:
            return None

    async def verify_step(
        self,
        goal: dict,
        history: list[dict],
        current_step: dict,
    ) -> list[dict]:
        """
        Verify one executor step for visual and logic bugs.

        Parameters
        ----------
        goal          Full goal dict (goal, instructions, success_criteria).
        history       Previous steps, each:
                        {step_index, instruction, screenshot_after: bytes,
                         url_after, success}
        current_step  Step being verified:
                        {step_index, instruction,
                         screenshot_before: bytes, screenshot_after: bytes,
                         url_before, url_after, success, error}

        Returns a (possibly empty) list of finding dicts.
        """
        content: list[dict] = []

        # Goal context
        criteria = "\n".join(f"  - {c}" for c in goal.get("success_criteria", []))
        content.append({"type": "text", "text": (
            f"TEST GOAL: {goal.get('goal', '')}\n\n"
            f"SUCCESS CRITERIA:\n{criteria}\n"
        )})

        # History — after-screenshot per previous step to show trajectory
        if history:
            content.append({"type": "text", "text": "── EXECUTION HISTORY (steps already done) ──"})
            for h in history:
                status = "✓" if h.get("success") else "✗"
                content.append({"type": "text", "text": (
                    f"Step {h['step_index']} [{status}]: {h['instruction']}\n"
                    f"  URL after: {h.get('url_after', '')}"
                )})
                if h.get("screenshot_after"):
                    content.append(self._provider.image_block(h["screenshot_after"]))

        # Current step
        idx    = current_step["step_index"]
        status = "✓ succeeded" if current_step.get("success") else f"✗ failed — {current_step.get('error', '')}"
        content.append({"type": "text", "text": (
            f"\n── CURRENT STEP {idx} [{status}] ──\n"
            f"Instruction: {current_step['instruction']}\n"
            f"URL before : {current_step.get('url_before', '')}\n"
            f"URL after  : {current_step.get('url_after', '')}\n\n"
            "BEFORE screenshot:"
        )})
        content.append(self._provider.image_block(current_step["screenshot_before"]))
        content.append({"type": "text", "text": "AFTER screenshot:"})
        content.append(self._provider.image_block(current_step["screenshot_after"]))
        content.append({"type": "text", "text": (
            "Analyse the CURRENT step for visual and logic bugs. "
            "Report only real, observable defects."
        )})

        messages = [{"role": "user", "content": content}]
        try:
            raw = await self._provider.complete(_VERIFIER_SYSTEM, messages, max_tokens=1200)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            findings = data.get("findings", [])
            return [f for f in findings if isinstance(f, dict) and "bug_type" in f]
        except Exception:
            return []

    async def write_trajectory_goal(
        self,
        trajectory: dict,
        screenshots: list[tuple[str, bytes]] | None = None,
    ) -> dict | None:
        """
        Convert a trajectory dict into a structured goal document with
        plain-English instructions and verifiable success criteria.

        screenshots — list of (label, png_bytes) pairs in state order, as
                      produced by goal_writer._load_screenshots().  When
                      provided the images are interleaved into the message so
                      the LLM can ground instructions in the actual UI.

        Returns {goal, instructions, success_criteria} or None on failure.
        """
        steps_text = ""
        for i, step in enumerate(trajectory.get("steps", []), start=1):
            action    = step.get("action", "?")
            desc      = step.get("description", "")
            sub_steps = step.get("steps", [])
            sub_line  = "  ".join(
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

        # Build the content list — text intro, then interleaved screenshot images.
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
                content.append(self._provider.image_block(png_bytes))

        content.append({"type": "text", "text": "Produce the test goal document as described."})

        messages = [{"role": "user", "content": content}]
        try:
            raw = await self._provider.complete(_GOAL_WRITER_SYSTEM, messages, max_tokens=1000)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            if "goal" in data and "instructions" in data and "success_criteria" in data:
                return data
            return None
        except Exception:
            return None

    async def judge_diff(
        self,
        before: PageState,
        after: PageState,
        action: str,
    ) -> OracleVerdict:
        """
        Compare before/after screenshots and judge whether the change is
        expected for the given action.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "BEFORE screenshot:"},
                    self._provider.image_block(before.screenshot),
                    {"type": "text", "text": "AFTER screenshot:"},
                    self._provider.image_block(after.screenshot),
                    {"type": "text", "text": f"Action performed: {action}"},
                ],
            }
        ]
        raw = await self._provider.complete(_DIFF_SYSTEM, messages)
        return _parse_verdict(raw)
