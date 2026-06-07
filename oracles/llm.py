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

_GROUPING_SYSTEM = """\
You are analyzing interactive elements on a web page to identify distinct user actions.

Your job:
1. Group semantically IDENTICAL actions — e.g. if there are 10 "Pin" buttons for
   different memos, that is ONE action: "pin_a_memo".
2. Give each group a snake_case name (e.g. "create_memo", "pin_a_memo", "search_memos").
3. Pick the best single selector from the group as the representative.

Return ONLY a JSON array — no markdown, no explanation outside the JSON:
[
  {
    "name": "snake_case_action_name",
    "description": "one sentence: what does this action do?",
    "representative_selector": "CSS or aria-label selector"
  }
]\
"""

_RESOLVE_SYSTEM = "You are a UI automation assistant. Reply with ONLY a CSS selector string or the word NONE."

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

    async def complete(self, system: str, messages: list[dict]) -> str:
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=512,
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

    async def complete(self, system: str, messages: list[dict]) -> str:
        # Prepend system as a system message.
        full = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=512,
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

    async def complete(self, system: str, messages: list[dict]) -> str:
        full = [{"role": "system", "content": system}] + messages
        payload = {
            "model": self._model,
            "messages": full,
            "max_tokens": 512,
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

    async def group_actions(
        self,
        elements: list[dict],
        screenshot: bytes,
    ) -> list[dict] | None:
        """
        Ask the LLM to semantically group raw DOM elements into distinct actions.

        Returns a list of dicts with keys: name, description, representative_selector.
        Returns None on any failure so the caller can fall back to DOM grouping.
        """
        elements_json = json.dumps(
            [{"tag": e.get("tag"), "role": e.get("role"),
              "label": e.get("label"), "selector": e.get("selector")}
             for e in elements],
            indent=2,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    self._provider.image_block(screenshot),
                    {"type": "text", "text": f"Elements:\n{elements_json}"},
                ],
            }
        ]
        try:
            raw = await self._provider.complete(_GROUPING_SYSTEM, messages)
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            groups = json.loads(raw)
            return [g for g in groups if "name" in g and "representative_selector" in g]
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
