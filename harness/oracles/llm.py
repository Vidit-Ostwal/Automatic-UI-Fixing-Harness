"""
LLM client — vision-capable provider transport for the harness.

Supports Anthropic (claude-sonnet-4-6), OpenAI (gpt-4o), and a local
OpenAI-compatible endpoint (Qwen/Qwen3.5-9B).  Provider priority:
  1. Anthropic  — explicit LLM_PROVIDER=anthropic, or ANTHROPIC_API_KEY set
  2. OpenAI     — explicit LLM_PROVIDER=openai,     or OPENAI_API_KEY set
  3. Local      — explicit LLM_PROVIDER=local,       or last-resort fallback
                  when neither cloud key is present

Domain-specific prompts live alongside their consumers:
  oracles/prompts.py   — visual and diff judgment
  planner/prompts.py   — workflow discovery, goal writing, form retry
  executor/prompts.py  — action resolution and fill retry
  verifier/prompts.py  — per-step verification

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

from harness.utils.llm import strip_code_fence


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
# Shared parsing helpers
# ---------------------------------------------------------------------------

_VALID_VERDICTS = {"bug", "ok", "noise"}
_VALID_SEVERITIES = {"critical", "high", "medium", "low", None}


def parse_verdict(raw: str) -> OracleVerdict:
    """Parse LLM JSON response into OracleVerdict, with safe fallbacks."""
    try:
        raw = strip_code_fence(raw)
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


# Backward-compatible alias used by tests.
_parse_verdict = parse_verdict


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

    return _LocalProvider()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LLMOracle:
    """
    Vision-capable LLM transport layer.

    Domain tasks are implemented in each module's prompts.py and call
    `complete()` / `image_block()` on this client.
    """

    def __init__(self, provider=None):
        self._provider = provider or _build_provider()

    @classmethod
    def from_env(cls) -> "LLMOracle":
        return cls(provider=_build_provider())

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        return await self._provider.complete(system, messages, max_tokens=max_tokens)

    def image_block(self, png_bytes: bytes) -> dict:
        return self._provider.image_block(png_bytes)

    # Convenience wrappers for oracle checks (prompts live in oracles/prompts.py).
    async def judge_screenshot(self, screenshot: bytes, context: str = "") -> OracleVerdict:
        from harness.oracles.prompts import judge_screenshot
        return await judge_screenshot(self, screenshot, context)

    async def judge_diff(self, before, after, action: str) -> OracleVerdict:
        from harness.oracles.prompts import judge_diff
        return await judge_diff(self, before, after, action)
