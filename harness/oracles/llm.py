"""
LLM client — vision-capable provider transport for the harness.

Supports Anthropic (claude-sonnet-4-6), OpenAI (gpt-4o), and a local
OpenAI-compatible endpoint (Qwen/Qwen3.5-9B).  Provider priority:
  1. Anthropic  — explicit LLM_PROVIDER=anthropic, or non-empty ANTHROPIC_API_KEY
  2. OpenAI     — explicit LLM_PROVIDER=openai,     or non-empty OPENAI_API_KEY
  3. Local      — explicit LLM_PROVIDER=local,       or last-resort fallback
                  when neither cloud key is present

Domain-specific prompts live alongside their consumers:
  planner/prompts.py   — workflow discovery, goal writing, form retry
  executor/prompts.py  — action resolution and fill retry
  verifier/prompts.py  — per-step verification

Configuration
-------------
LLM settings are loaded from .env via harness.config — see .env.example.
"""

import base64
import os

import httpx

from harness.config import env_str, load_env

load_env()


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

class _AnthropicProvider:
    def __init__(self):
        import anthropic
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )
        self._model = env_str("ANTHROPIC_MODEL")

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
    def __init__(self):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = env_str("OPENAI_MODEL")

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
    """

    def __init__(self):
        base = env_str("LOCAL_LLM_URL").rstrip("/")
        if not base:
            raise EnvironmentError(
                "LOCAL_LLM_URL is not set — add it to .env (see .env.example)"
            )
        self._endpoint = f"{base}/v1/chat/completions"
        self._model    = env_str("LOCAL_LLM_MODEL")
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


def _has_api_key(name: str) -> bool:
    """True when the env var is set to a non-empty value (not just present in .env)."""
    return bool(env_str(name).strip())


def _build_provider():
    """
    Select provider based on env vars.

    Priority:
      1. anthropic  (explicit or non-empty ANTHROPIC_API_KEY)
      2. openai     (explicit or non-empty OPENAI_API_KEY)
      3. local      (explicit LLM_PROVIDER=local, or last-resort fallback)
    """
    explicit = env_str("LLM_PROVIDER").lower()

    if explicit == "local":
        return _LocalProvider()

    if explicit == "anthropic" or (not explicit and _has_api_key("ANTHROPIC_API_KEY")):
        return _AnthropicProvider()

    if explicit == "openai" or (not explicit and _has_api_key("OPENAI_API_KEY")):
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
