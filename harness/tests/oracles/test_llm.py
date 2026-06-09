"""
Tests for oracles/llm.py

All tests use mock providers — no real API calls made.
"""

import base64

import httpx
import pytest
import respx

from harness.oracles.llm import (
    LLMOracle,
    _AnthropicProvider,
    _LocalProvider,
    _OpenAIProvider,
    _build_provider,
)

_TEST_LOCAL_URL = "http://127.0.0.1:8000"


class _MockProvider:
    def __init__(self, response: str):
        self._response = response
        self.calls: list[dict] = []

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 512) -> str:
        self.calls.append({"system": system, "messages": messages, "max_tokens": max_tokens})
        return self._response

    @staticmethod
    def image_block(png_bytes: bytes) -> dict:
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}


# ---------------------------------------------------------------------------
# _build_provider
# ---------------------------------------------------------------------------

def test_build_provider_anthropic_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    assert isinstance(_build_provider(), _AnthropicProvider)


def test_build_provider_openai_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(_build_provider(), _OpenAIProvider)


def test_build_provider_auto_prefers_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert isinstance(_build_provider(), _AnthropicProvider)


def test_build_provider_auto_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    assert isinstance(_build_provider(), _OpenAIProvider)


def test_build_provider_no_keys_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    assert isinstance(_build_provider(), _LocalProvider)


def test_build_provider_empty_keys_fall_back_to_local(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    assert isinstance(_build_provider(), _LocalProvider)


def test_build_provider_local_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(_build_provider(), _LocalProvider)


def test_build_provider_local_lower_priority_than_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    assert isinstance(_build_provider(), _LocalProvider)


def test_build_provider_anthropic_beats_local_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert isinstance(_build_provider(), _AnthropicProvider)


# ---------------------------------------------------------------------------
# LLMOracle.complete / image_block
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_delegates_to_provider():
    provider = _MockProvider("hello")
    oracle = LLMOracle(provider=provider)
    result = await oracle.complete("sys", [{"role": "user", "content": "hi"}])
    assert result == "hello"
    assert provider.calls[0]["system"] == "sys"


@pytest.mark.asyncio
async def test_image_block_delegates_to_provider():
    provider = _MockProvider("ok")
    oracle = LLMOracle(provider=provider)
    block = oracle.image_block(b"\x89PNG")
    assert block["type"] == "image_url"


# ---------------------------------------------------------------------------
# _LocalProvider
# ---------------------------------------------------------------------------

def test_local_provider_image_block_is_image_url():
    block = _LocalProvider.image_block(b"\x89PNG fake")
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_local_provider_image_block_base64_encodes_bytes():
    raw = b"hello-image"
    block = _LocalProvider.image_block(raw)
    encoded = block["image_url"]["url"].split(",", 1)[1]
    assert base64.standard_b64decode(encoded) == raw


def test_local_provider_requires_url(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    with pytest.raises(EnvironmentError, match="LOCAL_LLM_URL"):
        _LocalProvider()


def test_local_provider_default_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    p = _LocalProvider()
    assert p._endpoint == f"{_TEST_LOCAL_URL}/v1/chat/completions"
    assert p._model == "Qwen/Qwen3.5-9B"


def test_local_provider_env_var_overrides(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", "http://192.168.1.100:8080")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "my-custom-model")
    p = _LocalProvider()
    assert "192.168.1.100" in p._endpoint
    assert p._model == "my-custom-model"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_complete_text_only(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    respx.post(f"{_TEST_LOCAL_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })
    )

    p = _LocalProvider()
    result = await p.complete("You are a judge.", [{"role": "user", "content": "Hello"}])
    assert result == "ok"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_complete_sends_system_message(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    captured = {}

    def capture(request):
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    respx.post(f"{_TEST_LOCAL_URL}/v1/chat/completions").mock(side_effect=capture)

    p = _LocalProvider()
    await p.complete("SYSTEM PROMPT", [{"role": "user", "content": "Hello"}])

    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYSTEM PROMPT"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_complete_sends_correct_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "test-model-v1")
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)

    captured = {}

    def capture(request):
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    respx.post(f"{_TEST_LOCAL_URL}/v1/chat/completions").mock(side_effect=capture)

    p = _LocalProvider()
    await p.complete("sys", [{"role": "user", "content": "hi"}])
    assert captured["body"]["model"] == "test-model-v1"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_raises_on_http_error(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_URL", _TEST_LOCAL_URL)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    respx.post(f"{_TEST_LOCAL_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    p = _LocalProvider()
    with pytest.raises(httpx.HTTPStatusError):
        await p.complete("sys", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_llm_oracle_from_env_with_injected_provider(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _FakeLocal:
        async def complete(self, system, messages, max_tokens=512):
            return "response"
        @staticmethod
        def image_block(png_bytes):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}

    oracle = LLMOracle(provider=_FakeLocal())
    assert await oracle.complete("sys", [{"role": "user", "content": "test"}]) == "response"
