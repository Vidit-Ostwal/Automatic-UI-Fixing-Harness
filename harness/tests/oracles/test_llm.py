"""
Tests for oracles/llm.py

All tests use mock providers — no real API calls made.
Covers:
  - Provider auto-detection from env vars
  - Verdict parsing (valid JSON, malformed JSON, code-fenced JSON)
  - Severity clamped to None when verdict != "bug"
  - Unknown verdict falls back to "noise"
  - judge_screenshot / judge_diff call the right prompt paths
  - Missing API keys raise EnvironmentError
"""

import os
import pytest

from harness.oracles.llm import (
    LLMOracle,
    OracleVerdict,
    _parse_verdict,
    _build_provider,
    _AnthropicProvider,
    _OpenAIProvider,
    _LocalProvider,
)
from harness.models import PageState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(screenshot: bytes = b"\x89PNG fake") -> PageState:
    return PageState(
        url="http://localhost:5230/",
        screenshot=screenshot,
        a11y_tree={"role": "main", "tag": "main"},
        timestamp=1.0,
    )


class _MockProvider:
    """Injectable provider that returns a preset response string."""

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
# _parse_verdict
# ---------------------------------------------------------------------------

def test_parse_valid_bug_verdict():
    raw = '{"verdict": "bug", "description": "Overlapping buttons", "severity": "high", "reasoning": "Two buttons occupy same space."}'
    v = _parse_verdict(raw)
    assert v.verdict == "bug"
    assert v.severity == "high"
    assert "Overlapping" in v.description


def test_parse_ok_verdict_severity_is_none():
    raw = '{"verdict": "ok", "description": "Looks fine", "severity": "high", "reasoning": "All good."}'
    v = _parse_verdict(raw)
    assert v.verdict == "ok"
    assert v.severity is None  # severity clamped to None for non-bug


def test_parse_noise_verdict():
    raw = '{"verdict": "noise", "description": "Loading spinner", "severity": null, "reasoning": "Transient."}'
    v = _parse_verdict(raw)
    assert v.verdict == "noise"
    assert v.severity is None


def test_parse_unknown_verdict_falls_back_to_noise():
    raw = '{"verdict": "maybe", "description": "Unclear", "severity": null, "reasoning": ""}'
    v = _parse_verdict(raw)
    assert v.verdict == "noise"


def test_parse_invalid_json_returns_noise():
    v = _parse_verdict("this is not json at all")
    assert v.verdict == "noise"
    assert isinstance(v.reasoning, str)


def test_parse_code_fenced_json():
    raw = '```json\n{"verdict": "bug", "description": "Bad layout", "severity": "medium", "reasoning": "Clipped text."}\n```'
    v = _parse_verdict(raw)
    assert v.verdict == "bug"
    assert v.severity == "medium"


def test_parse_missing_verdict_field():
    raw = '{"description": "Something", "severity": null, "reasoning": ""}'
    v = _parse_verdict(raw)
    assert v.verdict == "noise"


def test_parse_invalid_severity_clamped():
    raw = '{"verdict": "bug", "description": "Bad", "severity": "catastrophic", "reasoning": ""}'
    v = _parse_verdict(raw)
    assert v.severity is None


# ---------------------------------------------------------------------------
# _build_provider
# ---------------------------------------------------------------------------

def test_build_provider_anthropic_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    provider = _build_provider()
    assert isinstance(provider, _AnthropicProvider)


def test_build_provider_openai_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    # Ensure no anthropic key interferes
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = _build_provider()
    assert isinstance(provider, _OpenAIProvider)


def test_build_provider_auto_prefers_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    provider = _build_provider()
    assert isinstance(provider, _AnthropicProvider)


def test_build_provider_auto_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    provider = _build_provider()
    assert isinstance(provider, _OpenAIProvider)


def test_build_provider_no_keys_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = _build_provider()
    assert isinstance(provider, _LocalProvider)


def test_build_provider_local_explicit(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = _build_provider()
    assert isinstance(provider, _LocalProvider)


def test_build_provider_local_lower_priority_than_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "local")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # LLM_PROVIDER=local takes precedence only when explicitly set
    provider = _build_provider()
    assert isinstance(provider, _LocalProvider)


def test_build_provider_anthropic_beats_local_fallback(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = _build_provider()
    assert isinstance(provider, _AnthropicProvider)


# ---------------------------------------------------------------------------
# LLMOracle.judge_screenshot
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_screenshot_returns_verdict():
    provider = _MockProvider('{"verdict": "ok", "description": "Fine", "severity": null, "reasoning": "All good."}')
    oracle = LLMOracle(provider=provider)
    state = _make_state()
    verdict = await oracle.judge_screenshot(state.screenshot, context="Home page")
    assert isinstance(verdict, OracleVerdict)
    assert verdict.verdict == "ok"


@pytest.mark.asyncio
async def test_judge_screenshot_sends_image_block():
    provider = _MockProvider('{"verdict": "noise", "description": "", "severity": null, "reasoning": ""}')
    oracle = LLMOracle(provider=provider)
    await oracle.judge_screenshot(b"\x89PNG fake", context="test")
    assert len(provider.calls) == 1
    content = provider.calls[0]["messages"][0]["content"]
    # First content block must be the image.
    assert content[0]["type"] in ("image", "image_url")


@pytest.mark.asyncio
async def test_judge_screenshot_context_in_text():
    provider = _MockProvider('{"verdict": "noise", "description": "", "severity": null, "reasoning": ""}')
    oracle = LLMOracle(provider=provider)
    await oracle.judge_screenshot(b"\x89PNG fake", context="After pinning a memo")
    content = provider.calls[0]["messages"][0]["content"]
    text_blocks = [c for c in content if c.get("type") == "text"]
    assert any("After pinning" in b["text"] for b in text_blocks)


@pytest.mark.asyncio
async def test_judge_screenshot_bug_verdict_has_severity():
    provider = _MockProvider('{"verdict": "bug", "description": "Overlap", "severity": "high", "reasoning": "Two elements."}')
    oracle = LLMOracle(provider=provider)
    verdict = await oracle.judge_screenshot(b"\x89PNG", context="")
    assert verdict.verdict == "bug"
    assert verdict.severity == "high"


# ---------------------------------------------------------------------------
# LLMOracle.judge_diff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_diff_sends_two_images():
    provider = _MockProvider('{"verdict": "ok", "description": "Expected change", "severity": null, "reasoning": "Pin worked."}')
    oracle = LLMOracle(provider=provider)
    before = _make_state(b"\x89PNG before")
    after  = _make_state(b"\x89PNG after")
    await oracle.judge_diff(before, after, action="pin_memo")
    content = provider.calls[0]["messages"][0]["content"]
    image_blocks = [c for c in content if c.get("type") in ("image", "image_url")]
    assert len(image_blocks) == 2


@pytest.mark.asyncio
async def test_judge_diff_action_in_message():
    provider = _MockProvider('{"verdict": "ok", "description": "", "severity": null, "reasoning": ""}')
    oracle = LLMOracle(provider=provider)
    before = _make_state()
    after  = _make_state()
    await oracle.judge_diff(before, after, action="archive_memo")
    content = provider.calls[0]["messages"][0]["content"]
    text_blocks = [c for c in content if c.get("type") == "text"]
    assert any("archive_memo" in b["text"] for b in text_blocks)


@pytest.mark.asyncio
async def test_judge_diff_returns_oracle_verdict():
    provider = _MockProvider('{"verdict": "bug", "description": "Nothing changed", "severity": "high", "reasoning": "Pin did not work."}')
    oracle = LLMOracle(provider=provider)
    verdict = await oracle.judge_diff(_make_state(), _make_state(), "pin_memo")
    assert isinstance(verdict, OracleVerdict)
    assert verdict.verdict == "bug"


# ---------------------------------------------------------------------------
# _LocalProvider
# ---------------------------------------------------------------------------

import httpx
import respx


def test_local_provider_image_block_is_image_url():
    block = _LocalProvider.image_block(b"\x89PNG fake")
    assert block["type"] == "image_url"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


def test_local_provider_image_block_base64_encodes_bytes():
    import base64
    raw = b"hello-image"
    block = _LocalProvider.image_block(raw)
    encoded = block["image_url"]["url"].split(",", 1)[1]
    assert base64.standard_b64decode(encoded) == raw


def test_local_provider_default_endpoint(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    p = _LocalProvider()
    assert "20.150.215.227" in p._endpoint
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
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    respx.post("http://20.150.215.227/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": '{"verdict":"ok","description":"fine","severity":null,"reasoning":"ok"}'}}]
        })
    )

    p = _LocalProvider()
    result = await p.complete("You are a judge.", [{"role": "user", "content": "Hello"}])
    assert "verdict" in result


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_complete_sends_system_message(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    captured = {}

    def capture(request):
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    respx.post("http://20.150.215.227/v1/chat/completions").mock(side_effect=capture)

    p = _LocalProvider()
    await p.complete("SYSTEM PROMPT", [{"role": "user", "content": "Hello"}])

    messages = captured["body"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYSTEM PROMPT"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_complete_sends_correct_model(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_MODEL", "test-model-v1")
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    captured = {}

    def capture(request):
        import json as _json
        captured["body"] = _json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    respx.post("http://20.150.215.227/v1/chat/completions").mock(side_effect=capture)

    p = _LocalProvider()
    await p.complete("sys", [{"role": "user", "content": "hi"}])
    assert captured["body"]["model"] == "test-model-v1"


@pytest.mark.asyncio
@respx.mock
async def test_local_provider_raises_on_http_error(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)

    respx.post("http://20.150.215.227/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    p = _LocalProvider()
    with pytest.raises(httpx.HTTPStatusError):
        await p.complete("sys", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_llm_oracle_works_with_local_provider(monkeypatch):
    """LLMOracle with _LocalProvider injected returns correct verdict."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _FakeLocal:
        async def complete(self, system, messages, max_tokens=512):
            return '{"verdict":"noise","description":"loading","severity":null,"reasoning":"spinner"}'
        @staticmethod
        def image_block(png_bytes):
            return {"type": "image_url", "image_url": {"url": "data:image/png;base64,fake"}}

    oracle = LLMOracle(provider=_FakeLocal())
    verdict = await oracle.judge_screenshot(b"\x89PNG", context="test")
    assert verdict.verdict == "noise"
