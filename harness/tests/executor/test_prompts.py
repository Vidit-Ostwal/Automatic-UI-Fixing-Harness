import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.executor.prompts import (
    ResolveOutcome,
    ResolveResult,
    _parse_resolve_response,
    build_resolve_instruction_messages,
    resolve_instruction,
    verify_instruction_outcome,
)


def test_parse_resolve_response_object():
    raw = json.dumps({
        "reason": "The Sign In link opens the login form.",
        "steps": [{"type": "click", "selector": "#sign-in"}],
    })
    result, kind, detail = _parse_resolve_response(raw)
    assert result == ResolveResult(
        reason="The Sign In link opens the login form.",
        steps=[{"type": "click", "selector": "#sign-in"}],
    )
    assert kind == ""
    assert detail == ""


def test_parse_resolve_response_legacy_array():
    raw = json.dumps([{"type": "click", "selector": "#sign-in"}])
    result, kind, detail = _parse_resolve_response(raw)
    assert result == ResolveResult(reason="", steps=[{"type": "click", "selector": "#sign-in"}])
    assert kind == ""


def test_parse_resolve_response_empty_steps():
    raw = json.dumps({"reason": "Signup form not on this page", "steps": []})
    result, kind, detail = _parse_resolve_response(raw)
    assert result is None
    assert kind == "empty_steps"
    assert detail == "Signup form not on this page"


def test_parse_resolve_response_json_error():
    result, kind, detail = _parse_resolve_response("not json at all")
    assert result is None
    assert kind == "json_parse_error"
    assert detail


@pytest.mark.asyncio
async def test_resolve_instruction_returns_reason_and_steps():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    llm.complete = AsyncMock(return_value=json.dumps({
        "reason": "Fill username then submit.",
        "steps": [
            {"type": "fill", "selector": "#user", "value": "alice"},
            {"type": "click", "selector": "#submit"},
        ],
    }))

    outcome = await resolve_instruction(llm, b"png", [{"selector": "#user"}], "Log in as alice")

    assert outcome.ok
    assert outcome.result.reason == "Fill username then submit."
    assert len(outcome.result.steps) == 2
    llm.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_instruction_reports_empty_steps_reason():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    llm.complete = AsyncMock(return_value=json.dumps({
        "reason": "No username field visible",
        "steps": [],
    }))

    outcome = await resolve_instruction(llm, b"png", [], "Log in as alice")

    assert not outcome.ok
    assert outcome.failure_kind == "empty_steps"
    assert outcome.failure_detail == "No username field visible"
    assert "No username field visible" in outcome.summary()


@pytest.mark.asyncio
async def test_resolve_instruction_reports_llm_error():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    llm.complete = AsyncMock(side_effect=RuntimeError("connection refused"))

    outcome = await resolve_instruction(llm, b"png", [], "Log in")

    assert not outcome.ok
    assert outcome.failure_kind == "llm_error"
    assert "connection refused" in outcome.failure_detail


@pytest.mark.asyncio
async def test_verify_instruction_outcome_parses_response():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    llm.complete = AsyncMock(return_value=json.dumps({
        "achieved": True,
        "explanation": "The dashboard is visible after login.",
    }))

    achieved, explanation = await verify_instruction_outcome(
        llm,
        "Log in",
        b"before",
        b"after",
        "Click submit after filling credentials.",
    )

    assert achieved is True
    assert "dashboard" in explanation.lower()


@pytest.mark.asyncio
async def test_verify_instruction_outcome_defaults_to_not_achieved_on_bad_json():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    llm.complete = AsyncMock(return_value="not json")

    achieved, explanation = await verify_instruction_outcome(
        llm,
        "Log in",
        b"before",
        b"after",
        "Click submit.",
    )

    assert achieved is False
    assert explanation


def test_resolve_messages_include_a11y_and_full_element_list():
    llm = MagicMock()
    llm.image_block = MagicMock(side_effect=lambda b: {"type": "image", "data": b})
    elements = [
        {"selector": f"#el-{i}", "role": "button", "label": str(i), "type": ""}
        for i in range(60)
    ]
    messages = build_resolve_instruction_messages(
        llm,
        b"png",
        elements,
        "Click save",
        a11y_tree={"role": "document", "name": "Dashboard"},
    )
    text = messages[0]["content"][1]["text"]
    assert "Accessibility tree:" in text
    assert "Interactive elements with REAL selectors" in text
    assert "Dashboard" in text
    assert "#el-0" in text
    assert text.index("Accessibility tree:") < text.index("#el-0")
