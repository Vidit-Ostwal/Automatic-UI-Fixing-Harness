"""
Tests for planner/action_identifier.py

Covers:
  - InteractionStep and SemanticAction data model
  - dom_group — buttons/links only (forms handled by LLM)
  - dom_group — deduplication by normalised label
  - dom_group — empty input
  - ActionIdentifier without LLM falls back to dom_group
  - ActionIdentifier with failing LLM (returns None) falls back to dom_group
  - ActionIdentifier with mock LLM uses analyze_page_actions output
"""

import json

import pytest
from harness.planner.action_identifier import (
    ActionIdentifier,
    InteractionStep,
    SemanticAction,
    _action_name,
    _strip_ordinals,
    dom_group,
    llm_group,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _el(tag, role, label, selector, placeholder="", **extra):
    return {"tag": tag, "role": role, "label": label,
            "selector": selector, "placeholder": placeholder, **extra}


SIDEBAR_ICON_ELS = [
    _el("a", "link", "earth", "#header-explore", id="header-explore", href="/explore-all"),
    _el("a", "link", "info", "#header-about", id="header-about", href="/about"),
    _el("a", "link", "circle user", "#header-auth", id="header-auth", href="/auth"),
]

RADIX_TRIGGER_EL = _el(
    "div", "button", "user round", "#radix-user",
    id="radix-user", data_slot="dropdown-menu-trigger",
)

BUTTON_ELS = [
    _el("button", "button", "Pin Memo 1",     "#pin-1"),
    _el("button", "button", "Pin Memo 2",     "#pin-2"),
    _el("button", "button", "Archive Memo 1", "#arch-1"),
    _el("a",      "link",   "Home",           "a[href='/']"),
]

FORM_ELS = [
    _el("input", "textbox", "Username", "input[name='username']"),
    _el("input", "textbox", "Password", "input[type='password']", placeholder="password"),
    _el("button", "button", "Sign up",  "button[type='submit']"),
]

MIXED_ELS = FORM_ELS + BUTTON_ELS

EMPTY_TREE = {}


# ---------------------------------------------------------------------------
# _strip_ordinals
# ---------------------------------------------------------------------------

def test_strip_ordinals_removes_trailing_number():
    assert _strip_ordinals("pin memo 1") == "pin memo"
    assert _strip_ordinals("delete item 42") == "delete item"

def test_strip_ordinals_leaves_non_numeric():
    assert _strip_ordinals("create memo") == "create memo"

def test_strip_ordinals_handles_empty():
    assert _strip_ordinals("") == ""


# ---------------------------------------------------------------------------
# dom_group — form elements (forms are LLM-only; dom_group returns buttons only)
# ---------------------------------------------------------------------------

def test_dom_group_form_elements_returns_only_buttons():
    # dom_group intentionally skips inputs — LLM handles form workflows.
    # FORM_ELS has one button ("Sign up") which should come through as a click.
    actions = dom_group(FORM_ELS)
    assert all(len(a.steps) == 1 and a.steps[0].type == "click" for a in actions)

def test_dom_group_form_elements_has_no_fill_steps():
    actions = dom_group(FORM_ELS)
    has_fill = any(s.type == "fill" for a in actions for s in a.steps)
    assert not has_fill


# ---------------------------------------------------------------------------
# dom_group — standalone buttons
# ---------------------------------------------------------------------------

def test_dom_group_buttons_produce_single_click_steps():
    actions = dom_group(BUTTON_ELS)
    for action in actions:
        assert len(action.steps) == 1
        assert action.steps[0].type == "click"

def test_dom_group_deduplicates_by_label():
    actions = dom_group(BUTTON_ELS)
    pin_actions = [a for a in actions if "pin" in a.name.lower()]
    assert len(pin_actions) == 1


def test_dom_group_sidebar_icon_links_stay_distinct():
    """Icon-only nav links must not collapse into a single 'link' action."""
    actions = dom_group(SIDEBAR_ICON_ELS)
    assert len(actions) == 3
    assert len({a.name for a in actions}) == 3


def test_dom_group_radix_dropdown_trigger():
    actions = dom_group([RADIX_TRIGGER_EL])
    assert len(actions) == 1
    assert actions[0].name == "user_round"
    assert actions[0].steps[0].selector == "#radix-user"


def test_dom_group_all_steps_are_interaction_steps():
    actions = dom_group(MIXED_ELS)
    for action in actions:
        for step in action.steps:
            assert isinstance(step, InteractionStep)

def test_dom_group_returns_semantic_action_instances():
    actions = dom_group(BUTTON_ELS)
    assert all(isinstance(a, SemanticAction) for a in actions)

def test_dom_group_empty_input():
    assert dom_group([]) == []


# ---------------------------------------------------------------------------
# ActionIdentifier — no LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identifier_no_llm_returns_actions():
    identifier = ActionIdentifier(llm_client=None)
    actions = await identifier.identify(BUTTON_ELS, EMPTY_TREE, screenshot=b"")
    assert len(actions) > 0
    assert all(isinstance(a, SemanticAction) for a in actions)

@pytest.mark.asyncio
async def test_identifier_no_llm_form_elements_returns_buttons_only():
    # Without LLM, form inputs are not expanded into fill workflows.
    # Only the submit button surfaces as a click action.
    identifier = ActionIdentifier(llm_client=None)
    actions = await identifier.identify(FORM_ELS, EMPTY_TREE, screenshot=b"")
    has_fill = any(any(s.type == "fill" for s in a.steps) for a in actions)
    assert not has_fill

@pytest.mark.asyncio
async def test_identifier_empty_elements():
    identifier = ActionIdentifier(llm_client=None)
    assert await identifier.identify([], EMPTY_TREE, screenshot=b"") == []


# ---------------------------------------------------------------------------
# ActionIdentifier — failing LLM falls back to dom_group
# ---------------------------------------------------------------------------

class _FailingOracle:
    async def complete(self, system, messages, max_tokens=512):
        raise RuntimeError("LLM failure")

    @staticmethod
    def image_block(png_bytes):
        return {"type": "image_url", "image_url": {"url": "fake"}}


@pytest.mark.asyncio
async def test_identifier_falls_back_on_llm_failure():
    identifier = ActionIdentifier(llm_client=_FailingOracle())
    actions = await identifier.identify(BUTTON_ELS, EMPTY_TREE, screenshot=b"fake")
    assert len(actions) > 0
    pin_actions = [a for a in actions if "pin" in a.name.lower()]
    assert len(pin_actions) == 1


# ---------------------------------------------------------------------------
# ActionIdentifier — mock LLM uses analyze_page_actions output
# ---------------------------------------------------------------------------

_MOCK_WORKFLOWS = [
    {
        "name": "sign_up",
        "description": "Create an account",
        "steps": [
            {"type": "fill",  "selector": "input[name='username']", "value": "harness_tester"},
            {"type": "fill",  "selector": "input[type='password']", "value": "Harness@2024!"},
            {"type": "click", "selector": "button[type='submit']"},
        ],
        "expected_outcome": "Redirected to main app",
    },
    {
        "name": "navigate_home",
        "description": "Go to the home page",
        "steps": [{"type": "click", "selector": "a[href='/']"}],
        "expected_outcome": "Home page loads",
    },
]


class _MockOracle:
    async def complete(self, system, messages, max_tokens=512):
        return json.dumps(_MOCK_WORKFLOWS)

    @staticmethod
    def image_block(png_bytes):
        return {"type": "image_url", "image_url": {"url": "fake"}}


@pytest.mark.asyncio
async def test_identifier_uses_llm_output_when_available():
    identifier = ActionIdentifier(llm_client=_MockOracle())
    actions = await identifier.identify(MIXED_ELS, EMPTY_TREE, screenshot=b"fake")
    names = {a.name for a in actions}
    assert "sign_up" in names
    assert "navigate_home" in names


@pytest.mark.asyncio
async def test_identifier_llm_sign_up_has_fill_steps():
    identifier = ActionIdentifier(llm_client=_MockOracle())
    actions = await identifier.identify(MIXED_ELS, EMPTY_TREE, screenshot=b"fake")
    signup = next(a for a in actions if a.name == "sign_up")
    fill_steps = [s for s in signup.steps if s.type == "fill"]
    assert len(fill_steps) == 2


@pytest.mark.asyncio
async def test_identifier_llm_last_step_of_form_is_click():
    identifier = ActionIdentifier(llm_client=_MockOracle())
    actions = await identifier.identify(MIXED_ELS, EMPTY_TREE, screenshot=b"fake")
    signup = next(a for a in actions if a.name == "sign_up")
    assert signup.steps[-1].type == "click"


@pytest.mark.asyncio
async def test_identifier_returns_semantic_action_instances():
    identifier = ActionIdentifier(llm_client=_MockOracle())
    actions = await identifier.identify(MIXED_ELS, EMPTY_TREE, screenshot=b"fake")
    for a in actions:
        assert isinstance(a, SemanticAction)
        for s in a.steps:
            assert isinstance(s, InteractionStep)


# ---------------------------------------------------------------------------
# ActionIdentifier — complement merge (LLM + dom_group for uncovered elements)
# ---------------------------------------------------------------------------

# Elements where LLM covers the form but ignores a standalone button.
_STANDALONE_EL = _el("button", "button", "Toggle language", "#lang-toggle")

_ELS_WITH_EXTRA = FORM_ELS + [_STANDALONE_EL]

# LLM returns only the form workflow; the standalone button is not referenced.
_FORM_ONLY_WORKFLOWS = [
    {
        "name": "sign_up",
        "description": "Create an account",
        "steps": [
            {"type": "fill",  "selector": "input[name='username']", "value": "tester"},
            {"type": "fill",  "selector": "input[type='password']", "value": "P@ss1"},
            {"type": "click", "selector": "button[type='submit']"},
        ],
        "expected_outcome": "Redirected to main app",
    },
]


class _FormOnlyOracle:
    async def complete(self, system, messages, max_tokens=512):
        return json.dumps(_FORM_ONLY_WORKFLOWS)

    @staticmethod
    def image_block(png_bytes):
        return {"type": "image_url", "image_url": {"url": "fake"}}


@pytest.mark.asyncio
async def test_complement_merge_includes_uncovered_element():
    # LLM covers form; dom_group should add the standalone button the LLM ignored.
    identifier = ActionIdentifier(llm_client=_FormOnlyOracle())
    actions = await identifier.identify(_ELS_WITH_EXTRA, EMPTY_TREE, screenshot=b"fake")
    names = {a.name for a in actions}
    assert "sign_up" in names
    assert "toggle_language" in names


@pytest.mark.asyncio
async def test_complement_merge_no_duplicates_for_covered_elements():
    # The submit button IS referenced in the LLM workflow — dom_group must not
    # add a second click action for it.
    identifier = ActionIdentifier(llm_client=_FormOnlyOracle())
    actions = await identifier.identify(_ELS_WITH_EXTRA, EMPTY_TREE, screenshot=b"fake")
    sign_up_actions = [a for a in actions if "sign" in a.name]
    assert len(sign_up_actions) == 1
