"""
Tests for planner/action_identifier.py

Tests cover:
  - DOM-only grouping (no LLM) — the fast deterministic path
  - Ordinal stripping (pin_memo_1 + pin_memo_2 → one group)
  - LLM fallback when llm_group returns None
  - Empty input edge case
  - SemanticAction fields are populated correctly
"""

import pytest
from planner.action_identifier import (
    ActionIdentifier,
    SemanticAction,
    dom_group,
    _strip_ordinals,
    _group_key,
)


# ---------------------------------------------------------------------------
# _strip_ordinals
# ---------------------------------------------------------------------------

def test_strip_ordinals_removes_trailing_number():
    assert _strip_ordinals("pin memo 1") == "pin memo"
    assert _strip_ordinals("delete item 42") == "delete item"


def test_strip_ordinals_leaves_non_numeric():
    assert _strip_ordinals("create memo") == "create memo"
    assert _strip_ordinals("search") == "search"


def test_strip_ordinals_handles_empty():
    assert _strip_ordinals("") == ""


# ---------------------------------------------------------------------------
# _group_key
# ---------------------------------------------------------------------------

def test_group_key_normalises_label():
    el1 = {"role": "button", "label": "Pin Memo 1", "tag": "button"}
    el2 = {"role": "button", "label": "Pin Memo 2", "tag": "button"}
    assert _group_key(el1) == _group_key(el2)


def test_group_key_differentiates_roles():
    btn = {"role": "button", "label": "Save", "tag": "button"}
    link = {"role": "link",   "label": "Save", "tag": "a"}
    assert _group_key(btn) != _group_key(link)


def test_group_key_case_insensitive_label():
    el1 = {"role": "button", "label": "Create Memo", "tag": "button"}
    el2 = {"role": "button", "label": "create memo", "tag": "button"}
    assert _group_key(el1) == _group_key(el2)


# ---------------------------------------------------------------------------
# dom_group
# ---------------------------------------------------------------------------

ELEMENTS_WITH_DUPLICATES = [
    {"tag": "button", "role": "button", "label": "Pin Memo 1", "selector": "#pin-1"},
    {"tag": "button", "role": "button", "label": "Pin Memo 2", "selector": "#pin-2"},
    {"tag": "button", "role": "button", "label": "Pin Memo 3", "selector": "#pin-3"},
    {"tag": "button", "role": "button", "label": "Archive Memo 1", "selector": "#arch-1"},
    {"tag": "a",      "role": "link",   "label": "Home",        "selector": "a[href='/']"},
    {"tag": "input",  "role": "textbox","label": "Search",      "selector": "input[type=search]"},
]

ELEMENTS_ALL_UNIQUE = [
    {"tag": "button", "role": "button", "label": "Save",   "selector": "#save"},
    {"tag": "button", "role": "button", "label": "Cancel", "selector": "#cancel"},
    {"tag": "button", "role": "button", "label": "Delete", "selector": "#delete"},
]


def test_dom_group_deduplicates_identical_actions():
    """3 pin buttons → 1 semantic action."""
    actions = dom_group(ELEMENTS_WITH_DUPLICATES)
    names = [a.name for a in actions]
    # Should NOT have three separate pin entries.
    pin_actions = [n for n in names if "pin" in n.lower()]
    assert len(pin_actions) == 1


def test_dom_group_keeps_unique_actions():
    """Each unique button → its own action."""
    actions = dom_group(ELEMENTS_ALL_UNIQUE)
    assert len(actions) == 3


def test_dom_group_returns_semantic_action_instances():
    actions = dom_group(ELEMENTS_WITH_DUPLICATES)
    for a in actions:
        assert isinstance(a, SemanticAction)


def test_dom_group_populates_selector():
    actions = dom_group(ELEMENTS_WITH_DUPLICATES)
    for a in actions:
        assert a.representative_selector != ""


def test_dom_group_populates_raw_elements():
    """The raw_elements list holds all members of the group."""
    actions = dom_group(ELEMENTS_WITH_DUPLICATES)
    pin_action = next(a for a in actions if "pin" in a.name.lower())
    assert len(pin_action.raw_elements) == 3


def test_dom_group_empty_input():
    assert dom_group([]) == []


def test_dom_group_differentiates_link_from_button():
    """'Home' link and 'Home' button are different actions."""
    elements = [
        {"tag": "button", "role": "button", "label": "Home", "selector": "#home-btn"},
        {"tag": "a",      "role": "link",   "label": "Home", "selector": "a[href='/']"},
    ]
    actions = dom_group(elements)
    assert len(actions) == 2


# ---------------------------------------------------------------------------
# ActionIdentifier (no LLM)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_identifier_no_llm_returns_dom_groups():
    """Without LLM, ActionIdentifier returns DOM-grouped actions."""
    identifier = ActionIdentifier(llm_client=None)
    actions = await identifier.identify(ELEMENTS_WITH_DUPLICATES, screenshot=b"")
    assert len(actions) > 0
    assert all(isinstance(a, SemanticAction) for a in actions)


@pytest.mark.asyncio
async def test_identifier_no_llm_deduplicates():
    identifier = ActionIdentifier(llm_client=None)
    actions = await identifier.identify(ELEMENTS_WITH_DUPLICATES, screenshot=b"")
    pin_actions = [a for a in actions if "pin" in a.name.lower()]
    assert len(pin_actions) == 1


@pytest.mark.asyncio
async def test_identifier_empty_elements():
    identifier = ActionIdentifier(llm_client=None)
    actions = await identifier.identify([], screenshot=b"")
    assert actions == []


# ---------------------------------------------------------------------------
# ActionIdentifier with failing LLM → falls back to DOM grouping
# ---------------------------------------------------------------------------

class _FailingLLM:
    """Simulates an LLMOracle whose group_actions returns None (failure)."""
    async def group_actions(self, elements, screenshot):
        return None


@pytest.mark.asyncio
async def test_identifier_falls_back_on_llm_failure():
    """If LLM raises, ActionIdentifier falls back to DOM grouping silently."""
    identifier = ActionIdentifier(llm_client=_FailingLLM())
    actions = await identifier.identify(ELEMENTS_WITH_DUPLICATES, screenshot=b"fake_png")
    # Should still return DOM-grouped results, not raise.
    assert len(actions) > 0
    pin_actions = [a for a in actions if "pin" in a.name.lower()]
    assert len(pin_actions) == 1


# ---------------------------------------------------------------------------
# ActionIdentifier with mock LLM → uses LLM output
# ---------------------------------------------------------------------------

_MOCK_GROUPS = [
    {"name": "pin_a_memo",    "description": "Pin a memo to top",   "representative_selector": "#pin-1"},
    {"name": "archive_a_memo","description": "Archive a memo",       "representative_selector": "#arch-1"},
    {"name": "navigate_home", "description": "Go to home page",      "representative_selector": "a[href='/']"},
    {"name": "search_memos",  "description": "Search through memos", "representative_selector": "input[type=search]"},
]


class _MockLLM:
    """Simulates an LLMOracle that returns a hard-coded group_actions response."""

    async def group_actions(self, elements, screenshot):
        return _MOCK_GROUPS


@pytest.mark.asyncio
async def test_identifier_uses_llm_output_when_available():
    """When LLM succeeds, its grouping is used over DOM grouping."""
    identifier = ActionIdentifier(llm_client=_MockLLM())
    actions = await identifier.identify(ELEMENTS_WITH_DUPLICATES, screenshot=b"fake_png")
    names = {a.name for a in actions}
    assert "pin_a_memo" in names
    assert "archive_a_memo" in names


@pytest.mark.asyncio
async def test_identifier_llm_returns_semantic_action_instances():
    identifier = ActionIdentifier(llm_client=_MockLLM())
    actions = await identifier.identify(ELEMENTS_WITH_DUPLICATES, screenshot=b"fake_png")
    for a in actions:
        assert isinstance(a, SemanticAction)
        assert a.name != ""
        assert a.representative_selector != ""
