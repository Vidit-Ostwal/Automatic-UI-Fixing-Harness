"""
Tests for oracles/logic.py

All tests are pure unit tests — no browser required.
PageState objects are constructed directly with synthetic a11y trees.
"""

import pytest
from harness.models import BugType, PageState, Severity
from harness.oracles.logic import LogicOracle, _page_text, _count_role, _find_by_role_and_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    a11y_tree: dict = None,
    console_errors: list[str] = None,
    screenshot: bytes = b"\x89PNG",
    url: str = "http://localhost:5230/",
) -> PageState:
    return PageState(
        url=url,
        screenshot=screenshot,
        a11y_tree=a11y_tree or {},
        console_errors=console_errors or [],
        timestamp=1.0,
    )


MEMO_LIST_TREE = {
    "role": "main", "tag": "main",
    "children": [
        {
            "role": "list", "tag": "ul",
            "children": [
                {"role": "listitem", "tag": "li", "name": "My first memo content",
                 "children": [
                     {"role": "button", "tag": "button", "name": "Pin",
                      "expanded": False, "checked": False}
                 ]},
                {"role": "listitem", "tag": "li", "name": "Another memo",
                 "children": [
                     {"role": "button", "tag": "button", "name": "Pin",
                      "expanded": False, "checked": False}
                 ]},
            ],
        }
    ],
}

MEMO_LIST_WITH_NEW = {
    "role": "main", "tag": "main",
    "children": [
        {
            "role": "list", "tag": "ul",
            "children": [
                {"role": "listitem", "tag": "li", "name": "My first memo content", "children": []},
                {"role": "listitem", "tag": "li", "name": "Another memo", "children": []},
                {"role": "listitem", "tag": "li", "name": "Brand new memo text", "children": []},
            ],
        }
    ],
}

MEMO_LIST_AFTER_DELETE = {
    "role": "main", "tag": "main",
    "children": [
        {
            "role": "list", "tag": "ul",
            "children": [
                {"role": "listitem", "tag": "li", "name": "Another memo", "children": []},
            ],
        }
    ],
}

PINNED_TREE = {
    "role": "main", "tag": "main",
    "children": [
        {
            "role": "list", "tag": "ul",
            "children": [
                {"role": "listitem", "tag": "li", "name": "My first memo content",
                 "children": [
                     {"role": "button", "tag": "button", "name": "Unpin",
                      "expanded": False, "checked": True}
                 ]},
            ],
        }
    ],
}

CRASHED_TREE = {}
TREE_WITH_CRASH_ERROR = {"role": "main", "tag": "main", "children": []}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def test_page_text_flattens_names():
    state = _make_state(MEMO_LIST_TREE)
    text = _page_text(state)
    assert "My first memo content" in text
    assert "Another memo" in text
    assert "Pin" in text


def test_page_text_empty_tree():
    assert _page_text(_make_state({})) == ""


def test_count_role_counts_listitem():
    state = _make_state(MEMO_LIST_TREE)
    assert _count_role(state, "listitem") == 2


def test_count_role_counts_button():
    state = _make_state(MEMO_LIST_TREE)
    assert _count_role(state, "button") == 2


def test_count_role_returns_zero_for_absent_role():
    state = _make_state(MEMO_LIST_TREE)
    assert _count_role(state, "dialog") == 0


def test_find_by_role_and_name_finds_node():
    state = _make_state(MEMO_LIST_TREE)
    node = _find_by_role_and_name(state, "button", "Pin")
    assert node is not None
    assert node["role"] == "button"


def test_find_by_role_and_name_case_insensitive():
    state = _make_state(MEMO_LIST_TREE)
    node = _find_by_role_and_name(state, "button", "pin")
    assert node is not None


def test_find_by_role_and_name_returns_none_when_absent():
    state = _make_state(MEMO_LIST_TREE)
    assert _find_by_role_and_name(state, "button", "Delete") is None


# ---------------------------------------------------------------------------
# check_text_appeared
# ---------------------------------------------------------------------------

def test_text_appeared_passes_when_present():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_WITH_NEW)
    result = oracle.check_text_appeared(after, "Brand new memo text", "create memo")
    assert result is None


def test_text_appeared_fails_when_absent():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_TREE)
    result = oracle.check_text_appeared(after, "Brand new memo text", "create memo")
    assert result is not None
    assert result.bug_type == BugType.LOGIC
    assert result.severity == Severity.HIGH
    assert "Brand new memo text" in result.title


def test_text_appeared_case_insensitive():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_WITH_NEW)
    result = oracle.check_text_appeared(after, "BRAND NEW MEMO TEXT", "create memo")
    assert result is None


# ---------------------------------------------------------------------------
# check_text_disappeared
# ---------------------------------------------------------------------------

def test_text_disappeared_passes_when_gone():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_AFTER_DELETE)
    result = oracle.check_text_disappeared(after, "My first memo content", "delete memo")
    assert result is None


def test_text_disappeared_fails_when_still_present():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_TREE)
    result = oracle.check_text_disappeared(after, "My first memo content", "delete memo")
    assert result is not None
    assert result.severity == Severity.HIGH
    assert "still visible" in result.title.lower() or "My first memo" in result.title


# ---------------------------------------------------------------------------
# check_state_toggled
# ---------------------------------------------------------------------------

def test_state_toggled_passes_when_label_changed():
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)      # button says "Pin"
    after  = _make_state(PINNED_TREE)         # button says "Unpin"
    result = oracle.check_state_toggled(before, after, "button", "Pin", "click pin")
    assert result is None


def test_state_toggled_fails_when_nothing_changed():
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)
    after  = _make_state(MEMO_LIST_TREE)      # identical — pin had no effect
    result = oracle.check_state_toggled(before, after, "button", "Pin", "click pin")
    assert result is not None
    assert result.severity == Severity.HIGH


def test_state_toggled_passes_when_element_disappeared():
    """If the element is gone after the action, state definitely changed."""
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)
    after  = _make_state(MEMO_LIST_AFTER_DELETE)
    result = oracle.check_state_toggled(before, after, "button", "Pin", "delete")
    assert result is None


def test_state_toggled_skips_when_element_not_found_before():
    """Nothing to compare if the element didn't exist in the before state."""
    oracle = LogicOracle()
    before = _make_state({"role": "main", "tag": "main"})
    after  = _make_state(MEMO_LIST_TREE)
    result = oracle.check_state_toggled(before, after, "button", "Pin", "something")
    assert result is None


# ---------------------------------------------------------------------------
# check_count_changed
# ---------------------------------------------------------------------------

def test_count_changed_passes_on_correct_delta():
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)
    after  = _make_state(MEMO_LIST_WITH_NEW)
    result = oracle.check_count_changed(before, after, "listitem", +1, "create memo")
    assert result is None


def test_count_changed_fails_on_wrong_delta():
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)
    after  = _make_state(MEMO_LIST_TREE)   # no change
    result = oracle.check_count_changed(before, after, "listitem", +1, "create memo")
    assert result is not None
    assert "count" in result.title.lower() or "item" in result.title.lower()


def test_count_changed_delete():
    oracle = LogicOracle()
    before = _make_state(MEMO_LIST_TREE)
    after  = _make_state(MEMO_LIST_AFTER_DELETE)
    result = oracle.check_count_changed(before, after, "listitem", -1, "delete memo")
    assert result is None


# ---------------------------------------------------------------------------
# check_no_crash
# ---------------------------------------------------------------------------

def test_no_crash_passes_on_healthy_page():
    oracle = LogicOracle()
    after = _make_state(MEMO_LIST_TREE)
    assert oracle.check_no_crash(after, "create memo") is None


def test_no_crash_detects_pageerror():
    oracle = LogicOracle()
    after = _make_state(
        MEMO_LIST_TREE,
        console_errors=["[pageerror] TypeError: Cannot read property of undefined"],
    )
    result = oracle.check_no_crash(after, "submit form")
    assert result is not None
    assert result.severity.value == "critical"
    assert "crash" in result.title.lower() or "JavaScript" in result.title


def test_no_crash_detects_blank_page():
    oracle = LogicOracle()
    after = _make_state(CRASHED_TREE)
    result = oracle.check_no_crash(after, "navigate")
    assert result is not None
    assert result.severity.value == "critical"


def test_no_crash_ignores_non_critical_console_errors():
    oracle = LogicOracle()
    after = _make_state(
        MEMO_LIST_TREE,
        console_errors=["[warning] Deprecation notice for something"],
    )
    assert oracle.check_no_crash(after, "navigate") is None
