"""
Tests for oracles/logic.py — BFS crash detection only.
"""

from harness.models import PageState
from harness.oracles.logic import LogicOracle


def _make_state(
    a11y_tree: dict = None,
    console_errors: list[str] = None,
    screenshot: bytes = b"\x89PNG",
) -> PageState:
    return PageState(
        url="http://localhost:5230/",
        screenshot=screenshot,
        a11y_tree=a11y_tree or {},
        console_errors=console_errors or [],
        timestamp=1.0,
    )


HEALTHY_TREE = {
    "role": "main", "tag": "main",
    "children": [
        {"role": "listitem", "tag": "li", "name": "My memo", "children": []},
    ],
}


def test_no_crash_passes_on_healthy_page():
    oracle = LogicOracle()
    after = _make_state(HEALTHY_TREE)
    assert oracle.check_no_crash(after, "create memo") is None


def test_no_crash_detects_pageerror():
    oracle = LogicOracle()
    after = _make_state(
        HEALTHY_TREE,
        console_errors=["[pageerror] TypeError: Cannot read property of undefined"],
    )
    result = oracle.check_no_crash(after, "submit form")
    assert result is not None
    assert result.severity.value == "critical"
    assert "crash" in result.title.lower() or "JavaScript" in result.title


def test_no_crash_detects_blank_page():
    oracle = LogicOracle()
    after = _make_state({})
    result = oracle.check_no_crash(after, "navigate")
    assert result is not None
    assert result.severity.value == "critical"


def test_no_crash_ignores_non_critical_console_errors():
    oracle = LogicOracle()
    after = _make_state(
        HEALTHY_TREE,
        console_errors=["[warning] Deprecation notice for something"],
    )
    assert oracle.check_no_crash(after, "navigate") is None
