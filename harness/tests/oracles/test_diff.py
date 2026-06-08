"""
Tests for oracles/diff.py

Unit tests only — no browser required.
Covers structural diff computation and oracle verdicts.
"""

import pytest
from harness.models import BugType, PageState, Severity
from harness.oracles.diff import DiffOracle, StructuralDiff, compute_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    url: str = "http://localhost:5230/",
    tree: dict = None,
    screenshot: bytes = b"\x89PNG",
) -> PageState:
    return PageState(
        url=url,
        screenshot=screenshot,
        a11y_tree=tree or {},
        timestamp=1.0,
    )


TREE_TWO_MEMOS = {
    "role": "main", "tag": "main",
    "children": [
        {"role": "list", "tag": "ul", "children": [
            {"role": "listitem", "tag": "li", "name": "Memo A", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
            {"role": "listitem", "tag": "li", "name": "Memo B", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
        ]},
    ],
}

TREE_THREE_MEMOS = {
    "role": "main", "tag": "main",
    "children": [
        {"role": "list", "tag": "ul", "children": [
            {"role": "listitem", "tag": "li", "name": "Memo A", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
            {"role": "listitem", "tag": "li", "name": "Memo B", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
            {"role": "listitem", "tag": "li", "name": "Memo C", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
        ]},
    ],
}

TREE_MEMO_A_PINNED = {
    "role": "main", "tag": "main",
    "children": [
        {"role": "list", "tag": "ul", "children": [
            {"role": "listitem", "tag": "li", "name": "Memo A", "children": [
                {"role": "button", "tag": "button", "name": "Unpin", "checked": True}
            ]},
            {"role": "listitem", "tag": "li", "name": "Memo B", "children": [
                {"role": "button", "tag": "button", "name": "Pin", "checked": False}
            ]},
        ]},
    ],
}

TREE_ALMOST_EMPTY = {
    "role": "main", "tag": "main",
    "children": [{"role": "paragraph", "tag": "p", "name": "Error"}],
}


# ---------------------------------------------------------------------------
# compute_diff
# ---------------------------------------------------------------------------

def test_diff_detects_no_change():
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_TWO_MEMOS)
    diff = compute_diff(before, after)
    assert not diff.anything_changed


def test_diff_detects_url_change():
    before = _state(url="http://localhost:5230/home")
    after  = _state(url="http://localhost:5230/explore")
    diff = compute_diff(before, after)
    assert diff.url_changed


def test_diff_detects_element_added():
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_THREE_MEMOS)
    diff = compute_diff(before, after)
    assert diff.element_delta > 0
    assert diff.anything_changed


def test_diff_detects_element_removed():
    before = _state(tree=TREE_THREE_MEMOS)
    after  = _state(tree=TREE_TWO_MEMOS)
    diff = compute_diff(before, after)
    assert diff.element_delta < 0


def test_diff_detects_state_attribute_change():
    # Use a tree where the node NAME stays the same but checked changes.
    # (Pin→Unpin changes the name, so we use a checkbox-style node instead.)
    before_tree = {"role": "main", "tag": "main", "children": [
        {"role": "checkbox", "tag": "input", "name": "pin_memo_A", "checked": False}
    ]}
    after_tree = {"role": "main", "tag": "main", "children": [
        {"role": "checkbox", "tag": "input", "name": "pin_memo_A", "checked": True}
    ]}
    diff = compute_diff(_state(tree=before_tree), _state(tree=after_tree))
    assert diff.state_changes


def test_diff_element_count_before_and_after():
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_THREE_MEMOS)
    diff = compute_diff(before, after)
    assert diff.element_count_before < diff.element_count_after


def test_diff_returns_structural_diff_type():
    diff = compute_diff(_state(), _state())
    assert isinstance(diff, StructuralDiff)


# ---------------------------------------------------------------------------
# DiffOracle.check — no LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_oracle_no_finding_when_change_expected_and_occurred():
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_THREE_MEMOS)
    result = await oracle.check(before, after, "create_memo", expect_change=True)
    assert result is None


@pytest.mark.asyncio
async def test_oracle_finding_when_nothing_changed_but_expected():
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_TWO_MEMOS)
    result = await oracle.check(before, after, "pin_memo", expect_change=True)
    assert result is not None
    assert result.severity == Severity.HIGH
    assert result.bug_type == BugType.LOGIC
    assert "No UI change" in result.title


@pytest.mark.asyncio
async def test_oracle_no_finding_when_no_change_and_not_expected():
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_TWO_MEMOS)
    result = await oracle.check(before, after, "view_memo", expect_change=False)
    assert result is None


@pytest.mark.asyncio
async def test_oracle_catastrophic_loss_detected():
    """Page losing >50% of elements is always flagged regardless of expect_change."""
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree=TREE_TWO_MEMOS)     # ~10 elements
    after  = _state(tree=TREE_ALMOST_EMPTY)  # 3 elements
    result = await oracle.check(before, after, "navigate", expect_change=False)
    assert result is not None
    assert result.severity == Severity.CRITICAL
    assert "lost" in result.title.lower() or "content" in result.title.lower()


@pytest.mark.asyncio
async def test_oracle_no_catastrophic_loss_on_small_pages():
    """Don't flag catastrophic loss when before-state had few elements."""
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree={"role": "main", "tag": "main"})   # 1 element
    after  = _state(tree=TREE_ALMOST_EMPTY)
    result = await oracle.check(before, after, "navigate", expect_change=True)
    # Should not trigger catastrophic threshold since before_count <= 5.
    assert result is None or result.severity != Severity.CRITICAL


# ---------------------------------------------------------------------------
# DiffOracle.check — with mock LLM
# ---------------------------------------------------------------------------

class _MockLLMOk:
    async def judge_diff(self, before, after, action):
        from harness.oracles.llm import OracleVerdict
        return OracleVerdict(verdict="ok", description="Expected", severity=None, reasoning="Fine.")


class _MockLLMBug:
    async def judge_diff(self, before, after, action):
        from harness.oracles.llm import OracleVerdict
        return OracleVerdict(verdict="bug", description="Wrong state", severity="high", reasoning="Did not change.")


class _MockLLMNoise:
    async def judge_diff(self, before, after, action):
        from harness.oracles.llm import OracleVerdict
        return OracleVerdict(verdict="noise", description="Loading", severity=None, reasoning="Transient.")


@pytest.mark.asyncio
async def test_oracle_llm_ok_suppresses_no_change_finding():
    """If LLM says ok, suppress the no-change finding."""
    oracle = DiffOracle(llm_oracle=_MockLLMOk())
    before = after = _state(tree=TREE_TWO_MEMOS)
    result = await oracle.check(before, after, "pin_memo", expect_change=True)
    assert result is None


@pytest.mark.asyncio
async def test_oracle_llm_noise_suppresses_no_change_finding():
    oracle = DiffOracle(llm_oracle=_MockLLMNoise())
    before = after = _state(tree=TREE_TWO_MEMOS)
    result = await oracle.check(before, after, "pin_memo", expect_change=True)
    assert result is None


@pytest.mark.asyncio
async def test_oracle_llm_bug_raises_finding_on_changed_state():
    oracle = DiffOracle(llm_oracle=_MockLLMBug())
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_MEMO_A_PINNED)
    result = await oracle.check(before, after, "pin_memo", expect_change=True)
    assert result is not None
    assert result.detected_by.value == "llm"
    assert result.severity == Severity.HIGH


@pytest.mark.asyncio
async def test_oracle_steps_include_action():
    oracle = DiffOracle(llm_oracle=None)
    before = _state(tree=TREE_TWO_MEMOS)
    after  = _state(tree=TREE_TWO_MEMOS)
    result = await oracle.check(before, after, "pin_memo", expect_change=True)
    assert result is not None
    assert "pin_memo" in result.steps
