"""
Tests for planner/state_hasher.py

All tests are pure unit tests — no browser required.
"""

import pytest
from planner.state_hasher import (
    extract_structural_skeleton,
    interactive_fingerprint,
    normalise_url,
    state_hash,
)


# ---------------------------------------------------------------------------
# extract_structural_skeleton
# ---------------------------------------------------------------------------

def test_skeleton_keeps_role_and_tag():
    node = {"role": "button", "tag": "button", "name": "Click Me", "id": "btn1"}
    skeleton = extract_structural_skeleton(node)
    assert skeleton["role"] == "button"
    assert skeleton["tag"] == "button"


def test_skeleton_strips_dynamic_name():
    """Text content (name/label) must not appear in the skeleton."""
    node = {"role": "heading", "tag": "h1", "name": "My Dynamic Title"}
    skeleton = extract_structural_skeleton(node)
    assert "name" not in skeleton
    assert "My Dynamic Title" not in str(skeleton)


def test_skeleton_strips_id():
    node = {"role": "button", "tag": "button", "id": "btn-abc123"}
    skeleton = extract_structural_skeleton(node)
    assert "id" not in skeleton


def test_skeleton_preserves_state_attributes():
    """aria-expanded / aria-checked must be preserved — they signal state."""
    node = {"role": "button", "tag": "button", "expanded": True, "checked": False}
    skeleton = extract_structural_skeleton(node)
    assert skeleton.get("expanded") is True
    assert skeleton.get("checked") is False


def test_skeleton_recurses_into_children():
    node = {
        "role": "list",
        "tag": "ul",
        "children": [
            {"role": "listitem", "tag": "li", "name": "Item A"},
            {"role": "listitem", "tag": "li", "name": "Item B"},
        ],
    }
    skeleton = extract_structural_skeleton(node)
    assert len(skeleton["children"]) == 2
    assert skeleton["children"][0]["role"] == "listitem"
    # names must be stripped from children too
    assert "name" not in skeleton["children"][0]


def test_skeleton_depth_limit():
    """Skeleton stops recursing at depth 12 to avoid stack overflow."""
    # Build a chain 20 levels deep.
    node = {"role": "div", "tag": "div"}
    current = node
    for _ in range(20):
        child = {"role": "div", "tag": "div"}
        current["children"] = [child]
        current = child

    # Should not raise; just truncates.
    skeleton = extract_structural_skeleton(node)
    assert isinstance(skeleton, dict)


def test_skeleton_empty_node():
    assert extract_structural_skeleton({}) == {}
    assert extract_structural_skeleton(None) == {}


# ---------------------------------------------------------------------------
# normalise_url
# ---------------------------------------------------------------------------

def test_normalise_url_strips_query():
    url = "http://localhost:5230/explore?tag=python&sort=newest"
    assert normalise_url(url) == "http://localhost:5230/explore"


def test_normalise_url_strips_fragment():
    url = "http://localhost:5230/home#section2"
    assert normalise_url(url) == "http://localhost:5230/home"


def test_normalise_url_strips_trailing_slash():
    url = "http://localhost:5230/home/"
    assert normalise_url(url) == "http://localhost:5230/home"


def test_normalise_url_preserves_path():
    url = "http://localhost:5230/m/abc123/edit"
    assert normalise_url(url) == "http://localhost:5230/m/abc123/edit"


def test_normalise_url_about_blank():
    assert normalise_url("about:blank") == "about://blank"


# ---------------------------------------------------------------------------
# state_hash
# ---------------------------------------------------------------------------

TREE_A = {
    "role": "main",
    "tag": "main",
    "children": [
        {"role": "button", "tag": "button", "name": "Create Memo"},
        {"role": "list",   "tag": "ul",     "name": ""},
    ],
}

TREE_B_SAME_STRUCTURE = {
    "role": "main",
    "tag": "main",
    "children": [
        {"role": "button", "tag": "button", "name": "New Note"},    # different label
        {"role": "list",   "tag": "ul",     "name": "Memos"},       # different label
    ],
}

TREE_C_DIFFERENT_STRUCTURE = {
    "role": "main",
    "tag": "main",
    "children": [
        {"role": "button", "tag": "button", "name": "Create Memo"},
        {"role": "list",   "tag": "ul",     "name": ""},
        {"role": "button", "tag": "button", "name": "Extra Button"},  # extra element
    ],
}


def test_same_structure_different_labels_same_hash():
    """Two pages with identical DOM structure but different text → same hash."""
    h1 = state_hash("http://localhost:5230/", TREE_A)
    h2 = state_hash("http://localhost:5230/", TREE_B_SAME_STRUCTURE)
    assert h1 == h2


def test_different_structure_different_hash():
    """Adding an extra element changes the hash."""
    h1 = state_hash("http://localhost:5230/", TREE_A)
    h3 = state_hash("http://localhost:5230/", TREE_C_DIFFERENT_STRUCTURE)
    assert h1 != h3


def test_different_url_path_different_hash():
    """Same tree on different routes → different hash."""
    h1 = state_hash("http://localhost:5230/home", TREE_A)
    h2 = state_hash("http://localhost:5230/explore", TREE_A)
    assert h1 != h2


def test_query_params_ignored_in_hash():
    """Query params are stripped — same page regardless of search terms."""
    h1 = state_hash("http://localhost:5230/explore?q=python", TREE_A)
    h2 = state_hash("http://localhost:5230/explore?q=rust", TREE_A)
    assert h1 == h2


def test_hash_is_deterministic():
    """Calling state_hash twice with identical inputs yields the same result."""
    h1 = state_hash("http://localhost:5230/", TREE_A)
    h2 = state_hash("http://localhost:5230/", TREE_A)
    assert h1 == h2


def test_hash_is_hex_string():
    h = state_hash("http://localhost:5230/", TREE_A)
    assert isinstance(h, str)
    assert len(h) == 32
    assert all(c in "0123456789abcdef" for c in h)


def test_state_attribute_changes_hash():
    """Expanding a dropdown (aria-expanded=True) is a different state."""
    collapsed = {"role": "button", "tag": "button", "expanded": False}
    expanded  = {"role": "button", "tag": "button", "expanded": True}
    h1 = state_hash("http://localhost:5230/", collapsed)
    h2 = state_hash("http://localhost:5230/", expanded)
    assert h1 != h2


def test_interactive_fingerprint_changes_hash():
    """New interactive controls change the hash even when skeleton is identical."""
    elements_before = [{"role": "button", "selector": '[aria-label="New memo"]'}]
    elements_after = elements_before + [
        {"role": "button", "selector": '[aria-label="Pin"]'},
    ]
    h1 = state_hash("http://localhost:5230/", TREE_A, elements_before)
    h2 = state_hash("http://localhost:5230/", TREE_A, elements_after)
    assert h1 != h2


def test_interactive_fingerprint_is_deterministic():
    elements = [
        {"role": "button", "selector": "#b"},
        {"role": "button", "selector": "#a"},
    ]
    assert interactive_fingerprint(elements) == interactive_fingerprint(list(reversed(elements)))
