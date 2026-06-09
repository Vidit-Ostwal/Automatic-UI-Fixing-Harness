"""
Shared page context serialization for LLM prompts.

Used by both the BFS planner (ActionIdentifier) and the goal executor so
they present the same DOM snapshot to the model: accessibility tree +
interactive element list with identical size limits.
"""

from __future__ import annotations

import json

from harness.utils.elements import serialize_elements

# Match harness/planner/prompts.py build_page_actions_messages limits.
A11Y_TREE_MAX_CHARS = 3000
ELEMENT_LIST_MAX_CHARS = 2000
ELEMENT_COUNT_LIMIT = 2000


def format_a11y_tree(a11y_tree: dict | None) -> str:
    if not a11y_tree:
        return "{}"
    return json.dumps(a11y_tree, indent=2)[:A11Y_TREE_MAX_CHARS]


def format_element_list(elements: list[dict] | None) -> str:
    serialized = serialize_elements(elements or [], limit=ELEMENT_COUNT_LIMIT)
    return serialized[:ELEMENT_LIST_MAX_CHARS]
