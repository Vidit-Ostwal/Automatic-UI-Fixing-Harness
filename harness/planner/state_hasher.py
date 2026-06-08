"""
State hasher for BFS cycle detection.

Converts (URL, a11y_tree) → a stable MD5 hash that represents the
*structural shape* of a page — independent of dynamic content like
memo text, timestamps, or user-specific labels.

Two pages are considered the same state if they have the same URL path
and the same DOM skeleton (element types, roles, nesting depth) even if
the visible text differs.
"""

import hashlib
import json
import re

from harness.utils.url import normalise_url


def extract_structural_skeleton(node: dict, depth: int = 0) -> dict:
    """
    Recursively strip dynamic values from an a11y tree node.

    Keeps:  role, tag, structural state attributes (aria-expanded,
            aria-checked, aria-selected), child count and sub-structure.
    Strips: name/label text, id values, href targets, placeholder text.
    """
    if not node or depth > 12:
        return {}

    skeleton: dict = {}

    if "role" in node:
        skeleton["role"] = node["role"]
    if "tag" in node:
        skeleton["tag"] = node["tag"]

    # Preserve state attributes — they signal different visual/logic states.
    for attr in ("expanded", "checked", "selected", "disabled", "required"):
        if attr in node:
            skeleton[attr] = node[attr]

    children = node.get("children", [])
    if children:
        skeleton["children"] = [
            extract_structural_skeleton(c, depth + 1) for c in children
        ]

    return skeleton


def interactive_fingerprint(elements: list[dict]) -> str:
    """
    Stable digest of which interactive controls are present on the page.

    Captures selector + role pairs so a new button or enabled/disabled control
    can produce a different state hash even when the a11y skeleton is unchanged.
    """
    if not elements:
        return ""
    parts = sorted(
        f"{e.get('role', '')}::{e.get('selector', '')}"
        for e in elements
        if e.get("selector")
    )
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def state_hash(
    url: str,
    a11y_tree: dict,
    interactive_elements: list[dict] | None = None,
) -> str:
    """
    Return a stable hex digest for a (URL, a11y_tree) pair.
    Used by the BFS explorer to detect already-visited states.

    When interactive_elements is provided, their fingerprint is mixed in so
    control-level changes (new buttons, toggled disabled state) can diverge
    even if the structural skeleton is identical.
    """
    path = normalise_url(url)
    skeleton = extract_structural_skeleton(a11y_tree)
    # json.dumps with sort_keys gives a deterministic string for any dict.
    payload = f"{path}:{json.dumps(skeleton, sort_keys=True)}"
    if interactive_elements is not None:
        payload += f":{interactive_fingerprint(interactive_elements)}"
    return hashlib.md5(payload.encode()).hexdigest()
