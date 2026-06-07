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
from urllib.parse import urlparse


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


def normalise_url(url: str) -> str:
    """
    Keep scheme + host + path. Drop query params and fragments.
    For SPAs these are often ephemeral (search terms, scroll position).
    """
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")


def state_hash(url: str, a11y_tree: dict) -> str:
    """
    Return a stable hex digest for a (URL, a11y_tree) pair.
    Used by the BFS explorer to detect already-visited states.
    """
    path = normalise_url(url)
    skeleton = extract_structural_skeleton(a11y_tree)
    # json.dumps with sort_keys gives a deterministic string for any dict.
    payload = f"{path}:{json.dumps(skeleton, sort_keys=True)}"
    return hashlib.md5(payload.encode()).hexdigest()
