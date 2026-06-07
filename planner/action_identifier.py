"""
Action identifier — turns a raw list of DOM elements into a deduplicated
set of semantic actions.

Two-layer pipeline:
  1. DOM pre-grouping  (deterministic, no LLM)
     Groups elements by (tag, role, inferred-type) so obviously identical
     elements (10 pin buttons) collapse immediately.

  2. LLM semantic grouping  (optional, uses LLMOracle)
     Sends the pre-grouped candidates + screenshot to the oracle.
     The LLM collapses remaining duplicates and assigns human-readable names.

If no LLM oracle is supplied the identifier falls back to DOM grouping
only — useful for tests and offline runs.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oracles.llm import LLMOracle


@dataclass
class SemanticAction:
    name: str                          # e.g. "pin_a_memo"
    description: str                   # e.g. "Pin a memo to the top of the list"
    representative_selector: str       # selector to click for this action
    raw_elements: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DOM pre-grouping (no LLM)
# ---------------------------------------------------------------------------

_ROLE_TYPE_KEY = {
    "button": "button",
    "link": "link",
    "a": "link",
    "textbox": "input_text",
    "input": "input_text",
    "checkbox": "input_checkbox",
    "combobox": "select",
    "select": "select",
    "tab": "tab",
    "menuitem": "menuitem",
    "switch": "switch",
}


def _group_key(el: dict) -> str:
    """Canonical group key for an element based on its role+label."""
    role = el.get("role", el.get("tag", "unknown"))
    label = el.get("label", "").strip().lower()
    # Normalise numeric suffixes so "Pin memo 1" and "Pin memo 2" share a key.
    label = _strip_ordinals(label)
    return f"{_ROLE_TYPE_KEY.get(role, role)}::{label}"


def _strip_ordinals(text: str) -> str:
    """Remove trailing numbers and common list indicators from labels."""
    import re
    return re.sub(r"\s+\d+$", "", text).strip()


def dom_group(elements: list[dict]) -> list[SemanticAction]:
    """
    Group elements by (role, normalised-label) without LLM.
    Returns one SemanticAction per unique group, using the first element
    in each group as the representative.
    """
    groups: dict[str, list[dict]] = {}
    for el in elements:
        key = _group_key(el)
        groups.setdefault(key, []).append(el)

    actions = []
    for key, members in groups.items():
        rep = members[0]
        role, label = key.split("::", 1)
        name = (label or role).replace(" ", "_").replace("-", "_")[:40] or role
        actions.append(
            SemanticAction(
                name=name,
                description=f"{role}: {label}" if label else role,
                representative_selector=rep.get("selector", rep.get("tag", "button")),
                raw_elements=members,
            )
        )
    return actions


# ---------------------------------------------------------------------------
# LLM semantic grouping
# ---------------------------------------------------------------------------

async def llm_group(
    elements: list[dict],
    screenshot: bytes,
    oracle: "LLMOracle",
) -> list[SemanticAction] | None:
    """
    Send elements + screenshot to the LLMOracle for semantic grouping.
    Returns None on any failure so the caller can fall back to dom_group().
    """
    groups = await oracle.group_actions(elements, screenshot)
    if groups is None:
        return None
    return [
        SemanticAction(
            name=g["name"],
            description=g.get("description", g["name"]),
            representative_selector=g["representative_selector"],
            raw_elements=[
                e for e in elements
                if e.get("selector") == g.get("representative_selector")
            ],
        )
        for g in groups
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ActionIdentifier:
    """
    Identifies distinct semantic actions available on the current page.

    Usage:
        identifier = ActionIdentifier(llm_client=llm_oracle)  # LLMOracle instance
        actions = await identifier.identify(elements, screenshot)

    Without an LLM oracle it falls back to pure DOM grouping — useful for
    tests and offline runs.
    """

    def __init__(self, llm_client: "LLMOracle | None" = None):
        self._llm = llm_client

    async def identify(
        self,
        elements: list[dict],
        screenshot: bytes,
    ) -> list[SemanticAction]:
        """
        Return a deduplicated list of SemanticActions for the given elements.
        Tries LLM grouping first; falls back to DOM grouping on any failure.
        """
        if not elements:
            return []

        if self._llm is not None:
            result = await llm_group(elements, screenshot, self._llm)
            if result is not None:
                return result

        return dom_group(elements)
