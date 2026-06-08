"""
Action identifier — converts a page's DOM + screenshot into a list of
complete user workflows (not just clickable elements).

A workflow is a sequence of interactions that together accomplish one
user intention:
  - A standalone button → single click
  - A form            → fill every required field + click submit
  - A search bar      → type a query + press Enter

Two-layer pipeline
------------------
1. LLM analysis  (primary, uses LLMOracle.analyze_page_actions)
   Sends the real element list + a11y tree + screenshot to the LLM, which
   reasons about distinct workflows and infers contextually appropriate fill
   values for each field from its label, placeholder, and surrounding UI.
   Selectors are constrained to the real element list (no hallucination).

2. DOM grouping  (fallback, no LLM)
   Used when LLM is unavailable or fails. Handles standalone buttons and
   links only — forms require LLM reasoning to fill correctly.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

_log = logging.getLogger(__name__)

# ISO date labels like "2026-06-01" — individual calendar-day buttons.
# The LLM handles these as a single "select_date" workflow; adding each day
# as a separate dom_group action creates dozens of redundant BFS branches.
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")

# Strips trailing date/ordinal/numeric tokens from a snake_case action name
# to produce a "semantic stem" used for deduplication.
# e.g. "select_date_june_1st_2026" → "select_date"
#      "click_item_3"              → "click_item"
_REPETITIVE_SUFFIX_RE = re.compile(
    r"(_\d{4}|_(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*"
    r"|_\d{1,2}(st|nd|rd|th)|_\d+)+$"
)


def _semantic_stem(name: str) -> str:
    return _REPETITIVE_SUFFIX_RE.sub("", name)


def _collapse_repetitive(actions: list["SemanticAction"]) -> list["SemanticAction"]:
    """
    Keep only the first action per semantic stem.
    Actions like select_date_june_1st_2026 … select_date_june_30th_2026
    all reduce to the stem 'select_date' — only the first survives.
    """
    seen_stems: set[str] = set()
    result: list["SemanticAction"] = []
    dropped = 0
    for action in actions:
        stem = _semantic_stem(action.name)
        if stem in seen_stems:
            dropped += 1
            continue
        seen_stems.add(stem)
        result.append(action)
    if dropped:
        _log.info("AI:  collapsed %d repetitive action(s) (same semantic stem)", dropped)
    return result

if TYPE_CHECKING:
    from oracles.llm import LLMOracle


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InteractionStep:
    type: str       # "fill" | "click" | "select" | "press"
    selector: str
    value: str = ""  # text for fill/select; key name for press; empty for click


@dataclass
class SemanticAction:
    name: str
    description: str
    steps: list[InteractionStep]
    expected_outcome: str = ""


# ---------------------------------------------------------------------------
# DOM grouping (no LLM)
# ---------------------------------------------------------------------------

_INPUT_ROLES  = {"textbox", "input_text", "searchbox", "input"}
_INPUT_TAGS   = {"input", "textarea"}
# combobox is a clickable control (language picker, theme toggle, dropdown menu);
# treat it like a button for exploration rather than a field to fill.
_BUTTON_ROLES = {"button", "menuitem", "tab", "switch", "option", "combobox"}
_SUBMIT_HINTS = {"submit", "sign", "log in", "login", "register", "create", "save",
                 "send", "confirm", "continue", "next", "done", "ok"}
_LINK_ROLES   = {"link"}


def _strip_ordinals(text: str) -> str:
    return re.sub(r"\s+\d+$", "", text).strip()


def _action_name(label: str, role: str) -> str:
    base = _strip_ordinals(label.lower()) or role
    return re.sub(r"[^a-z0-9]+", "_", base)[:40].strip("_") or role


def _element_label(el: dict) -> str:
    """Best human-readable label for action naming and descriptions."""
    label = (el.get("label") or "").strip()
    if label:
        return label
    el_id = (el.get("id") or "").strip()
    if el_id:
        return re.sub(r"^header[-_]", "", el_id).replace("-", " ").replace("_", " ")
    href = (el.get("href") or "").strip()
    if href:
        segment = href.rstrip("/").split("/")[-1]
        if segment:
            return segment.replace("-", " ").replace("_", " ")
    sel = el.get("selector", "")
    if sel.startswith("#"):
        return sel[1:].replace("-", " ").replace("_", " ")
    return el.get("tag", "element")


def _action_name_for_element(el: dict) -> str:
    """
    Unique snake_case action name. Icon-only sidebar links often share an empty
    label — fall back to id/href/selector so each nav item stays distinct.
    """
    name = _action_name(_element_label(el), el.get("role", "click"))
    if name not in ("link", "button", "a", "click"):
        return name
    sel = el.get("selector", "")
    slug = re.sub(r"[^a-z0-9]+", "_", sel.lower()).strip("_")[:40]
    return slug or name


def dom_group(elements: list[dict]) -> list[SemanticAction]:
    """
    Build SemanticActions for standalone buttons and links only.

    Forms (anything involving text inputs) are intentionally excluded here —
    the LLM path handles those because it can reason about what values to fill
    based on field labels, placeholders, and surrounding context.

    This function is the fallback for when no LLM is available, or as a
    complement to llm_group() for pages with no input fields.
    """
    # Skip individual calendar-day buttons — too numerous and already covered
    # by any LLM "select_date" workflow.
    def _is_calendar_date(el: dict) -> bool:
        return bool(_ISO_DATE_RE.match(el.get("label", "")))

    submits = [e for e in elements
               if e.get("role", "") in _BUTTON_ROLES
               and not _is_calendar_date(e)
               and any(h in e.get("label", "").lower() for h in _SUBMIT_HINTS)]
    buttons = [e for e in elements
               if e.get("role", "") in _BUTTON_ROLES and e not in submits
               and not _is_calendar_date(e)]
    links   = [e for e in elements if e.get("role", "") in _LINK_ROLES]

    actions: list[SemanticAction] = []
    seen_names: set[str] = set()

    for el in buttons + links + submits:
        sel = el.get("selector", "")
        if not sel:
            continue
        label = _element_label(el)
        name  = _action_name_for_element(el)
        if name in seen_names:
            continue
        seen_names.add(name)
        actions.append(SemanticAction(
            name=name,
            description=f"Click: {label or sel}",
            steps=[InteractionStep(type="click", selector=sel)],
        ))

    return actions


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

async def llm_group(
    elements: list[dict],
    a11y_tree: dict,
    screenshot: bytes,
    oracle: "LLMOracle",
    explored_context: str = "",
) -> list[SemanticAction] | None:
    """
    Ask the LLMOracle to reason about complete workflows on this page.
    Returns None on any failure so the caller falls back to dom_group().
    """
    groups = await oracle.analyze_page_actions(
        a11y_tree, screenshot, elements=elements, explored_context=explored_context
    )
    if groups is None:
        return None

    selector_map = {e.get("label", ""): e.get("selector", "")
                    for e in elements if e.get("selector")}

    return [
        SemanticAction(
            name=g["name"],
            description=g.get("description", g["name"]),
            steps=[
                InteractionStep(
                    type=s.get("type", "click"),
                    selector=s.get("selector", ""),
                    value=s.get("value", ""),
                )
                for s in g.get("steps", [])
                if s.get("selector")
            ],
            expected_outcome=g.get("expected_outcome", ""),
        )
        for g in groups
        if g.get("name") and g.get("steps")
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ActionIdentifier:
    """
    Identifies complete user workflows available on the current page.

    Parameters
    ----------
    llm_client   Optional LLMOracle. Without it, falls back to DOM grouping.

    Usage
    -----
        identifier = ActionIdentifier(llm_client=llm_oracle)
        actions = await identifier.identify(elements, a11y_tree, screenshot)
    """

    def __init__(self, llm_client: "LLMOracle | None" = None):
        self._llm = llm_client

    async def identify(
        self,
        elements: list[dict],
        a11y_tree: dict,
        screenshot: bytes,
        explored_context: str = "",
    ) -> list[SemanticAction]:
        if not elements:
            return []

        if self._llm is not None:
            result = await llm_group(elements, a11y_tree, screenshot, self._llm, explored_context)
            if result is not None:
                _log.info(
                    "AI:  LLM → %d workflow(s): %s",
                    len(result),
                    ", ".join(a.name for a in result),
                )
                covered = {
                    s.selector
                    for action in result
                    for s in action.steps
                    if s.selector
                }
                uncovered = [e for e in elements if e.get("selector") not in covered]
                extras = dom_group(uncovered)
                if extras:
                    _log.info(
                        "AI:  DOM complement → %d extra(s): %s",
                        len(extras),
                        ", ".join(e.name for e in extras),
                    )
                return _collapse_repetitive(result + extras)

        return _collapse_repetitive(dom_group(elements))
