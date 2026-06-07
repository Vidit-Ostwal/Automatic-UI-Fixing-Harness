"""
Logic oracle — deterministic CRUD invariant checks.

Each check takes explicit before/after context (what was done, what text was
involved) and verifies the invariant holds in the current DOM.  None of these
checks need LLM — they are pure string / structure comparisons.

Invariants covered
------------------
text_appeared      After creating/editing, typed text should be visible.
text_disappeared   After deleting/archiving, text should no longer be visible.
state_toggled      After a toggle action, the trigger element's state should change.
count_changed      After adding/removing, a list's item count should change.
no_crash           Page has no JS errors and is not blank after an action.
"""

import uuid
from typing import Optional

from models import BugType, DetectedBy, Finding, PageState, Severity


def _page_text(state: PageState) -> str:
    """Flatten all text values from the a11y tree into one searchable string."""
    parts: list[str] = []

    def _walk(node: dict) -> None:
        if not node:
            return
        if name := node.get("name"):
            parts.append(name)
        for child in node.get("children", []):
            _walk(child)

    _walk(state.a11y_tree)
    return " ".join(parts)


def _count_role(state: PageState, role: str) -> int:
    """Count how many nodes with a given role exist in the a11y tree."""
    count = 0

    def _walk(node: dict) -> None:
        nonlocal count
        if not node:
            return
        if node.get("role") == role:
            count += 1
        for child in node.get("children", []):
            _walk(child)

    _walk(state.a11y_tree)
    return count


def _find_by_role_and_name(
    state: PageState, role: str, name_fragment: str
) -> Optional[dict]:
    """Return the first node matching role + partial name (case-insensitive)."""
    needle = name_fragment.lower()

    def _walk(node: dict) -> Optional[dict]:
        if not node:
            return None
        if node.get("role") == role and needle in node.get("name", "").lower():
            return node
        for child in node.get("children", []):
            result = _walk(child)
            if result:
                return result
        return None

    return _walk(state.a11y_tree)


class LogicOracle:
    """
    Deterministic CRUD invariant checks.

    All methods return a Finding when the invariant is violated, or None
    when everything looks correct.
    """

    def check_text_appeared(
        self,
        after: PageState,
        expected_text: str,
        action_description: str = "",
    ) -> Optional[Finding]:
        """
        Verify that expected_text is visible somewhere on the page after an action.
        Used after: create memo, edit memo, submit form.
        """
        page_text = _page_text(after)
        if expected_text.lower() not in page_text.lower():
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"Created content not visible: '{expected_text[:50]}'",
                bug_type=BugType.LOGIC,
                severity=Severity.HIGH,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"After '{action_description}', expected text "
                    f"'{expected_text}' was not found in the page content."
                ),
                screenshot_after=after.screenshot,
            )
        return None

    def check_text_disappeared(
        self,
        after: PageState,
        removed_text: str,
        action_description: str = "",
    ) -> Optional[Finding]:
        """
        Verify that removed_text is no longer visible after a delete/archive action.
        """
        page_text = _page_text(after)
        if removed_text.lower() in page_text.lower():
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"Deleted/archived content still visible: '{removed_text[:50]}'",
                bug_type=BugType.LOGIC,
                severity=Severity.HIGH,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"After '{action_description}', text '{removed_text}' "
                    f"is still present in the page."
                ),
                screenshot_after=after.screenshot,
            )
        return None

    def check_state_toggled(
        self,
        before: PageState,
        after: PageState,
        role: str,
        name_fragment: str,
        action_description: str = "",
    ) -> Optional[Finding]:
        """
        Verify that a toggle element (button/checkbox) changed its state attribute
        (aria-expanded, aria-checked, aria-pressed, or label text) after the action.
        Used after: pin/unpin, archive, expand/collapse.
        """
        before_node = _find_by_role_and_name(before, role, name_fragment)
        after_node  = _find_by_role_and_name(after,  role, name_fragment)

        # If the element disappeared entirely, the state definitely changed.
        if before_node and not after_node:
            return None

        # If the element didn't exist before, nothing to compare.
        if not before_node:
            return None

        state_keys = ("expanded", "checked", "pressed", "selected")
        before_attrs = {k: before_node.get(k) for k in state_keys}
        after_attrs  = {k: after_node.get(k)  for k in state_keys} if after_node else {}

        # Label change also counts as a state change (e.g. "Pin" → "Unpin").
        before_name = (before_node.get("name") or "").lower()
        after_name  = (after_node.get("name")  or "").lower() if after_node else ""
        label_changed = before_name != after_name

        attrs_changed = before_attrs != after_attrs

        if not attrs_changed and not label_changed:
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"Toggle had no effect: {role} '{name_fragment}'",
                bug_type=BugType.LOGIC,
                severity=Severity.HIGH,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"After '{action_description}', the {role} element matching "
                    f"'{name_fragment}' shows no change in state or label. "
                    f"Before: {before_attrs} / '{before_name}'. "
                    f"After:  {after_attrs} / '{after_name}'."
                ),
                screenshot_before=before.screenshot,
                screenshot_after=after.screenshot,
            )
        return None

    def check_count_changed(
        self,
        before: PageState,
        after: PageState,
        role: str,
        expected_delta: int,
        action_description: str = "",
    ) -> Optional[Finding]:
        """
        Verify that the count of elements with a given role changed by expected_delta.
        e.g. after creating a memo, listitem count should increase by 1 (+1).
        e.g. after deleting a memo, listitem count should decrease by 1 (-1).
        """
        before_count = _count_role(before, role)
        after_count  = _count_role(after,  role)
        actual_delta = after_count - before_count

        if actual_delta != expected_delta:
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"Item count wrong after '{action_description}'",
                bug_type=BugType.LOGIC,
                severity=Severity.HIGH,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"Expected {role} count to change by {expected_delta:+d} "
                    f"(from {before_count} to {before_count + expected_delta}). "
                    f"Actual: {before_count} → {after_count} (delta={actual_delta:+d})."
                ),
                screenshot_before=before.screenshot,
                screenshot_after=after.screenshot,
            )
        return None

    def check_no_crash(
        self,
        after: PageState,
        action_description: str = "",
    ) -> Optional[Finding]:
        """
        Verify the page didn't crash: no JS errors AND a11y tree is non-empty.
        """
        critical_errors = [
            e for e in after.console_errors
            if "[pageerror]" in e or "uncaught" in e.lower()
        ]

        tree_empty = not after.a11y_tree or not after.a11y_tree.get("children")

        if critical_errors:
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"JavaScript crash after '{action_description}'",
                bug_type=BugType.LOGIC,
                severity=Severity.CRITICAL,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=f"Uncaught JS errors: {'; '.join(critical_errors[:3])}",
                console_errors=critical_errors,
                screenshot_after=after.screenshot,
            )

        if tree_empty and not critical_errors:
            return Finding(
                id=f"LOG-{uuid.uuid4().hex[:6]}",
                title=f"Page appears blank after '{action_description}'",
                bug_type=BugType.LOGIC,
                severity=Severity.CRITICAL,
                steps=[action_description] if action_description else [],
                detected_by=DetectedBy.HEURISTIC,
                reasoning="Accessibility tree is empty — page may have crashed or failed to render.",
                screenshot_after=after.screenshot,
            )

        return None
