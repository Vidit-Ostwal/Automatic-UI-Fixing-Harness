"""
Diff oracle — detects unexpected state changes between two PageState snapshots.

The diff oracle fires after every action in the executor loop:
  before = capture_state()
  execute(action)
  after  = capture_state()
  finding = await diff_oracle.check(before, after, action)

It computes a structural diff of the a11y trees and optionally asks the
LLM oracle to judge whether the visual change matches what was expected.

A finding is raised when:
  - Nothing changed at all after an action that should cause a change.
  - The URL changed unexpectedly (navigation when none was expected).
  - The page lost significant structure (element count dropped dramatically).
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from harness.models import BugType, DetectedBy, Finding, PageState, Severity


# ---------------------------------------------------------------------------
# Structural diff
# ---------------------------------------------------------------------------

@dataclass
class StructuralDiff:
    url_changed: bool
    url_before: str
    url_after: str
    roles_added: list[str]
    roles_removed: list[str]
    state_changes: list[str]           # human-readable descriptions
    element_count_before: int
    element_count_after: int

    @property
    def element_delta(self) -> int:
        return self.element_count_after - self.element_count_before

    @property
    def anything_changed(self) -> bool:
        return (
            self.url_changed
            or bool(self.roles_added)
            or bool(self.roles_removed)
            or bool(self.state_changes)
            or self.element_delta != 0
        )


def _collect_nodes(node: dict) -> list[dict]:
    """Flatten a11y tree into a list of all nodes."""
    if not node:
        return []
    nodes = [node]
    for child in node.get("children", []):
        nodes.extend(_collect_nodes(child))
    return nodes


def _role_multiset(state: PageState) -> dict[str, int]:
    """Count of each role present in the a11y tree."""
    counts: dict[str, int] = {}
    for n in _collect_nodes(state.a11y_tree):
        role = n.get("role", "")
        if role:
            counts[role] = counts.get(role, 0) + 1
    return counts


def compute_diff(before: PageState, after: PageState) -> StructuralDiff:
    """Return a StructuralDiff describing what changed between two states."""
    before_roles = _role_multiset(before)
    after_roles  = _role_multiset(after)

    all_roles = set(before_roles) | set(after_roles)
    roles_added:   list[str] = []
    roles_removed: list[str] = []

    for role in all_roles:
        b_count = before_roles.get(role, 0)
        a_count = after_roles.get(role, 0)
        if a_count > b_count:
            roles_added.append(f"{role} (+{a_count - b_count})")
        elif a_count < b_count:
            roles_removed.append(f"{role} (-{b_count - a_count})")

    # Detect state attribute changes (expanded, checked, selected) on any node.
    before_nodes = {
        f"{n.get('role')}::{n.get('name', '')}": n
        for n in _collect_nodes(before.a11y_tree)
    }
    after_nodes = {
        f"{n.get('role')}::{n.get('name', '')}": n
        for n in _collect_nodes(after.a11y_tree)
    }
    state_changes: list[str] = []
    for key in set(before_nodes) & set(after_nodes):
        b = before_nodes[key]
        a = after_nodes[key]
        for attr in ("expanded", "checked", "selected", "disabled"):
            if b.get(attr) != a.get(attr):
                state_changes.append(
                    f"{key}: {attr} {b.get(attr)} → {a.get(attr)}"
                )

    before_count = len(_collect_nodes(before.a11y_tree))
    after_count  = len(_collect_nodes(after.a11y_tree))

    return StructuralDiff(
        url_changed=before.url != after.url,
        url_before=before.url,
        url_after=after.url,
        roles_added=roles_added,
        roles_removed=roles_removed,
        state_changes=state_changes,
        element_count_before=before_count,
        element_count_after=after_count,
    )


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

class DiffOracle:
    """
    Checks whether a before→after state transition is valid for a given action.

    Parameters
    ----------
    llm_oracle   Optional LLMOracle for screenshot-level judgment.
                 If provided, is consulted when the structural diff is ambiguous.
    """

    # If element count drops by more than this fraction, treat as potential crash.
    CATASTROPHIC_LOSS_THRESHOLD = 0.5

    def __init__(self, llm_oracle=None):
        self._llm = llm_oracle

    async def check(
        self,
        before: PageState,
        after: PageState,
        action: str,
        expect_change: bool = True,
    ) -> Optional[Finding]:
        """
        Compute the structural diff and raise a Finding when:
          - expect_change=True but nothing changed at all.
          - Page lost > 50% of its elements (probable crash/blank page).
          - LLM oracle (if present) says the diff looks like a bug.

        Parameters
        ----------
        before, after    PageState snapshots around the action.
        action           Human-readable description of the action performed.
        expect_change    Set False for read-only actions (navigation, search)
                         where a no-change result is normal.
        """
        diff = compute_diff(before, after)

        # 1. Catastrophic element loss.
        if (
            diff.element_count_before > 5
            and diff.element_count_after
            < diff.element_count_before * (1 - self.CATASTROPHIC_LOSS_THRESHOLD)
        ):
            return Finding(
                id=f"DIFF-{uuid.uuid4().hex[:6]}",
                title=f"Page lost most content after '{action}'",
                bug_type=BugType.LOGIC,
                severity=Severity.CRITICAL,
                steps=[action],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"Element count dropped from {diff.element_count_before} "
                    f"to {diff.element_count_after} after '{action}'. "
                    "Page may have crashed or navigated to an error screen."
                ),
                screenshot_before=before.screenshot,
                screenshot_after=after.screenshot,
            )

        # 2. Expected change but nothing changed.
        if expect_change and not diff.anything_changed:
            # Defer to LLM before filing — it might be a cosmetic-only change.
            if self._llm:
                verdict = await self._llm.judge_diff(before, after, action)
                if verdict.verdict == "ok":
                    return None
                if verdict.verdict == "noise":
                    return None
                if verdict.verdict == "bug":
                    return Finding(
                        id=f"DIFF-LLM-{uuid.uuid4().hex[:6]}",
                        title=verdict.description or f"No UI change after '{action}'",
                        bug_type=BugType.LOGIC,
                        severity=Severity(verdict.severity or "high"),
                        steps=[action],
                        detected_by=DetectedBy.LLM,
                        reasoning=verdict.reasoning,
                        screenshot_before=before.screenshot,
                        screenshot_after=after.screenshot,
                    )

            return Finding(
                id=f"DIFF-{uuid.uuid4().hex[:6]}",
                title=f"No UI change after '{action}'",
                bug_type=BugType.LOGIC,
                severity=Severity.HIGH,
                steps=[action],
                detected_by=DetectedBy.HEURISTIC,
                reasoning=(
                    f"After performing '{action}', the page structure did not change. "
                    "URL, element count, roles, and state attributes are identical."
                ),
                screenshot_before=before.screenshot,
                screenshot_after=after.screenshot,
            )

        # 3. LLM visual judgment on significant structural changes.
        if self._llm and diff.anything_changed:
            verdict = await self._llm.judge_diff(before, after, action)
            if verdict.verdict == "bug":
                return Finding(
                    id=f"DIFF-LLM-{uuid.uuid4().hex[:6]}",
                    title=verdict.description,
                    bug_type=BugType.LOGIC,
                    severity=Severity(verdict.severity or "medium"),
                    steps=[action],
                    detected_by=DetectedBy.LLM,
                    reasoning=verdict.reasoning,
                    screenshot_before=before.screenshot,
                    screenshot_after=after.screenshot,
                )

        return None
