"""
Logic oracle — deterministic crash detection during BFS exploration.

Checks that the page has no uncaught JS errors and that the accessibility
tree is non-empty after each action.
"""

import uuid
from typing import Optional

from harness.models import BugType, DetectedBy, Finding, PageState, Severity


class LogicOracle:
    """Deterministic page-health checks used by the BFS planner."""

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
