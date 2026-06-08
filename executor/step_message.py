"""
StepMessage — the unit of data published to the async step queue after every
executor action.

The verifier agent reads from this queue and inspects before/after screenshots
and accessibility trees to detect logical or UI bugs.

Queue contract
--------------
- Producers  : GoalExecutor instances (one message per executed action)
- Consumer   : VerifierAgent (to be wired in; currently a drain loop logs them)
- Transport  : asyncio.Queue[StepMessage] — created once, shared across all
               GoalExecutor instances in a run

Sentinel
--------
Put a StepMessage with action_name == QUEUE_DONE_SENTINEL to signal that a
producer has finished.  The consumer can use this to know when all executors
are done.
"""

from dataclasses import dataclass
from typing import Optional

QUEUE_DONE_SENTINEL = "__EXECUTOR_DONE__"


@dataclass
class StepMessage:
    """
    Everything the verifier needs about one completed executor action.

    Fields
    ------
    test_case_id        Trajectory id this run belongs to (e.g. "T-001").
    run_id              Unique hex id for this executor run instance.
    step_index          0-based position within the trajectory.
    action_name         Semantic name of the action (e.g. "create_new_memo").
    action_description  Human-readable description of the action.
    screenshot_before   PNG bytes captured immediately before the action.
    screenshot_after    PNG bytes captured immediately after the action.
    url_before          Page URL before the action.
    url_after           Page URL after the action.
    a11y_before         Accessibility tree dict before the action.
    a11y_after          Accessibility tree dict after the action.
    success             Whether the action steps completed without exception.
    error               Exception message if success is False, else None.
    screenshot_before_path  Path to the saved PNG (relative to output_dir).
    screenshot_after_path   Path to the saved PNG (relative to output_dir).
    """

    test_case_id: str
    run_id: str
    step_index: int
    action_name: str
    action_description: str
    screenshot_before: bytes
    screenshot_after: bytes
    url_before: str
    url_after: str
    a11y_before: dict
    a11y_after: dict
    success: bool
    error: Optional[str] = None
    screenshot_before_path: str = ""
    screenshot_after_path: str = ""
