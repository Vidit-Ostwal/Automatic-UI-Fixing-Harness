"""
step_runner.py — shared low-level step execution helpers.

Extracted from BFS explorer so GoalExecutor uses the exact same execution
model: fill verification with LLM fallback, overlay dismissal, and
post-action navigation wait.

Public API
----------
execute_steps(session, steps, llm_oracle)
    Execute a list of {type, selector, value} dicts in sequence.
    Returns (success, reason, had_fill_issues).
    Mirrors BFS explorer's _execute_action() exactly.

wait_for_navigation(session, prev_url, timeout_ms=2000)
    Poll for a SPA URL change after an action, then wait for stability.
    Mirrors BFS explorer's _wait_for_navigation() exactly.

dismiss_overlays(session)
    Press Escape to close any open Radix dropdown or dialog.
    Mirrors BFS explorer's _dismiss_overlays() exactly.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def dismiss_overlays(session) -> None:
    """Press Escape to close any open Radix dropdown/dialog before acting."""
    try:
        await session.page.keyboard.press("Escape")
        await asyncio.sleep(0.15)
    except Exception:
        pass


async def wait_for_navigation(session, prev_url: str, timeout_ms: int = 2000) -> None:
    """
    After an action, poll for a URL change signalling client-side navigation
    (React Router pushState).  If the URL changes, wait for the new page to
    stabilise before returning.
    """
    from planner.state_hasher import normalise_url

    deadline = timeout_ms / 1000
    slept    = 0.0
    interval = 0.1
    prev_norm = normalise_url(prev_url)

    while slept < deadline:
        await asyncio.sleep(interval)
        slept += interval
        curr_norm = normalise_url(session.page.url)
        if curr_norm != prev_norm:
            logger.debug(
                "step_runner: nav settled %s → %s (%.1fs)", prev_norm, curr_norm, slept
            )
            await session._wait_stable()
            return


async def execute_steps(
    session,
    steps: list[dict],
    llm_oracle=None,
) -> tuple[bool, str, bool]:
    """
    Execute a sequence of {type, selector, value} step dicts.

    Identical behaviour to BFS explorer's _execute_action():
      - fill: writes the value then reads back input_value() to verify
              acceptance. If the field rejected the value, asks the LLM
              for an alternative (suggest_fill_value) and retries once.
      - click / press / select: executed directly.

    Returns
    -------
    (success, reason, had_fill_issues)
      success          True when all steps completed without error.
      reason           Empty string on success; error description on failure.
      had_fill_issues  True when at least one fill value was initially rejected
                       and an LLM alternative was needed.
    """
    if not steps:
        return False, "no steps to execute", False

    had_fill_issues = False

    for step in steps:
        step_type = step.get("type", "click")
        selector  = step.get("selector", "")
        value     = step.get("value", "")

        try:
            if step_type == "fill":
                await session.fill(selector, value)
                try:
                    actual = await session.page.input_value(selector, timeout=2000)
                except Exception:
                    actual = value  # can't verify — assume accepted

                if actual != value:
                    had_fill_issues = True
                    logger.info(
                        "step_runner: fill(%r) rejected %r → got %r, asking LLM",
                        selector, value, actual,
                    )
                    alt = None
                    if llm_oracle is not None:
                        screenshot = await session.page.screenshot()
                        alt = await llm_oracle.suggest_fill_value(
                            selector, value, actual, screenshot
                        )
                    if alt is not None:
                        logger.info("step_runner: fill retry with LLM value %r", alt)
                        await session.fill(selector, alt)
                        try:
                            actual2 = await session.page.input_value(selector, timeout=2000)
                        except Exception:
                            actual2 = alt
                        if actual2 != alt:
                            return (
                                False,
                                f"fill({selector!r}) LLM retry also rejected: "
                                f"wrote {alt!r}, got {actual2!r}",
                                had_fill_issues,
                            )
                    else:
                        return (
                            False,
                            f"fill({selector!r}) value not accepted: "
                            f"wrote {value!r}, got {actual!r}",
                            had_fill_issues,
                        )

            elif step_type == "click":
                await session.click(selector)

            elif step_type == "press":
                await session.press(selector, value)

            elif step_type == "select":
                await session.fill(selector, value)

            else:
                return False, f"unknown step type '{step_type}'", had_fill_issues

        except Exception as exc:
            return (
                False,
                f"{step_type}({selector!r}) raised {type(exc).__name__}: {exc}",
                had_fill_issues,
            )

    return True, "", had_fill_issues
