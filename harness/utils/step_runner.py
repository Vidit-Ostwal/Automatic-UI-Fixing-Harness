"""
Shared low-level browser step execution.

Used by both planner (BFS explorer) and executor (GoalExecutor) so workflow
replay and goal-driven execution share identical fill/click/nav behaviour.
"""

import asyncio
import logging
import time

from harness.utils.fill_retry import suggest_fill_value
from harness.utils.url import normalise_url

logger = logging.getLogger(__name__)

# Post-action settle tuning — wait for DOM + animations before screenshots.
SETTLE_TIMEOUT_MS = 4_000
SETTLE_STABLE_MS = 500
SETTLE_POLL_MS = 100
SETTLE_MIN_MS = 250


async def dismiss_overlays(session) -> None:
    """Press Escape to close any open Radix dropdown/dialog before acting."""
    try:
        await session.page.keyboard.press("Escape")
        await asyncio.sleep(0.15)
    except Exception:
        pass


async def _dom_signature(session) -> str:
    """Cheap fingerprint of visible DOM — changes when SPA re-renders."""
    return await session.page.evaluate(
        """() => {
            const body = document.body;
            if (!body) return '0|0';
            return body.innerText.length + '|' + document.querySelectorAll('*').length;
        }"""
    )


async def _wait_animation_frames(session, frames: int = 2) -> None:
    """Let CSS transitions / React paint cycles finish."""
    await session.page.evaluate(
        f"""() => new Promise(resolve => {{
            let n = 0;
            const tick = () => {{
                if (++n >= {frames}) resolve();
                else requestAnimationFrame(tick);
            }};
            requestAnimationFrame(tick);
        }})"""
    )


async def wait_for_settle(
    session,
    prev_url: str | None = None,
    timeout_ms: int = SETTLE_TIMEOUT_MS,
) -> None:
    """
    Wait for the UI to settle after an action, before capturing screenshots.

    Handles both URL-changing navigations and same-URL SPA updates (calendar
    flips, modals, list refreshes) by polling until the DOM stops changing.
    """
    await asyncio.sleep(SETTLE_MIN_MS / 1000)

    if prev_url is not None:
        nav_deadline = time.monotonic() + timeout_ms / 1000
        prev_norm = normalise_url(prev_url)
        while time.monotonic() < nav_deadline:
            curr_norm = normalise_url(session.page.url)
            if curr_norm != prev_norm:
                logger.debug(
                    "step_runner: nav settled %s → %s", prev_norm, curr_norm
                )
                await session._wait_stable()
                break
            await asyncio.sleep(SETTLE_POLL_MS / 1000)

    deadline = time.monotonic() + timeout_ms / 1000
    last_sig: str | None = None
    stable_since: float | None = None
    poll_s = SETTLE_POLL_MS / 1000
    stable_s = SETTLE_STABLE_MS / 1000

    while time.monotonic() < deadline:
        try:
            sig = await _dom_signature(session)
        except Exception:
            break

        now = time.monotonic()
        if sig == last_sig:
            if stable_since is None:
                stable_since = now
            elif now - stable_since >= stable_s:
                logger.debug("step_runner: DOM stable for %.0fms", SETTLE_STABLE_MS)
                break
        else:
            last_sig = sig
            stable_since = None

        await asyncio.sleep(poll_s)

    try:
        await _wait_animation_frames(session)
    except Exception:
        pass

    await session._wait_stable()


async def wait_for_navigation(session, prev_url: str, timeout_ms: int = SETTLE_TIMEOUT_MS) -> None:
    """Backward-compatible alias — waits for navigation *and* UI settle."""
    await wait_for_settle(session, prev_url=prev_url, timeout_ms=timeout_ms)


async def execute_steps(
    session,
    steps: list[dict],
    llm_oracle=None,
) -> tuple[bool, str, bool]:
    """
    Execute a sequence of {type, selector, value} step dicts.

    Returns (success, reason, had_fill_issues).
    """
    if not steps:
        return False, "no steps to execute", False

    had_fill_issues = False

    for step in steps:
        step_type = step.get("type", "click")
        selector = step.get("selector", "")
        value = step.get("value", "")

        try:
            if step_type == "fill":
                await session.fill(selector, value)
                try:
                    actual = await session.page.input_value(selector, timeout=2000)
                except Exception:
                    actual = value

                if actual != value:
                    had_fill_issues = True
                    logger.info(
                        "step_runner: fill(%r) rejected %r → got %r, asking LLM",
                        selector, value, actual,
                    )
                    alt = None
                    if llm_oracle is not None:
                        screenshot = await session.page.screenshot()
                        alt = await suggest_fill_value(
                            llm_oracle, selector, value, actual, screenshot
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
