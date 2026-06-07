"""
Auth workflow — signs up the first (admin/host) account on a fresh Memos instance.

Memos starts with an empty database. The very first POST to /api/v1/auth/signup
creates the host account. The workflow drives this through the real UI.

Since we treat the app as a black box, we locate the form fields by type and
placeholder heuristics rather than hardcoded selectors — this keeps the workflow
adaptable to minor UI changes.
"""

import logging
from dataclasses import dataclass

from browser.session import BrowserSession

logger = logging.getLogger(__name__)

# Default credentials used by the harness for the seed account.
DEFAULT_USERNAME = "harness_admin"
DEFAULT_PASSWORD = "Harness@2024!"

# Candidate selectors tried in order for each field.
_USERNAME_SELECTORS = [
    'input[placeholder*="username" i]',
    'input[placeholder*="Username" i]',
    'input[name="username"]',
    'input[type="text"]:first-of-type',
]

_PASSWORD_SELECTORS = [
    'input[type="password"]:first-of-type',
    'input[placeholder*="password" i]',
    'input[name="password"]',
]

_CONFIRM_SELECTORS = [
    'input[type="password"]:nth-of-type(2)',
    'input[placeholder*="confirm" i]',
    'input[placeholder*="repeat" i]',
    'input[name="confirmPassword"]',
    'input[name="passwordConfirm"]',
]

_SUBMIT_SELECTORS = [
    'button[type="submit"]',
    'button:has-text("Sign up")',
    'button:has-text("Register")',
    'button:has-text("Create")',
    'button:has-text("Get started")',
]


@dataclass
class AuthResult:
    success: bool
    username: str
    url_after: str
    error: str = ""


async def _find_and_fill(session: BrowserSession, selectors: list[str], value: str) -> bool:
    """Try each selector in order; fill the first one found. Returns True on success."""
    for sel in selectors:
        if await session.element_exists(sel):
            await session.fill(sel, value)
            return True
    return False


async def signup(
    session: BrowserSession,
    app_url: str,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD,
) -> AuthResult:
    """
    Drive the sign-up form to create the host account.

    Returns an AuthResult indicating success or the reason for failure.
    The session is left on whatever page the app navigates to after signup.
    """
    await session.navigate(app_url)

    # If already signed in (redirected away from /auth), return success.
    if "/auth" not in session.page.url and "signup" not in session.page.url.lower():
        logger.info("Auth: already authenticated at %s", session.page.url)
        return AuthResult(success=True, username=username, url_after=session.page.url)

    logger.info("Auth: filling sign-up form at %s", session.page.url)

    # Fill username.
    if not await _find_and_fill(session, _USERNAME_SELECTORS, username):
        return AuthResult(
            success=False, username=username,
            url_after=session.page.url,
            error="Could not find username input field",
        )

    # Fill password.
    if not await _find_and_fill(session, _PASSWORD_SELECTORS, password):
        return AuthResult(
            success=False, username=username,
            url_after=session.page.url,
            error="Could not find password input field",
        )

    # Fill confirm-password (optional — not all apps have it).
    await _find_and_fill(session, _CONFIRM_SELECTORS, password)

    # Submit.
    submitted = False
    for sel in _SUBMIT_SELECTORS:
        if await session.element_exists(sel):
            await session.click(sel)
            submitted = True
            break

    if not submitted:
        # Last resort: press Enter on the password field.
        for sel in _PASSWORD_SELECTORS:
            if await session.element_exists(sel):
                await session.press(sel, "Enter")
                submitted = True
                break

    if not submitted:
        return AuthResult(
            success=False, username=username,
            url_after=session.page.url,
            error="Could not find submit button",
        )

    url_after = session.page.url
    # Success: redirected away from the auth/signup page.
    success = "/auth" not in url_after and "signup" not in url_after.lower()

    if success:
        logger.info("Auth: signup succeeded, now at %s", url_after)
    else:
        logger.warning("Auth: signup may have failed, still at %s", url_after)

    return AuthResult(success=success, username=username, url_after=url_after)
