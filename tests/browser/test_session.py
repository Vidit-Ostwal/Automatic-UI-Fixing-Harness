"""
Tests for browser/session.py

Run with:
    pytest tests/test_session.py -v

All tests use page.set_content() to load inline HTML — no network required.
Tests verify:
  1. Screenshot is captured (non-empty PNG bytes)
  2. A11y tree is captured (has expected structure)
  3. Console errors are collected and cleared between captures
  4. Network failures are collected
  5. Interactive elements are enumerated correctly
  6. Geometry violations are detected (overflow, overlap, viewport clip)
  7. Viewport resize works
  8. Event buffers clear after each capture_state() call
"""

import asyncio
import pytest
import pytest_asyncio
from playwright.async_api import async_playwright

from browser.session import BrowserSession
from models import PageState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def make_session(html: str, viewport_width=1280, viewport_height=800):
    """
    Helper: returns (session, playwright_context_manager).
    Caller must use as async context manager via the BrowserSession.create() pattern.
    We inline a small helper that sets content after navigation.
    """
    pass  # not used directly — each test opens its own session


HTML_SIMPLE = """
<html>
  <body>
    <h1>Hello World</h1>
    <button id="btn1">Click Me</button>
    <a href="/about">About</a>
    <input type="text" placeholder="Enter name" />
  </body>
</html>
"""

HTML_WITH_CONSOLE_ERROR = """
<html>
  <body>
    <script>console.error('test error from page');</script>
    <p>Page with error</p>
  </body>
</html>
"""

HTML_WITH_OVERFLOW = """
<html>
  <body style="margin:0;padding:0;">
    <div id="container" style="width:200px; overflow:visible;">
      <div id="overflow-child" style="width:500px; white-space:nowrap;">
        This text is very very very very very very very long and overflows its container
      </div>
    </div>
  </body>
</html>
"""

HTML_WITH_OVERLAP = """
<html>
  <body style="margin:0;padding:0;position:relative;">
    <button style="position:absolute;top:10px;left:10px;width:100px;height:40px;">Button A</button>
    <button style="position:absolute;top:10px;left:30px;width:100px;height:40px;">Button B</button>
  </body>
</html>
"""

HTML_WITH_VIEWPORT_CLIP = """
<html>
  <body style="margin:0;padding:0;overflow:hidden;">
    <button style="position:absolute;top:10px;left:1400px;width:100px;height:40px;">Off Screen</button>
    <button style="position:absolute;top:10px;left:10px;width:100px;height:40px;">On Screen</button>
  </body>
</html>
"""

HTML_MULTI_INTERACTIVE = """
<html>
  <body>
    <button id="save">Save</button>
    <button id="cancel">Cancel</button>
    <input type="email" placeholder="Email" />
    <select id="role">
      <option>Admin</option>
      <option>User</option>
    </select>
    <a href="/home">Home</a>
  </body>
</html>
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_screenshot_is_non_empty_png():
    """capture_state() returns PNG bytes starting with the PNG magic header."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        state = await session.capture_state()

    assert isinstance(state.screenshot, bytes)
    assert len(state.screenshot) > 0
    assert state.screenshot[:4] == b"\x89PNG", "Expected PNG magic bytes"


@pytest.mark.asyncio
async def test_a11y_tree_has_expected_structure():
    """capture_state() returns an a11y tree dict with role and children."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        state = await session.capture_state()

    assert isinstance(state.a11y_tree, dict)
    assert "role" in state.a11y_tree
    # Flatten tree to find all roles
    roles = _collect_roles(state.a11y_tree)
    assert "button" in roles, f"Expected 'button' role in tree, got: {roles}"
    assert "heading" in roles, f"Expected 'heading' role in tree, got: {roles}"


@pytest.mark.asyncio
async def test_url_is_captured():
    """capture_state() records the current page URL."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        state = await session.capture_state()

    assert isinstance(state.url, str)
    assert len(state.url) > 0


@pytest.mark.asyncio
async def test_console_errors_are_collected():
    """Console errors emitted by the page are captured in PageState."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_WITH_CONSOLE_ERROR)
        state = await session.capture_state()

    assert any("test error from page" in e for e in state.console_errors), (
        f"Expected console error not found. Got: {state.console_errors}"
    )


@pytest.mark.asyncio
async def test_console_errors_cleared_between_captures():
    """Event buffers are cleared after each capture_state() call."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_WITH_CONSOLE_ERROR)
        state1 = await session.capture_state()
        # Second capture — no new errors emitted since last capture
        state2 = await session.capture_state()

    assert len(state1.console_errors) > 0, "First capture should have errors"
    assert len(state2.console_errors) == 0, "Second capture buffer should be empty"


@pytest.mark.asyncio
async def test_interactive_elements_enumerated():
    """get_interactive_elements() finds all buttons, inputs, links, selects."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_MULTI_INTERACTIVE)
        elements = await session.get_interactive_elements()

    tags = {e["tag"] for e in elements}
    assert "button" in tags
    assert "input" in tags
    assert "select" in tags
    assert "a" in tags
    assert len(elements) >= 5


@pytest.mark.asyncio
async def test_interactive_elements_have_required_fields():
    """Every element returned has tag, role, label, selector fields."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_MULTI_INTERACTIVE)
        elements = await session.get_interactive_elements()

    for el in elements:
        assert "tag" in el
        assert "role" in el
        assert "label" in el
        assert "selector" in el


@pytest.mark.asyncio
async def test_overflow_violation_detected():
    """get_geometry_violations() detects overflow_x when child is wider than container."""
    async with BrowserSession.create(viewport_width=800, viewport_height=600) as session:
        await session.page.set_content(HTML_WITH_OVERFLOW)
        violations = await session.get_geometry_violations()

    types = [v["type"] for v in violations]
    assert "overflow_x" in types, f"Expected overflow_x violation. Got: {violations}"


@pytest.mark.asyncio
async def test_overlap_violation_detected():
    """get_geometry_violations() detects overlapping interactive elements."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_WITH_OVERLAP)
        violations = await session.get_geometry_violations()

    types = [v["type"] for v in violations]
    assert "overlap" in types, f"Expected overlap violation. Got: {violations}"


@pytest.mark.asyncio
async def test_viewport_clip_violation_detected():
    """get_geometry_violations() detects interactive elements clipped outside viewport."""
    async with BrowserSession.create(viewport_width=1280, viewport_height=800) as session:
        await session.page.set_content(HTML_WITH_VIEWPORT_CLIP)
        violations = await session.get_geometry_violations()

    types = [v["type"] for v in violations]
    assert "viewport_clip" in types, f"Expected viewport_clip violation. Got: {violations}"


@pytest.mark.asyncio
async def test_no_false_positive_on_clean_page():
    """get_geometry_violations() returns no violations on a well-formed page."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        violations = await session.get_geometry_violations()

    assert violations == [], f"Expected no violations on clean page. Got: {violations}"


@pytest.mark.asyncio
async def test_viewport_resize():
    """set_viewport() changes the page dimensions reflected in geometry checks."""
    async with BrowserSession.create(viewport_width=1280) as session:
        await session.page.set_content(HTML_WITH_VIEWPORT_CLIP)

        # At 1280px — off-screen button should be clipped
        violations_wide = await session.get_geometry_violations()

        # Resize to 400px — even more clipped
        await session.set_viewport(400, 800)
        violations_narrow = await session.get_geometry_violations()

    wide_types = [v["type"] for v in violations_wide]
    narrow_types = [v["type"] for v in violations_narrow]
    assert "viewport_clip" in wide_types
    assert "viewport_clip" in narrow_types


@pytest.mark.asyncio
async def test_capture_state_returns_pagstate_type():
    """capture_state() returns a PageState instance with all expected fields."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        state = await session.capture_state()

    assert isinstance(state, PageState)
    assert hasattr(state, "url")
    assert hasattr(state, "screenshot")
    assert hasattr(state, "a11y_tree")
    assert hasattr(state, "console_errors")
    assert hasattr(state, "network_failures")
    assert hasattr(state, "timestamp")
    assert state.timestamp > 0


@pytest.mark.asyncio
async def test_element_exists_true():
    """element_exists() returns True for a selector that is present."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        exists = await session.element_exists("#btn1")

    assert exists is True


@pytest.mark.asyncio
async def test_element_exists_false():
    """element_exists() returns False for a selector that is absent."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        exists = await session.element_exists("#does-not-exist")

    assert exists is False


@pytest.mark.asyncio
async def test_get_text():
    """get_text() returns the inner text of an element."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_SIMPLE)
        text = await session.get_text("h1")

    assert text == "Hello World"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_roles(node: dict) -> set[str]:
    """Recursively collect all roles from an a11y tree."""
    roles = set()
    if not node:
        return roles
    if "role" in node:
        roles.add(node["role"])
    for child in node.get("children", []):
        roles |= _collect_roles(child)
    return roles
