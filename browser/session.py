"""
Browser session manager.

Provides a single async context for Playwright operations.
Core primitive: capture_state() — atomically grabs screenshot + a11y tree
+ any console errors / network failures accumulated since last call.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    async_playwright,
    ConsoleMessage,
    Request,
    Response,
)

from models import PageState


# How long to wait for network to go idle before capturing state.
NETWORK_IDLE_TIMEOUT_MS = 10_000
# How long to wait for a selector before declaring it absent.
SELECTOR_TIMEOUT_MS = 5_000


class BrowserSession:
    """
    Wraps a single Playwright browser context for the test harness.

    Always use as an async context manager via BrowserSession.create() —
    it handles browser launch and teardown automatically.

    Key methods
    -----------
    capture_state()             Atomically grab screenshot + a11y tree + any
                                console errors / network failures since the last
                                call. Buffers are cleared after each call so
                                successive captures only see new events.

    get_interactive_elements()  DOM scan returning every visible button, link,
                                input, select, and ARIA widget — used by the
                                planner to enumerate actions at each BFS node.

    get_geometry_violations()   Runs three layout checks directly in the page:
                                overflow_x (content wider than container),
                                overlap (two interactive elements sharing space),
                                viewport_clip (element outside visible viewport).

    navigate(url)               Go to a URL and wait for network idle.
    click(selector)             Click an element and wait for network idle.
    fill(selector, value)       Type into an input field.
    press(selector, key)        Send a key press (e.g. "Enter") to an element.
    element_exists(selector)    Non-throwing check — returns True/False.
    get_text(selector)          Return inner text of an element, or None.
    set_viewport(w, h)          Resize the browser window (used for multi-
                                viewport layout checks).

    Usage
    -----
        async with BrowserSession.create() as session:
            await session.navigate("http://localhost:5230")
            state = await session.capture_state()
    """

    def __init__(self, page: Page, context: BrowserContext, browser: Browser):
        self._page = page
        self._context = context
        self._browser = browser
        self._console_errors: list[str] = []
        self._network_failures: list[dict] = []
        self._attach_listeners()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    @asynccontextmanager
    async def create(
        cls,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        base_url: str = "",
    ) -> AsyncIterator["BrowserSession"]:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=headless)
            context = await browser.new_context(
                viewport={"width": viewport_width, "height": viewport_height},
                base_url=base_url,
            )
            page = await context.new_page()
            session = cls(page, context, browser)
            try:
                yield session
            finally:
                await browser.close()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, url: str) -> None:
        await self._page.goto(url, wait_until="domcontentloaded")
        await self._wait_stable()

    async def _wait_stable(self) -> None:
        try:
            await self._page.wait_for_load_state(
                "networkidle", timeout=NETWORK_IDLE_TIMEOUT_MS
            )
        except Exception:
            # networkidle timeout is non-fatal — page may have long-polling.
            pass

    # ------------------------------------------------------------------
    # Core primitive
    # ------------------------------------------------------------------

    async def capture_state(self) -> PageState:
        """
        Atomically capture screenshot + a11y tree + accumulated events.
        Clears the event buffers after capture so the next call only sees
        what happened since this one.
        """
        await self._wait_stable()

        screenshot = await self._page.screenshot(full_page=True)

        try:
            a11y_tree = await self._build_a11y_tree()
        except Exception:
            a11y_tree = {}

        console_errors = self._console_errors[:]
        network_failures = self._network_failures[:]
        self._console_errors.clear()
        self._network_failures.clear()

        return PageState(
            url=self._page.url,
            screenshot=screenshot,
            a11y_tree=a11y_tree,
            console_errors=console_errors,
            network_failures=network_failures,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    async def click(self, selector: str) -> None:
        await self._page.click(selector, timeout=SELECTOR_TIMEOUT_MS)
        await self._wait_stable()

    async def fill(self, selector: str, value: str) -> None:
        await self._page.fill(selector, value, timeout=SELECTOR_TIMEOUT_MS)

    async def press(self, selector: str, key: str) -> None:
        await self._page.press(selector, key, timeout=SELECTOR_TIMEOUT_MS)
        await self._wait_stable()

    async def element_exists(self, selector: str) -> bool:
        try:
            await self._page.wait_for_selector(
                selector, timeout=SELECTOR_TIMEOUT_MS, state="attached"
            )
            return True
        except Exception:
            return False

    async def get_text(self, selector: str) -> Optional[str]:
        try:
            el = await self._page.wait_for_selector(
                selector, timeout=SELECTOR_TIMEOUT_MS
            )
            return await el.inner_text() if el else None
        except Exception:
            return None

    async def set_viewport(self, width: int, height: int) -> None:
        await self._page.set_viewport_size({"width": width, "height": height})
        await self._wait_stable()

    # ------------------------------------------------------------------
    # DOM introspection
    # ------------------------------------------------------------------

    async def get_interactive_elements(self) -> list[dict]:
        """
        Return all interactive elements on the page with selector, role, label.
        Used by the planner's action identifier.
        """
        return await self._page.evaluate("""() => {
            const selectors = [
                'button:not([disabled])',
                'a[href]',
                'input:not([disabled])',
                'select:not([disabled])',
                'textarea:not([disabled])',
                '[role="button"]:not([disabled])',
                '[role="link"]',
                '[role="menuitem"]',
                '[role="tab"]',
                '[role="checkbox"]',
                '[role="switch"]',
                '[onclick]',
            ];
            const seen = new Set();
            const results = [];
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    if (seen.has(el)) continue;
                    seen.add(el);
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    results.push({
                        tag: el.tagName.toLowerCase(),
                        role: el.getAttribute('role') || el.tagName.toLowerCase(),
                        label: (
                            el.getAttribute('aria-label') ||
                            el.getAttribute('placeholder') ||
                            el.innerText?.trim().slice(0, 80) ||
                            el.getAttribute('title') ||
                            ''
                        ),
                        selector: buildSelector(el),
                        type: el.getAttribute('type') || '',
                        visible: rect.top >= 0 && rect.bottom <= window.innerHeight,
                    });
                }
            }
            function buildSelector(el) {
                if (el.id) return '#' + CSS.escape(el.id);
                if (el.getAttribute('data-testid'))
                    return `[data-testid="${el.getAttribute('data-testid')}"]`;
                if (el.getAttribute('aria-label'))
                    return `[aria-label="${el.getAttribute('aria-label')}"]`;
                const text = el.innerText?.trim().slice(0, 30);
                if (text) return `${el.tagName.toLowerCase()}:has-text("${text}")`;
                return el.tagName.toLowerCase();
            }
            return results;
        }""")

    async def get_geometry_violations(self) -> list[dict]:
        """
        Detect overflow and overlap violations via DOM geometry.
        Returns a list of violation dicts for the visual oracle.
        """
        return await self._page.evaluate("""() => {
            const violations = [];
            const vw = window.innerWidth;
            const vh = window.innerHeight;

            // Overflow check: content wider than its container.
            // Skip root elements (html/body) whose scroll width includes browser chrome.
            const OVERFLOW_THRESHOLD = 20;
            const skipTags = new Set(['HTML', 'BODY']);
            for (const el of document.querySelectorAll('*')) {
                if (skipTags.has(el.tagName)) continue;
                const style = window.getComputedStyle(el);
                if (style.overflow === 'hidden' || style.overflow === 'scroll') continue;
                if (el.scrollWidth > el.offsetWidth + OVERFLOW_THRESHOLD) {
                    violations.push({
                        type: 'overflow_x',
                        element: el.tagName + (el.id ? '#' + el.id : ''),
                        detail: `scrollWidth ${el.scrollWidth} > offsetWidth ${el.offsetWidth}`,
                    });
                }
            }

            // Viewport clip: element partially outside visible viewport
            const interactive = document.querySelectorAll('button, a, input, [role="button"]');
            for (const el of interactive) {
                const r = el.getBoundingClientRect();
                if (r.width === 0 && r.height === 0) continue;
                if (r.right > vw + 2 || r.bottom > vh + 2) {
                    violations.push({
                        type: 'viewport_clip',
                        element: el.tagName + (el.id ? '#' + el.id : ''),
                        detail: `element at (${Math.round(r.right)}, ${Math.round(r.bottom)}) outside viewport (${vw}, ${vh})`,
                    });
                }
            }

            // Overlap check: two sibling interactive elements occupying same space
            const els = Array.from(document.querySelectorAll('button, a[href], input'));
            for (let i = 0; i < els.length; i++) {
                for (let j = i + 1; j < els.length; j++) {
                    const a = els[i].getBoundingClientRect();
                    const b = els[j].getBoundingClientRect();
                    if (a.width === 0 || b.width === 0) continue;
                    const overlapX = Math.min(a.right, b.right) - Math.max(a.left, b.left);
                    const overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
                    if (overlapX > 5 && overlapY > 5) {
                        violations.push({
                            type: 'overlap',
                            element: `${els[i].tagName} + ${els[j].tagName}`,
                            detail: `overlap ${Math.round(overlapX)}x${Math.round(overlapY)}px`,
                        });
                        if (violations.length > 20) return violations;  // cap noise
                    }
                }
            }

            return violations;
        }""")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _build_a11y_tree(self) -> dict:
        """
        Build a lightweight accessibility tree via JS.
        More reliable than page.accessibility.snapshot() which returns None
        for pages loaded via set_content or certain SPA states.
        """
        return await self._page.evaluate("""() => {
            function nodeToTree(el, depth) {
                if (depth > 10) return null;
                const role = el.getAttribute('role') ||
                    ({'A': 'link', 'BUTTON': 'button', 'INPUT': 'textbox',
                      'SELECT': 'combobox', 'TEXTAREA': 'textbox',
                      'H1': 'heading', 'H2': 'heading', 'H3': 'heading',
                      'H4': 'heading', 'H5': 'heading', 'H6': 'heading',
                      'NAV': 'navigation', 'MAIN': 'main', 'HEADER': 'banner',
                      'FOOTER': 'contentinfo', 'FORM': 'form', 'IMG': 'img',
                      'UL': 'list', 'OL': 'list', 'LI': 'listitem',
                    }[el.tagName] || el.tagName.toLowerCase());
                const name = (
                    el.getAttribute('aria-label') ||
                    el.getAttribute('alt') ||
                    el.getAttribute('placeholder') ||
                    el.getAttribute('title') ||
                    (el.tagName !== 'DIV' && el.tagName !== 'SPAN'
                        ? el.innerText?.trim().slice(0, 80) : '') || ''
                );
                const children = [];
                for (const child of el.children) {
                    const subtree = nodeToTree(child, depth + 1);
                    if (subtree) children.push(subtree);
                }
                return { role, name, tag: el.tagName.toLowerCase(),
                         id: el.id || undefined, children };
            }
            return nodeToTree(document.body || document.documentElement, 0) || {};
        }""")

    def _attach_listeners(self) -> None:
        self._page.on("console", self._on_console)
        self._page.on("response", self._on_response)
        self._page.on("pageerror", self._on_page_error)

    def _on_console(self, msg: ConsoleMessage) -> None:
        if msg.type in ("error", "warning"):
            self._console_errors.append(f"[{msg.type}] {msg.text}")

    def _on_page_error(self, error: Exception) -> None:
        self._console_errors.append(f"[pageerror] {error}")

    def _on_response(self, response: Response) -> None:
        if response.status >= 400:
            self._network_failures.append(
                {"url": response.url, "status": response.status}
            )

    @property
    def page(self) -> Page:
        return self._page
