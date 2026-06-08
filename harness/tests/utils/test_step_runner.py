import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.utils.step_runner import wait_for_settle


@pytest.mark.asyncio
async def test_wait_for_settle_waits_for_dom_stability():
    page = MagicMock()
    page.url = "http://localhost:5230/"
    signatures = ["10|5", "20|8", "20|8", "20|8", "20|8", "20|8"]
    page.evaluate = AsyncMock(side_effect=signatures + ["done"])

    session = MagicMock()
    session.page = page
    session._wait_stable = AsyncMock()

    await wait_for_settle(session, prev_url="http://localhost:5230/", timeout_ms=2000)

    assert session._wait_stable.await_count >= 1
    assert page.evaluate.await_count >= 4


@pytest.mark.asyncio
async def test_wait_for_settle_always_finishes_with_wait_stable():
    page = MagicMock()
    page.url = "http://localhost:5230/"
    page.evaluate = AsyncMock(return_value="1|1")

    session = MagicMock()
    session.page = page
    session._wait_stable = AsyncMock()

    await wait_for_settle(session, prev_url=None, timeout_ms=800)

    session._wait_stable.assert_awaited()
