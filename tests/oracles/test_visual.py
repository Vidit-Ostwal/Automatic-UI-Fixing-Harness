"""
Tests for oracles/visual.py

Browser-based tests using inline HTML with known geometry violations.
Also tests the retry confirmation logic and LLM fallback with mocks.
"""

import pytest
from browser.session import BrowserSession
from models import BugType, DetectedBy, Finding, Severity
from oracles.visual import VisualOracle


# ---------------------------------------------------------------------------
# HTML fixtures
# ---------------------------------------------------------------------------

HTML_CLEAN = """
<html><body style="margin:16px;">
  <h1>Clean Page</h1>
  <button id="a" style="margin:4px;">Save</button>
  <button id="b" style="margin:4px; margin-left: 120px;">Cancel</button>
</body></html>
"""

HTML_OVERFLOW = """
<html><body style="margin:0;padding:0;">
  <div id="box" style="width:100px;overflow:visible;">
    <div id="wide" style="width:600px;white-space:nowrap;">
      This content is way too wide for its container element
    </div>
  </div>
</body></html>
"""

HTML_OVERLAP = """
<html><body style="margin:0;padding:0;position:relative;">
  <button style="position:absolute;top:10px;left:10px;width:120px;height:40px;">Button A</button>
  <button style="position:absolute;top:10px;left:40px;width:120px;height:40px;">Button B</button>
</body></html>
"""

HTML_VIEWPORT_CLIP = """
<html><body style="margin:0;padding:0;overflow:hidden;">
  <button style="position:absolute;top:10px;left:1500px;width:100px;height:40px;">Off Screen</button>
  <button style="position:absolute;top:10px;left:10px;width:100px;height:40px;">On Screen</button>
</body></html>
"""


# ---------------------------------------------------------------------------
# Mock LLM oracle
# ---------------------------------------------------------------------------

class _MockLLMBug:
    async def judge_screenshot(self, screenshot, context=""):
        from oracles.llm import OracleVerdict
        return OracleVerdict(
            verdict="bug",
            description="Visual defect detected by LLM",
            severity="medium",
            reasoning="LLM noticed something wrong.",
        )


class _MockLLMOk:
    async def judge_screenshot(self, screenshot, context=""):
        from oracles.llm import OracleVerdict
        return OracleVerdict(
            verdict="ok",
            description="Looks fine",
            severity=None,
            reasoning="No issues.",
        )


# ---------------------------------------------------------------------------
# Tests — no LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_clean_page_no_findings():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_CLEAN)
        oracle = VisualOracle()
        findings = await oracle.check(session, "clean page")
    assert findings == []


@pytest.mark.asyncio
async def test_overflow_detected():
    async with BrowserSession.create(viewport_width=800) as session:
        await session.page.set_content(HTML_OVERFLOW)
        oracle = VisualOracle()
        findings = await oracle.check(session, "overflow test")
    types = [f.title for f in findings]
    assert any("overflow" in t.lower() or "Overflow" in t for t in types)


@pytest.mark.asyncio
async def test_overlap_detected():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session, "overlap test")
    types = [f.title for f in findings]
    assert any("overlap" in t.lower() or "Overlapping" in t for t in types)


@pytest.mark.asyncio
async def test_viewport_clip_detected():
    async with BrowserSession.create(viewport_width=1280) as session:
        await session.page.set_content(HTML_VIEWPORT_CLIP)
        oracle = VisualOracle()
        findings = await oracle.check(session, "clip test")
    types = [f.title for f in findings]
    assert any("clip" in t.lower() or "viewport" in t.lower() for t in types)


@pytest.mark.asyncio
async def test_findings_are_finding_instances():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session)
    for f in findings:
        assert isinstance(f, Finding)


@pytest.mark.asyncio
async def test_findings_have_correct_bug_type():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session)
    for f in findings:
        assert f.bug_type == BugType.VISUAL


@pytest.mark.asyncio
async def test_findings_detected_by_heuristic():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session)
    heuristic = [f for f in findings if f.detected_by == DetectedBy.HEURISTIC]
    assert len(heuristic) > 0


@pytest.mark.asyncio
async def test_step_context_in_finding_steps():
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session, step_context="After pinning a memo")
    for f in findings:
        assert "After pinning a memo" in f.steps


@pytest.mark.asyncio
async def test_severity_mapping():
    """overlap → HIGH, viewport_clip → MEDIUM, overflow_x → LOW."""
    async with BrowserSession.create(viewport_width=1280) as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle()
        findings = await oracle.check(session)
    overlap_findings = [f for f in findings if "Overlapping" in f.title]
    assert all(f.severity == Severity.HIGH for f in overlap_findings)


# ---------------------------------------------------------------------------
# Tests — with LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_finding_added_when_violations_exist():
    """When geometry violations exist AND LLM says bug, an LLM finding is added."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle(llm_oracle=_MockLLMBug())
        findings = await oracle.check(session, "overlap test")

    llm_findings = [f for f in findings if f.detected_by == DetectedBy.LLM]
    assert len(llm_findings) >= 1
    assert llm_findings[0].severity == Severity.MEDIUM


@pytest.mark.asyncio
async def test_llm_finding_not_added_when_ok():
    """LLM saying 'ok' should not add an extra finding."""
    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_OVERLAP)
        oracle = VisualOracle(llm_oracle=_MockLLMOk())
        findings = await oracle.check(session, "overlap test")

    llm_findings = [f for f in findings if f.detected_by == DetectedBy.LLM]
    assert len(llm_findings) == 0


@pytest.mark.asyncio
async def test_llm_not_called_on_clean_page():
    """LLM oracle should not be consulted when there are no geometry violations."""
    call_count = {"n": 0}

    class _CountingLLM:
        async def judge_screenshot(self, screenshot, context=""):
            call_count["n"] += 1
            from oracles.llm import OracleVerdict
            return OracleVerdict("ok", "", None, "")

    async with BrowserSession.create() as session:
        await session.page.set_content(HTML_CLEAN)
        oracle = VisualOracle(llm_oracle=_CountingLLM())
        await oracle.check(session)

    assert call_count["n"] == 0
