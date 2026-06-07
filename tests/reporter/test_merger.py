"""
Tests for reporter/merger.py
"""

import pytest
from models import BugType, DetectedBy, Finding, Severity, SuppressedNoise
from reporter.collector import FindingCollector
from reporter.merger import deduplicate, merge, _fingerprint, _evidence_score


def _finding(
    title="Bug",
    severity=Severity.HIGH,
    bug_type=BugType.LOGIC,
    fid=None,
    screenshot_after=None,
    steps=None,
    reasoning="test",
) -> Finding:
    return Finding(
        id=fid or f"TEST-{title[:6]}",
        title=title,
        bug_type=bug_type,
        severity=severity,
        steps=steps or [],
        detected_by=DetectedBy.HEURISTIC,
        reasoning=reasoning,
        screenshot_after=screenshot_after,
    )


def _collector(*findings, trajectory_id="") -> FindingCollector:
    c = FindingCollector(trajectory_id=trajectory_id)
    for f in findings:
        c.add(f)
    return c


# ---------------------------------------------------------------------------
# _fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_same_title_same_key():
    f1 = _finding("Overlapping buttons")
    f2 = _finding("Overlapping buttons")
    assert _fingerprint(f1) == _fingerprint(f2)


def test_fingerprint_strips_uuid_suffix():
    """Auto-generated IDs like 'VIS-abc123' appear in titles; strip them."""
    f1 = _finding("Element overflow: DIV#abc123")
    f2 = _finding("Element overflow: DIV#def456")
    assert _fingerprint(f1) == _fingerprint(f2)


def test_fingerprint_different_severity_different_key():
    f1 = _finding("Same title", severity=Severity.HIGH)
    f2 = _finding("Same title", severity=Severity.LOW)
    assert _fingerprint(f1) != _fingerprint(f2)


def test_fingerprint_different_bug_type_different_key():
    f1 = _finding("Same title", bug_type=BugType.VISUAL)
    f2 = _finding("Same title", bug_type=BugType.LOGIC)
    assert _fingerprint(f1) != _fingerprint(f2)


def test_fingerprint_case_insensitive():
    f1 = _finding("Overlapping Buttons")
    f2 = _finding("overlapping buttons")
    assert _fingerprint(f1) == _fingerprint(f2)


# ---------------------------------------------------------------------------
# _evidence_score
# ---------------------------------------------------------------------------

def test_evidence_score_no_evidence():
    assert _evidence_score(_finding()) == 1  # reasoning = +1


def test_evidence_score_with_screenshots():
    f = _finding(screenshot_after=b"\x89PNG")
    score = _evidence_score(f)
    assert score >= 3  # screenshot_after=+2, reasoning=+1


def test_evidence_score_with_steps():
    f = _finding(steps=["step 1", "step 2"])
    assert _evidence_score(f) >= 3


def test_evidence_score_ordering():
    plain = _finding()
    rich  = _finding(screenshot_after=b"\x89PNG", steps=["s1", "s2"])
    assert _evidence_score(rich) > _evidence_score(plain)


# ---------------------------------------------------------------------------
# deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_keeps_unique_findings():
    findings = [
        _finding("Bug A", Severity.HIGH),
        _finding("Bug B", Severity.MEDIUM),
        _finding("Bug C", Severity.LOW),
    ]
    result = deduplicate(findings)
    assert len(result) == 3


def test_deduplicate_removes_exact_duplicates():
    f1 = _finding("Overlapping buttons", fid="ID-1")
    f2 = _finding("Overlapping buttons", fid="ID-2")
    result = deduplicate([f1, f2])
    assert len(result) == 1


def test_deduplicate_keeps_richer_duplicate():
    plain = _finding("Overlapping buttons", fid="plain")
    rich  = _finding("Overlapping buttons", fid="rich", screenshot_after=b"\x89PNG", steps=["s"])
    result = deduplicate([plain, rich])
    assert len(result) == 1
    assert result[0].id == "rich"


def test_deduplicate_sorted_by_severity():
    findings = [
        _finding("Low bug",      severity=Severity.LOW),
        _finding("Critical bug", severity=Severity.CRITICAL),
        _finding("Medium bug",   severity=Severity.MEDIUM),
        _finding("High bug",     severity=Severity.HIGH),
    ]
    result = deduplicate(findings)
    order = [f.severity.value for f in result]
    assert order == ["critical", "high", "medium", "low"]


def test_deduplicate_empty_list():
    assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def test_merge_combines_collectors():
    c1 = _collector(_finding("Bug A"), _finding("Bug B"))
    c2 = _collector(_finding("Bug C"))
    findings, noise = merge([c1, c2])
    assert len(findings) == 3


def test_merge_deduplicates_across_collectors():
    c1 = _collector(_finding("Overlapping buttons", fid="ID-1"))
    c2 = _collector(_finding("Overlapping buttons", fid="ID-2"))
    findings, _ = merge([c1, c2])
    assert len(findings) == 1


def test_merge_combines_noise():
    c1 = FindingCollector()
    c1.suppress("Spinner A", "loading")
    c2 = FindingCollector()
    c2.suppress("Spinner B", "loading")
    _, noise = merge([c1, c2])
    assert len(noise) == 2


def test_merge_empty_collectors():
    findings, noise = merge([FindingCollector(), FindingCollector()])
    assert findings == []
    assert noise == []


def test_merge_single_collector():
    c = _collector(_finding("Only bug"))
    findings, _ = merge([c])
    assert len(findings) == 1


def test_merge_preserves_trajectory_ids():
    c1 = _collector(_finding("Bug T1"), trajectory_id="T-001")
    c2 = _collector(_finding("Bug T2"), trajectory_id="T-002")
    for f in c1.findings:
        assert f.trajectory_id == "T-001"
    for f in c2.findings:
        assert f.trajectory_id == "T-002"
