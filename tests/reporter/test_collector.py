"""
Tests for reporter/collector.py
"""

import pytest
from models import BugType, DetectedBy, Finding, Severity, SuppressedNoise
from reporter.collector import FindingCollector, CollectorStats


def _finding(title="Bug", severity=Severity.HIGH, bug_type=BugType.LOGIC) -> Finding:
    return Finding(
        id=f"TEST-{title[:4]}",
        title=title,
        bug_type=bug_type,
        severity=severity,
        steps=[],
        detected_by=DetectedBy.HEURISTIC,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# Basic accumulation
# ---------------------------------------------------------------------------

def test_empty_collector():
    c = FindingCollector()
    assert c.findings == []
    assert c.noise == []
    assert len(c) == 0


def test_add_finding():
    c = FindingCollector()
    f = _finding("A bug")
    c.add(f)
    assert len(c) == 1
    assert c.findings[0].title == "A bug"


def test_add_multiple_findings():
    c = FindingCollector()
    c.add(_finding("Bug 1"))
    c.add(_finding("Bug 2"))
    c.add(_finding("Bug 3"))
    assert len(c) == 3


def test_findings_returns_copy():
    """Mutating the returned list must not affect the collector."""
    c = FindingCollector()
    c.add(_finding("Bug"))
    result = c.findings
    result.clear()
    assert len(c) == 1


def test_suppress_noise():
    c = FindingCollector()
    c.suppress("Loading spinner visible", "networkidle wait")
    assert len(c.noise) == 1
    assert c.noise[0].description == "Loading spinner visible"
    assert c.noise[0].reason == "networkidle wait"


def test_suppress_multiple():
    c = FindingCollector()
    c.suppress("A", "reason A")
    c.suppress("B", "reason B")
    assert len(c.noise) == 2


def test_noise_returns_copy():
    c = FindingCollector()
    c.suppress("Noise", "reason")
    result = c.noise
    result.clear()
    assert len(c.noise) == 1


# ---------------------------------------------------------------------------
# Trajectory ID stamping
# ---------------------------------------------------------------------------

def test_trajectory_id_stamped_from_constructor():
    c = FindingCollector(trajectory_id="T-001")
    f = _finding("Bug")
    assert f.trajectory_id == ""
    c.add(f)
    assert f.trajectory_id == "T-001"


def test_trajectory_id_override_at_add():
    c = FindingCollector(trajectory_id="T-001")
    f = _finding("Bug")
    c.add(f, trajectory_id="T-999")
    assert f.trajectory_id == "T-999"


def test_trajectory_id_not_overwritten_if_set():
    c = FindingCollector(trajectory_id="T-001")
    f = _finding("Bug")
    f.trajectory_id = "already-set"
    c.add(f)
    assert f.trajectory_id == "already-set"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_stats_counts_by_severity():
    c = FindingCollector()
    c.add(_finding("A", Severity.CRITICAL))
    c.add(_finding("B", Severity.HIGH))
    c.add(_finding("C", Severity.HIGH))
    c.add(_finding("D", Severity.MEDIUM))
    c.add(_finding("E", Severity.LOW))
    s = c.stats()
    assert s.critical == 1
    assert s.high == 2
    assert s.medium == 1
    assert s.low == 1
    assert s.total == 5


def test_stats_counts_by_type():
    c = FindingCollector()
    c.add(_finding("A", bug_type=BugType.VISUAL))
    c.add(_finding("B", bug_type=BugType.VISUAL))
    c.add(_finding("C", bug_type=BugType.LOGIC))
    s = c.stats()
    assert s.visual == 2
    assert s.logic == 1


def test_stats_counts_suppressed():
    c = FindingCollector()
    c.suppress("A", "r")
    c.suppress("B", "r")
    s = c.stats()
    assert s.suppressed == 2


def test_stats_empty_collector():
    s = FindingCollector().stats()
    assert s.total == 0
    assert s.critical == 0
    assert s.suppressed == 0
