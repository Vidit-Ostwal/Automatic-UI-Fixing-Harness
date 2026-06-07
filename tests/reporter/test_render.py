"""
Tests for reporter/render.py

Tests verify structure of JSON output and key elements in HTML output.
No browser required.
"""

import json
import base64
from pathlib import Path
import pytest

from models import BugType, DetectedBy, Finding, Severity, SuppressedNoise
from reporter.render import RunReport, render_json, render_html


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _finding(
    title="Test bug",
    severity=Severity.HIGH,
    bug_type=BugType.LOGIC,
    fid="BUG-001",
    steps=None,
    screenshot_before=None,
    screenshot_after=None,
    console_errors=None,
    trajectory_id="T-001",
    reasoning="Something went wrong.",
) -> Finding:
    return Finding(
        id=fid,
        title=title,
        bug_type=bug_type,
        severity=severity,
        steps=steps or ["Step 1: Sign up", "Step 2: Create memo"],
        detected_by=DetectedBy.HEURISTIC,
        reasoning=reasoning,
        trajectory_id=trajectory_id,
        screenshot_before=screenshot_before,
        screenshot_after=screenshot_after,
        console_errors=console_errors or [],
    )


def _report(findings=None, noise=None) -> RunReport:
    return RunReport(
        run_id="testrun001",
        app_url="http://localhost:5230",
        findings=findings or [],
        suppressed_noise=noise or [],
        trajectories_explored=5,
        duration_seconds=42.3,
    )


# ---------------------------------------------------------------------------
# render_json — structure
# ---------------------------------------------------------------------------

def test_render_json_creates_file(tmp_path):
    report = _report()
    path = render_json(report, tmp_path)
    assert path.exists()
    assert path.name == "report.json"


def test_render_json_top_level_keys(tmp_path):
    report = _report()
    render_json(report, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    for key in ("run_id", "timestamp", "app_url", "trajectories_explored",
                "duration_seconds", "summary", "findings", "suppressed_noise"):
        assert key in data, f"Missing key: {key}"


def test_render_json_run_id(tmp_path):
    report = _report()
    render_json(report, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["run_id"] == "testrun001"


def test_render_json_summary_counts(tmp_path):
    findings = [
        _finding("Critical", severity=Severity.CRITICAL, fid="C1"),
        _finding("High",     severity=Severity.HIGH,     fid="H1"),
        _finding("High 2",   severity=Severity.HIGH,     fid="H2"),
        _finding("Low",      severity=Severity.LOW,      fid="L1", bug_type=BugType.VISUAL),
    ]
    render_json(_report(findings), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    s = data["summary"]
    assert s["total"]    == 4
    assert s["critical"] == 1
    assert s["high"]     == 2
    assert s["low"]      == 1
    assert s["visual"]   == 1
    assert s["logic"]    == 3


def test_render_json_finding_fields(tmp_path):
    f = _finding()
    render_json(_report([f]), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    entry = data["findings"][0]
    assert entry["id"]            == "BUG-001"
    assert entry["title"]         == "Test bug"
    assert entry["type"]          == "logic"
    assert entry["severity"]      == "high"
    assert entry["trajectory_id"] == "T-001"
    assert entry["reasoning"]     == "Something went wrong."
    assert "Step 1: Sign up" in entry["steps"]


def test_render_json_no_screenshot_when_none(tmp_path):
    f = _finding()
    render_json(_report([f]), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    evidence = data["findings"][0]["evidence"]
    assert "screenshot_before" not in evidence
    assert "screenshot_after"  not in evidence


def test_render_json_saves_screenshot_files(tmp_path):
    f = _finding(screenshot_before=FAKE_PNG, screenshot_after=FAKE_PNG)
    render_json(_report([f]), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    evidence = data["findings"][0]["evidence"]
    assert "screenshot_before" in evidence
    assert "screenshot_after"  in evidence
    # Files must actually exist.
    assert (tmp_path / evidence["screenshot_before"]).exists()
    assert (tmp_path / evidence["screenshot_after"]).exists()


def test_render_json_suppressed_noise(tmp_path):
    noise = [SuppressedNoise("Loading spinner", "networkidle wait")]
    render_json(_report(noise=noise), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert len(data["suppressed_noise"]) == 1
    assert data["suppressed_noise"][0]["description"] == "Loading spinner"
    assert data["suppressed_noise"][0]["reason"]      == "networkidle wait"


def test_render_json_empty_report(tmp_path):
    render_json(_report(), tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["findings"] == []
    assert data["summary"]["total"] == 0


def test_render_json_trajectories_and_duration(tmp_path):
    report = _report()
    render_json(report, tmp_path)
    data = json.loads((tmp_path / "report.json").read_text())
    assert data["trajectories_explored"] == 5
    assert data["duration_seconds"] == 42.3


# ---------------------------------------------------------------------------
# render_html — structure
# ---------------------------------------------------------------------------

def test_render_html_creates_file(tmp_path):
    path = render_html(_report(), tmp_path)
    assert path.exists()
    assert path.name == "report.html"


def test_render_html_is_valid_html(tmp_path):
    render_html(_report(), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "<!DOCTYPE html>" in content
    assert "<html" in content
    assert "</html>" in content


def test_render_html_contains_run_id(tmp_path):
    render_html(_report(), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "testrun001" in content


def test_render_html_contains_app_url(tmp_path):
    render_html(_report(), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "http://localhost:5230" in content


def test_render_html_contains_finding_title(tmp_path):
    findings = [_finding("Pin button has no effect")]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "Pin button has no effect" in content


def test_render_html_contains_severity_badge(tmp_path):
    findings = [_finding(severity=Severity.CRITICAL, fid="C1")]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "CRITICAL" in content


def test_render_html_embeds_screenshot_as_base64(tmp_path):
    findings = [_finding(screenshot_after=FAKE_PNG)]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    expected_prefix = "data:image/png;base64,"
    assert expected_prefix in content


def test_render_html_contains_reproduction_steps(tmp_path):
    findings = [_finding(steps=["Sign up as admin", "Create a memo"])]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "Sign up as admin" in content
    assert "Create a memo" in content


def test_render_html_contains_suppressed_noise_section(tmp_path):
    noise = [SuppressedNoise("Loading spinner", "networkidle wait")]
    render_html(_report(noise=noise), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "Suppressed Noise" in content
    assert "Loading spinner" in content
    assert "networkidle wait" in content


def test_render_html_no_findings_message(tmp_path):
    render_html(_report(), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "No findings detected" in content


def test_render_html_multiple_severities(tmp_path):
    findings = [
        _finding("High bug",   severity=Severity.HIGH,   fid="H1"),
        _finding("Low bug",    severity=Severity.LOW,    fid="L1"),
    ]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert "HIGH" in content
    assert "LOW" in content


def test_render_html_self_contained_no_external_links(tmp_path):
    """HTML report must not reference external stylesheets or scripts."""
    findings = [_finding(screenshot_after=FAKE_PNG)]
    render_html(_report(findings), tmp_path)
    content = (tmp_path / "report.html").read_text()
    assert '<link rel="stylesheet"' not in content
    assert "<script src=" not in content


# ---------------------------------------------------------------------------
# RunReport auto-fields
# ---------------------------------------------------------------------------

def test_run_report_auto_timestamp():
    r = RunReport(run_id="abc", app_url="http://localhost", findings=[])
    assert r.timestamp != ""
    assert "T" in r.timestamp  # ISO format


def test_run_report_auto_run_id():
    r = RunReport(run_id="", app_url="http://localhost", findings=[])
    assert r.run_id != ""
    assert len(r.run_id) >= 8
