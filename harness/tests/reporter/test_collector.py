import json
from pathlib import Path

from harness.reporter.collector import load_report


def _write_claims(root: Path, folder: str, payload: dict) -> None:
    d = root / "verifier_claims" / folder
    d.mkdir(parents=True)
    (d / "claims.json").write_text(json.dumps(payload))


def test_load_report_empty_when_no_claims(tmp_path: Path):
    report = load_report(tmp_path)
    assert report.total_runs == 0
    assert report.total_findings == 0


def test_load_report_aggregates_findings(tmp_path: Path):
    _write_claims(tmp_path, "T-001_abc", {
        "test_case_id": "T-001",
        "run_id": "abc",
        "goal": "Verify signup",
        "description": "create_account",
        "total_steps_verified": 3,
        "total_findings": 1,
        "findings": [{
            "id": "V-T-001-001",
            "step_index": 1,
            "instruction": "Click sign up",
            "bug_type": "visual",
            "severity": "high",
            "title": "Button clipped",
            "description": "The button is cut off",
            "evidence": "Screenshot shows clipping",
            "reproduction_steps": ["Open page", "Resize window"],
            "screenshot_before": "executor_runs/T-001_abc/step_01_before.png",
            "screenshot_after": "executor_runs/T-001_abc/step_01_after.png",
        }],
    })
    _write_claims(tmp_path, "T-002_def", {
        "test_case_id": "T-002",
        "run_id": "def",
        "goal": "Verify memo",
        "description": "create_memo",
        "total_steps_verified": 5,
        "total_findings": 0,
        "findings": [],
    })

    report = load_report(tmp_path)
    assert report.total_runs == 2
    assert report.total_findings == 1
    assert report.clean_runs == 1
    assert report.severity_counts["high"] == 1
    assert report.findings[0].title == "Button clipped"
