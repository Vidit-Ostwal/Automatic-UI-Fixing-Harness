import json
from pathlib import Path

from harness.reporter.collector import load_report
from harness.reporter.render import render_html


def test_render_html_writes_report(tmp_path: Path):
    claims_dir = tmp_path / "verifier_claims" / "T-007_run1"
    claims_dir.mkdir(parents=True)
    (claims_dir / "claims.json").write_text(json.dumps({
        "test_case_id": "T-007",
        "run_id": "run1",
        "goal": "Verify calendar",
        "description": "create_account → june",
        "total_steps_verified": 2,
        "total_findings": 1,
        "findings": [{
            "id": "V-T-007-001",
            "step_index": 2,
            "instruction": "Click sign up",
            "bug_type": "visual",
            "severity": "high",
            "title": "Content faded",
            "description": "Low opacity text",
            "evidence": "Screenshot shows faint text",
            "reproduction_steps": ["Sign up"],
            "screenshot_before": "executor_runs/T-007_run1/step_02_before.png",
            "screenshot_after": "executor_runs/T-007_run1/step_02_after.png",
        }],
    }))

    report = load_report(tmp_path)
    path = render_html(report, tmp_path)
    html = path.read_text()

    assert path.name == "report.html"
    assert "Content faded" in html
    assert "T-007" in html
    assert "executor_runs/T-007_run1/step_02_before.png" in html
    assert "<!DOCTYPE html>" in html
