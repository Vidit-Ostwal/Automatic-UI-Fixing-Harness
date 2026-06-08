"""
Load verifier claims from output/verifier_claims into a unified report model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReportFinding:
    id: str
    test_case_id: str
    run_id: str
    goal: str
    trajectory: str
    step_index: int
    instruction: str
    bug_type: str
    severity: str
    title: str
    description: str
    evidence: str
    reproduction_steps: list[str]
    screenshot_before: str
    screenshot_after: str


@dataclass
class TestRunReport:
    test_case_id: str
    run_id: str
    goal: str
    description: str
    total_steps_verified: int
    total_findings: int
    findings: list[ReportFinding] = field(default_factory=list)
    claims_path: str = ""


@dataclass
class HarnessReport:
    output_dir: str
    runs: list[TestRunReport] = field(default_factory=list)
    findings: list[ReportFinding] = field(default_factory=list)

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def total_findings(self) -> int:
        return len(self.findings)

    @property
    def severity_counts(self) -> dict[str, int]:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in self.findings:
            sev = f.severity if f.severity in counts else "low"
            counts[sev] += 1
        return counts

    @property
    def clean_runs(self) -> int:
        return sum(1 for r in self.runs if r.total_findings == 0)


def load_report(output_dir: Path) -> HarnessReport:
    """
    Scan output_dir/verifier_claims/*/claims.json and build a HarnessReport.
    """
    claims_root = output_dir / "verifier_claims"
    report = HarnessReport(output_dir=str(output_dir.resolve()))

    if not claims_root.is_dir():
        return report

    for claims_path in sorted(claims_root.glob("*/claims.json")):
        raw = json.loads(claims_path.read_text())
        test_case_id = raw.get("test_case_id", "T-???")
        run_id = raw.get("run_id", "unknown")
        goal = raw.get("goal", "")
        description = raw.get("description", "")

        run_findings: list[ReportFinding] = []
        for f in raw.get("findings", []):
            finding = ReportFinding(
                id=f.get("id", ""),
                test_case_id=test_case_id,
                run_id=run_id,
                goal=goal,
                trajectory=description,
                step_index=f.get("step_index", 0),
                instruction=f.get("instruction", ""),
                bug_type=f.get("bug_type", "logic"),
                severity=f.get("severity", "medium"),
                title=f.get("title", ""),
                description=f.get("description", ""),
                evidence=f.get("evidence", ""),
                reproduction_steps=f.get("reproduction_steps", []),
                screenshot_before=f.get("screenshot_before", ""),
                screenshot_after=f.get("screenshot_after", ""),
            )
            run_findings.append(finding)
            report.findings.append(finding)

        report.runs.append(
            TestRunReport(
                test_case_id=test_case_id,
                run_id=run_id,
                goal=goal,
                description=description,
                total_steps_verified=raw.get("total_steps_verified", 0),
                total_findings=raw.get("total_findings", len(run_findings)),
                findings=run_findings,
                claims_path=str(claims_path.relative_to(output_dir)),
            )
        )

    _sort_findings(report.findings)
    return report


def _sort_findings(findings: list[ReportFinding]) -> None:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    findings.sort(
        key=lambda f: (
            order.get(f.severity, 4),
            f.test_case_id,
            f.step_index,
            f.id,
        )
    )
