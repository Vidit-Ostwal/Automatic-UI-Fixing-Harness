"""
Report renderer — writes report.json and report.html to an output directory.

report.json   Machine-readable structured findings (screenshots as file paths).
report.html   Self-contained human-readable report (screenshots embedded as
              base64 data-URLs — open one file, see everything).
"""

import base64
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models import Finding, SuppressedNoise


# ---------------------------------------------------------------------------
# RunReport
# ---------------------------------------------------------------------------

@dataclass
class RunReport:
    run_id: str
    app_url: str
    findings: list[Finding]
    suppressed_noise: list[SuppressedNoise] = field(default_factory=list)
    trajectories_explored: int = 0
    duration_seconds: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.run_id:
            self.run_id = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------

def _screenshot_path(finding_id: str, which: str, screenshots_dir: Path) -> Optional[str]:
    return str(screenshots_dir / f"{finding_id}_{which}.png")


def _save_screenshot(data: bytes, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def render_json(report: RunReport, output_dir: Path) -> Path:
    """Write report.json and save any embedded screenshots as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_dir / "screenshots"

    findings_data = []
    for f in report.findings:
        entry: dict = {
            "id":            f.id,
            "title":         f.title,
            "type":          f.bug_type.value,
            "severity":      f.severity.value,
            "trajectory_id": f.trajectory_id,
            "steps":         f.steps,
            "detected_by":   f.detected_by.value,
            "reasoning":     f.reasoning,
            "console_errors": f.console_errors,
            "network_errors": f.network_errors,
            "evidence":      {},
        }

        if f.screenshot_before:
            path = _screenshot_path(f.id, "before", screenshots_dir)
            _save_screenshot(f.screenshot_before, path)
            entry["evidence"]["screenshot_before"] = str(
                Path(path).relative_to(output_dir)
            )
        if f.screenshot_after:
            path = _screenshot_path(f.id, "after", screenshots_dir)
            _save_screenshot(f.screenshot_after, path)
            entry["evidence"]["screenshot_after"] = str(
                Path(path).relative_to(output_dir)
            )

        findings_data.append(entry)

    noise_data = [
        {"description": n.description, "reason": n.reason}
        for n in report.suppressed_noise
    ]

    payload = {
        "run_id":                report.run_id,
        "timestamp":             report.timestamp,
        "app_url":               report.app_url,
        "trajectories_explored": report.trajectories_explored,
        "duration_seconds":      round(report.duration_seconds, 1),
        "summary": {
            "total":    len(report.findings),
            "critical": sum(1 for f in report.findings if f.severity.value == "critical"),
            "high":     sum(1 for f in report.findings if f.severity.value == "high"),
            "medium":   sum(1 for f in report.findings if f.severity.value == "medium"),
            "low":      sum(1 for f in report.findings if f.severity.value == "low"),
            "visual":   sum(1 for f in report.findings if f.bug_type.value == "visual"),
            "logic":    sum(1 for f in report.findings if f.bug_type.value == "logic"),
        },
        "findings":         findings_data,
        "suppressed_noise": noise_data,
    }

    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------

_SEVERITY_COLOR = {
    "critical": ("#dc2626", "#fee2e2"),
    "high":     ("#ea580c", "#ffedd5"),
    "medium":   ("#d97706", "#fef3c7"),
    "low":      ("#2563eb", "#dbeafe"),
}

_TYPE_COLOR = {
    "visual": ("#7c3aed", "#ede9fe"),
    "logic":  ("#0891b2", "#cffafe"),
}


def _b64_img(data: bytes) -> str:
    return "data:image/png;base64," + base64.standard_b64encode(data).decode()


def _badge(text: str, fg: str, bg: str) -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:600;'
        f'white-space:nowrap;">{text.upper()}</span>'
    )


def _screenshot_block(label: str, data: bytes) -> str:
    src = _b64_img(data)
    return (
        f'<div style="margin-top:8px;">'
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:4px;">{label}</div>'
        f'<img src="{src}" style="max-width:100%;border:1px solid #e5e7eb;'
        f'border-radius:4px;" />'
        f"</div>"
    )


def _finding_card(f: Finding, index: int) -> str:
    sev_fg, sev_bg = _SEVERITY_COLOR.get(f.severity.value, ("#374151", "#f3f4f6"))
    typ_fg, typ_bg = _TYPE_COLOR.get(f.bug_type.value,   ("#374151", "#f3f4f6"))

    steps_html = ""
    if f.steps:
        items = "".join(f"<li>{s}</li>" for s in f.steps)
        steps_html = (
            f'<div style="margin-top:10px;">'
            f'<strong>Reproduction steps</strong>'
            f'<ol style="margin:6px 0 0 20px;padding:0;">{items}</ol>'
            f"</div>"
        )

    screenshots_html = ""
    if f.screenshot_before:
        screenshots_html += _screenshot_block("Before", f.screenshot_before)
    if f.screenshot_after:
        screenshots_html += _screenshot_block("After", f.screenshot_after)

    console_html = ""
    if f.console_errors:
        errs = "".join(
            f'<li style="font-family:monospace;font-size:11px;">{e}</li>'
            for e in f.console_errors[:5]
        )
        console_html = (
            f'<div style="margin-top:10px;">'
            f'<strong>Console errors</strong>'
            f'<ul style="margin:6px 0 0 16px;padding:0;color:#dc2626;">{errs}</ul>'
            f"</div>"
        )

    detected_label = {
        "heuristic": "Heuristic",
        "llm":       "LLM",
        "both":      "Heuristic + LLM",
    }.get(f.detected_by.value, f.detected_by.value)

    traj = f'<span style="color:#6b7280;font-size:11px;">trajectory: {f.trajectory_id}</span>' if f.trajectory_id else ""

    return f"""
<details style="border:1px solid #e5e7eb;border-radius:8px;margin-bottom:12px;overflow:hidden;">
  <summary style="padding:14px 16px;cursor:pointer;display:flex;align-items:center;
                  gap:8px;background:#fafafa;list-style:none;user-select:none;">
    <span style="color:#9ca3af;font-size:13px;min-width:24px;">#{index}</span>
    {_badge(f.severity.value, sev_fg, sev_bg)}
    {_badge(f.bug_type.value, typ_fg, typ_bg)}
    <span style="flex:1;font-weight:600;font-size:14px;">{f.title}</span>
    {traj}
    <span style="color:#9ca3af;font-size:11px;">{detected_label}</span>
  </summary>
  <div style="padding:16px;border-top:1px solid #e5e7eb;background:#fff;">
    <p style="margin:0;color:#374151;">{f.reasoning}</p>
    {steps_html}
    {console_html}
    {screenshots_html}
    <div style="margin-top:10px;font-size:11px;color:#9ca3af;">ID: {f.id}</div>
  </div>
</details>"""


def _summary_bar(report: RunReport) -> str:
    counts = {
        "critical": sum(1 for f in report.findings if f.severity.value == "critical"),
        "high":     sum(1 for f in report.findings if f.severity.value == "high"),
        "medium":   sum(1 for f in report.findings if f.severity.value == "medium"),
        "low":      sum(1 for f in report.findings if f.severity.value == "low"),
    }
    pills = ""
    for sev, count in counts.items():
        if count:
            fg, bg = _SEVERITY_COLOR[sev]
            pills += (
                f'<div style="text-align:center;background:{bg};'
                f'border:1px solid {fg}33;border-radius:8px;padding:10px 20px;">'
                f'<div style="font-size:28px;font-weight:700;color:{fg};">{count}</div>'
                f'<div style="font-size:12px;color:{fg};text-transform:uppercase;">{sev}</div>'
                f"</div>"
            )
    return f'<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:24px;">{pills}</div>'


def render_html(report: RunReport, output_dir: Path) -> Path:
    """Write a self-contained report.html with embedded screenshots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(report.findings)
    noise_count = len(report.suppressed_noise)

    # Group findings by severity for the main body.
    order = ["critical", "high", "medium", "low"]
    grouped: dict[str, list[Finding]] = {s: [] for s in order}
    for f in report.findings:
        grouped[f.severity.value].append(f)

    cards_html = ""
    idx = 1
    for sev in order:
        if grouped[sev]:
            fg, bg = _SEVERITY_COLOR[sev]
            cards_html += (
                f'<h3 style="color:{fg};margin:24px 0 8px;text-transform:uppercase;'
                f'font-size:13px;letter-spacing:0.05em;">{sev} ({len(grouped[sev])})</h3>'
            )
            for f in grouped[sev]:
                cards_html += _finding_card(f, idx)
                idx += 1

    noise_rows = ""
    for n in report.suppressed_noise:
        noise_rows += (
            f"<tr><td style='padding:6px 12px;color:#374151;'>{n.description}</td>"
            f"<td style='padding:6px 12px;color:#6b7280;'>{n.reason}</td></tr>"
        )
    noise_section = ""
    if noise_rows:
        noise_section = f"""
<h2 style="margin:32px 0 12px;">Suppressed Noise <span style="font-size:14px;color:#6b7280;">({noise_count})</span></h2>
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <thead>
    <tr style="background:#f3f4f6;">
      <th style="padding:8px 12px;text-align:left;color:#374151;">Description</th>
      <th style="padding:8px 12px;text-align:left;color:#374151;">Reason suppressed</th>
    </tr>
  </thead>
  <tbody>{noise_rows}</tbody>
</table>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UI Test Report — {report.run_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f9fafb; color: #111827; margin: 0; padding: 0;
    }}
    .header {{
      background: #1e293b; color: #f8fafc;
      padding: 24px 40px; border-bottom: 3px solid #3b82f6;
    }}
    .header h1 {{ margin: 0 0 4px; font-size: 22px; }}
    .header p  {{ margin: 0; color: #94a3b8; font-size: 13px; }}
    .container {{ max-width: 960px; margin: 32px auto; padding: 0 24px 64px; }}
    details > summary::-webkit-details-marker {{ display: none; }}
  </style>
</head>
<body>

<div class="header">
  <h1>UI Test Report</h1>
  <p>
    Run: <strong>{report.run_id}</strong> &nbsp;|&nbsp;
    App: <strong>{report.app_url}</strong> &nbsp;|&nbsp;
    {report.timestamp} &nbsp;|&nbsp;
    {report.trajectories_explored} trajectories &nbsp;|&nbsp;
    {round(report.duration_seconds, 1)}s
  </p>
</div>

<div class="container">
  <h2 style="margin:0 0 16px;">
    {total} finding{"s" if total != 1 else ""} found
  </h2>

  {_summary_bar(report)}

  {cards_html if cards_html else '<p style="color:#6b7280;">No findings detected.</p>'}

  {noise_section}
</div>

</body>
</html>"""

    html_path = output_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path
