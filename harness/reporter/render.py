"""
Render verifier findings as a self-contained HTML report.
Screenshots are referenced by relative path and served by the local HTTP server.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from harness.reporter.collector import HarnessReport, ReportFinding


def render_html(report: HarnessReport, output_dir: Path) -> Path:
    """Write report.html to output_dir and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "report.html"
    out_path.write_text(_build_page(report), encoding="utf-8")
    return out_path


def _build_page(report: HarnessReport) -> str:
    counts = report.severity_counts
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    findings_html = "\n".join(_finding_card(f) for f in report.findings)
    runs_html = "\n".join(_run_row(r) for r in report.runs)

    if not report.findings:
        findings_html = """
        <section class="empty-state" aria-live="polite">
          <div class="empty-icon" aria-hidden="true">✓</div>
          <h2>No issues found</h2>
          <p>All verified test runs completed without reported findings.</p>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Harness Verification Report</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --surface: #ffffff;
      --surface-muted: #f8fafc;
      --border: #e2e8f0;
      --text: #1e293b;
      --text-muted: #64748b;
      --primary: #4f8cff;
      --primary-soft: #e8f1ff;
      --critical: #dc2626;
      --critical-bg: #fef2f2;
      --high: #ea580c;
      --high-bg: #fff7ed;
      --medium: #ca8a04;
      --medium-bg: #fefce8;
      --low: #2563eb;
      --low-bg: #eff6ff;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.04);
      --radius: 14px;
      --radius-sm: 8px;
    }}

    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
      color: var(--text);
      line-height: 1.55;
      min-height: 100vh;
    }}

    .skip-link {{
      position: absolute;
      left: -9999px;
      top: 0;
      background: var(--primary);
      color: white;
      padding: 8px 16px;
      z-index: 100;
      border-radius: var(--radius-sm);
    }}
    .skip-link:focus {{ left: 16px; top: 16px; }}

    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 28px 24px;
      box-shadow: var(--shadow);
    }}
    .header-inner {{
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      gap: 16px 32px;
      align-items: flex-end;
      justify-content: space-between;
    }}
    .brand h1 {{
      margin: 0 0 6px;
      font-size: clamp(1.4rem, 2.5vw, 1.9rem);
      font-weight: 700;
      letter-spacing: -0.02em;
    }}
    .brand p {{
      margin: 0;
      color: var(--text-muted);
      font-size: 0.95rem;
    }}
    .meta {{
      text-align: right;
      color: var(--text-muted);
      font-size: 0.85rem;
    }}

    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}

    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .stat {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .stat-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .stat-value {{
      font-size: 1.75rem;
      font-weight: 700;
      line-height: 1.1;
    }}
    .stat-value.critical {{ color: var(--critical); }}
    .stat-value.high {{ color: var(--high); }}
    .stat-value.medium {{ color: var(--medium); }}
    .stat-value.low {{ color: var(--low); }}

    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      margin-bottom: 20px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px 16px;
      box-shadow: var(--shadow);
    }}
    .toolbar label {{
      font-size: 0.85rem;
      color: var(--text-muted);
      font-weight: 600;
    }}
    .search {{
      flex: 1 1 220px;
      min-width: 180px;
      padding: 10px 14px;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-size: 0.95rem;
      background: var(--surface-muted);
      color: var(--text);
    }}
    .search:focus {{
      outline: 2px solid var(--primary);
      outline-offset: 1px;
      border-color: var(--primary);
      background: var(--surface);
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .filter-btn {{
      border: 1px solid var(--border);
      background: var(--surface-muted);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s, border-color 0.15s, color 0.15s;
    }}
    .filter-btn:hover {{ background: var(--primary-soft); border-color: #c7d9ff; }}
    .filter-btn:focus-visible {{
      outline: 2px solid var(--primary);
      outline-offset: 2px;
    }}
    .filter-btn.active {{
      background: var(--primary);
      border-color: var(--primary);
      color: white;
    }}

    .layout {{
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 20px;
      align-items: start;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ order: 2; }}
    }}

    .sidebar {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow);
      position: sticky;
      top: 16px;
    }}
    .sidebar h2 {{
      margin: 0 0 12px;
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .run-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      max-height: 60vh;
      overflow-y: auto;
    }}
    .run-item {{
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 10px 12px;
      margin-bottom: 8px;
      background: var(--surface-muted);
      font-size: 0.85rem;
    }}
    .run-item strong {{ display: block; margin-bottom: 2px; }}
    .run-item span {{ color: var(--text-muted); }}
    .run-item.has-issues {{ border-color: #fecaca; background: #fffafa; }}

    .findings {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .finding {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .finding.hidden {{ display: none; }}

    .finding-header {{
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      align-items: flex-start;
      justify-content: space-between;
    }}
    .finding-title {{
      margin: 0;
      font-size: 1.1rem;
      font-weight: 700;
      flex: 1 1 240px;
    }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .badge.severity-critical {{ background: var(--critical-bg); color: var(--critical); }}
    .badge.severity-high {{ background: var(--high-bg); color: var(--high); }}
    .badge.severity-medium {{ background: var(--medium-bg); color: var(--medium); }}
    .badge.severity-low {{ background: var(--low-bg); color: var(--low); }}
    .badge.type {{ background: var(--primary-soft); color: #1d4ed8; }}

    .finding-body {{ padding: 16px 20px 20px; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px 16px;
      margin-bottom: 16px;
      font-size: 0.88rem;
    }}
    .meta-item strong {{
      display: block;
      color: var(--text-muted);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 2px;
    }}

    .prose {{
      margin-bottom: 14px;
      color: var(--text);
      font-size: 0.95rem;
    }}
    .prose h3 {{
      margin: 0 0 6px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .prose p {{ margin: 0; }}

    .steps {{
      margin: 0;
      padding-left: 1.2rem;
      font-size: 0.92rem;
    }}
    .steps li {{ margin-bottom: 4px; }}

    .screenshots {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }}
    @media (max-width: 640px) {{
      .screenshots {{ grid-template-columns: 1fr; }}
    }}
    .shot {{
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      overflow: hidden;
      background: var(--surface-muted);
    }}
    .shot figcaption {{
      padding: 8px 10px;
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      background: var(--surface);
    }}
    .shot img {{
      display: block;
      width: 100%;
      height: auto;
      background: #e2e8f0;
    }}

    .empty-state {{
      text-align: center;
      padding: 48px 24px;
      background: var(--surface);
      border: 1px dashed var(--border);
      border-radius: var(--radius);
    }}
    .empty-icon {{
      width: 56px;
      height: 56px;
      margin: 0 auto 12px;
      border-radius: 50%;
      background: #ecfdf5;
      color: #059669;
      font-size: 1.6rem;
      line-height: 56px;
    }}
    .empty-state h2 {{ margin: 0 0 8px; }}
    .empty-state p {{ margin: 0; color: var(--text-muted); }}

    footer {{
      max-width: 1200px;
      margin: 32px auto 0;
      padding: 0 24px 32px;
      color: var(--text-muted);
      font-size: 0.82rem;
      text-align: center;
    }}
  </style>
</head>
<body>
  <a class="skip-link" href="#findings">Skip to findings</a>

  <header>
    <div class="header-inner">
      <div class="brand">
        <h1>Harness Verification Report</h1>
        <p>Findings from verifier claims across all test runs</p>
      </div>
      <div class="meta">
        <div>{html.escape(report.output_dir)}</div>
        <div>Generated {generated}</div>
      </div>
    </div>
  </header>

  <main>
    <section class="stats" aria-label="Summary statistics">
      <div class="stat">
        <div class="stat-label">Test runs</div>
        <div class="stat-value">{report.total_runs}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Clean runs</div>
        <div class="stat-value">{report.clean_runs}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Total findings</div>
        <div class="stat-value">{report.total_findings}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Critical</div>
        <div class="stat-value critical">{counts["critical"]}</div>
      </div>
      <div class="stat">
        <div class="stat-label">High</div>
        <div class="stat-value high">{counts["high"]}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Medium</div>
        <div class="stat-value medium">{counts["medium"]}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Low</div>
        <div class="stat-value low">{counts["low"]}</div>
      </div>
    </section>

    <div class="toolbar" role="search">
      <label for="search">Search</label>
      <input id="search" class="search" type="search"
             placeholder="Search by title, test case, goal…"
             aria-label="Search findings">
      <div class="filters" role="group" aria-label="Filter by severity">
        <button class="filter-btn active" data-filter="all" type="button">All</button>
        <button class="filter-btn" data-filter="critical" type="button">Critical</button>
        <button class="filter-btn" data-filter="high" type="button">High</button>
        <button class="filter-btn" data-filter="medium" type="button">Medium</button>
        <button class="filter-btn" data-filter="low" type="button">Low</button>
      </div>
    </div>

    <div class="layout">
      <aside class="sidebar" aria-label="Test runs">
        <h2>Test runs ({report.total_runs})</h2>
        <ul class="run-list">
          {runs_html}
        </ul>
      </aside>

      <section id="findings" class="findings" aria-label="Findings">
        {findings_html}
      </section>
    </div>
  </main>

  <footer>
    Automatic UI Fixing Harness · Verifier report viewer
  </footer>

  <script>
    const cards = Array.from(document.querySelectorAll(".finding"));
    const search = document.getElementById("search");
    const buttons = Array.from(document.querySelectorAll(".filter-btn"));
    let activeFilter = "all";

    function applyFilters() {{
      const q = (search.value || "").trim().toLowerCase();
      cards.forEach(card => {{
        const sev = card.dataset.severity || "";
        const text = (card.dataset.search || "").toLowerCase();
        const sevMatch = activeFilter === "all" || sev === activeFilter;
        const textMatch = !q || text.includes(q);
        card.classList.toggle("hidden", !(sevMatch && textMatch));
      }});
    }}

    search.addEventListener("input", applyFilters);
    buttons.forEach(btn => {{
      btn.addEventListener("click", () => {{
        buttons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        activeFilter = btn.dataset.filter;
        applyFilters();
      }});
    }});
  </script>
</body>
</html>"""


def _run_row(run) -> str:
    issue_class = "has-issues" if run.total_findings > 0 else ""
    findings_label = (
        f"{run.total_findings} finding(s)"
        if run.total_findings
        else "Clean"
    )
    return f"""
          <li class="run-item {issue_class}">
            <strong>{html.escape(run.test_case_id)}</strong>
            <span>{html.escape(run.description or run.goal[:60])}</span><br>
            <span>{run.total_steps_verified} steps · {html.escape(findings_label)}</span>
          </li>"""


def _finding_card(f: ReportFinding) -> str:
    search_blob = " ".join([
        f.id, f.test_case_id, f.run_id, f.title, f.goal,
        f.trajectory, f.instruction, f.description, f.evidence,
        f.bug_type, f.severity,
    ])
    steps_html = "\n".join(
        f"<li>{html.escape(s)}</li>" for s in f.reproduction_steps
    ) or "<li>No reproduction steps recorded</li>"

    before = _shot(f.screenshot_before, "Before")
    after = _shot(f.screenshot_after, "After")

    return f"""
        <article class="finding" id="{html.escape(f.id)}"
                 data-severity="{html.escape(f.severity)}"
                 data-search="{html.escape(search_blob)}">
          <div class="finding-header">
            <h2 class="finding-title">{html.escape(f.title)}</h2>
            <div class="badges">
              <span class="badge severity-{html.escape(f.severity)}">{html.escape(f.severity)}</span>
              <span class="badge type">{html.escape(f.bug_type)}</span>
            </div>
          </div>
          <div class="finding-body">
            <div class="meta-grid">
              <div class="meta-item">
                <strong>Test case</strong>
                {html.escape(f.test_case_id)} / {html.escape(f.run_id)}
              </div>
              <div class="meta-item">
                <strong>Step</strong>
                {f.step_index}
              </div>
              <div class="meta-item">
                <strong>Trajectory</strong>
                {html.escape(f.trajectory)}
              </div>
              <div class="meta-item">
                <strong>Instruction</strong>
                {html.escape(f.instruction)}
              </div>
            </div>
            <div class="prose">
              <h3>Goal</h3>
              <p>{html.escape(f.goal)}</p>
            </div>
            <div class="prose">
              <h3>Description</h3>
              <p>{html.escape(f.description)}</p>
            </div>
            <div class="prose">
              <h3>Evidence</h3>
              <p>{html.escape(f.evidence)}</p>
            </div>
            <div class="prose">
              <h3>Reproduction steps</h3>
              <ol class="steps">{steps_html}</ol>
            </div>
            <div class="screenshots">
              {before}
              {after}
            </div>
          </div>
        </article>"""


def _shot(path: str, label: str) -> str:
    if not path:
        return f"""
              <figure class="shot">
                <figcaption>{html.escape(label)}</figcaption>
                <p style="padding:16px;color:#64748b;margin:0;">No screenshot available</p>
              </figure>"""
    src = html.escape(path)
    alt = html.escape(f"{label} screenshot")
    return f"""
              <figure class="shot">
                <figcaption>{html.escape(label)}</figcaption>
                <img src="{src}" alt="{alt}" loading="lazy">
              </figure>"""
