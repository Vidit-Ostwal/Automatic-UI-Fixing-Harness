"""
Render verifier findings as a self-contained HTML report.
Screenshots are referenced by relative path and served by the local HTTP server.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

from harness.reporter.collector import HarnessReport, ReportFinding, TestRunReport


def render_html(report: HarnessReport, output_dir: Path) -> Path:
    """Write report.html to output_dir and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "report.html"
    out_path.write_text(_build_page(report), encoding="utf-8")
    return out_path


# ─── severity helpers ────────────────────────────────────────────────────────

_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _top_severity(run: TestRunReport) -> str | None:
    if not run.findings:
        return None
    return min(run.findings, key=lambda f: _SEV_ORDER.get(f.severity, 4)).severity


# ─── page builder ────────────────────────────────────────────────────────────

def _build_page(report: HarnessReport) -> str:
    counts = report.severity_counts
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if report.runs:
        pages_html = "\n".join(_run_page(i, run) for i, run in enumerate(report.runs))
    else:
        pages_html = """
        <div class="run-page active" id="empty-run">
          <div class="empty-state">
            <div class="empty-glyph">◎</div>
            <p class="empty-title">No test runs found</p>
            <p class="empty-body">Run the harness verifier to populate verifier_claims.</p>
          </div>
        </div>"""

    total = report.total_runs
    clean = report.clean_runs

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Harness Report</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    /* ── tokens ─────────────────────────────────── */
    :root {{
      --ink:          #0d1117;
      --ink-2:        #424a53;
      --ink-3:        #7d8590;
      --surface:      #ffffff;
      --surface-2:    #f6f8fa;
      --surface-3:    #eaeef2;
      --border:       #d0d7de;
      --border-sub:   #e8ecef;
      --blue:         #0969da;
      --blue-bg:      #ddf4ff;
      --red:          #cf222e;
      --red-bg:       #ffebe9;
      --orange:       #bc4c00;
      --orange-bg:    #fff1e5;
      --amber:        #9a6700;
      --amber-bg:     #fff8c5;
      --green:        #1a7f37;
      --green-bg:     #dafbe1;
      --frame-bg:     #161b22;
      --divider-clr:  #30363d;
      --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
      --sans: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
      --r:   6px;
      --r-sm: 4px;
    }}

    /* ── reset ──────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{ height: 100%; margin: 0; padding: 0; }}
    body {{
      font-family: var(--sans);
      background: var(--surface-2);
      color: var(--ink);
      font-size: 13px;
      line-height: 1.5;
      display: flex;
      flex-direction: column;
      height: 100dvh;
      overflow: hidden;
    }}

    /* ── skip nav ───────────────────────────────── */
    .skip {{
      position: absolute; left: -9999px; top: 8px;
      background: var(--ink); color: #fff;
      padding: 6px 12px; border-radius: var(--r-sm);
      font-size: 12px; font-weight: 600; z-index: 500;
      text-decoration: none;
    }}
    .skip:focus {{ left: 8px; }}

    /* ── topbar ─────────────────────────────────── */
    .topbar {{
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 20px;
      height: 44px;
      background: var(--ink);
      flex-shrink: 0;
    }}
    .topbar-brand {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: #fff;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      flex-shrink: 0;
    }}
    .topbar-brand-dot {{
      width: 6px; height: 6px;
      border-radius: 50%;
      background: #3fb950;
    }}
    .topbar-sep {{
      width: 1px; height: 18px;
      background: rgba(255,255,255,.15);
      flex-shrink: 0;
    }}
    .topbar-stats {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex: 1;
    }}
    .tstat {{
      display: flex;
      align-items: center;
      gap: 5px;
      font-size: 11px;
      color: rgba(255,255,255,.5);
      font-family: var(--mono);
    }}
    .tstat-v {{
      color: #e6edf3;
      font-weight: 500;
    }}
    .tstat-v.red   {{ color: #f85149; }}
    .tstat-v.orange{{ color: #ffa657; }}
    .tstat-v.amber {{ color: #e3b341; }}
    .tstat-v.blue  {{ color: #79c0ff; }}
    .topbar-meta {{
      margin-left: auto;
      font-size: 10.5px;
      color: rgba(255,255,255,.35);
      font-family: var(--mono);
      flex-shrink: 0;
    }}

    /* ── controls bar ───────────────────────────── */
    .controls {{
      display: flex;
      align-items: center;
      gap: 0;
      height: 38px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 16px;
      flex-shrink: 0;
    }}
    .pager {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }}
    .pager-btn {{
      display: flex; align-items: center; justify-content: center;
      width: 26px; height: 26px;
      border: 1px solid var(--border);
      border-radius: var(--r-sm);
      background: var(--surface);
      color: var(--ink-2);
      font-size: 13px;
      cursor: pointer;
      transition: background 0.1s, border-color 0.1s;
      line-height: 1;
    }}
    .pager-btn:hover:not(:disabled) {{
      background: var(--surface-2);
      border-color: var(--ink-3);
      color: var(--ink);
    }}
    .pager-btn:focus-visible {{
      outline: 2px solid var(--blue);
      outline-offset: 1px;
    }}
    .pager-btn:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    .pager-label {{
      font-family: var(--mono);
      font-size: 11px;
      color: var(--ink-2);
      min-width: 150px;
      text-align: center;
      white-space: nowrap;
    }}
    .pager-label strong {{
      color: var(--ink);
      font-weight: 600;
    }}
    .pager-hint {{
      font-size: 10px;
      color: var(--ink-3);
      margin-left: 4px;
    }}
    .controls-divider {{
      width: 1px; height: 20px;
      background: var(--border-sub);
      margin: 0 14px;
      flex-shrink: 0;
    }}
    .filters {{
      display: flex;
      align-items: center;
      gap: 4px;
    }}
    .filter-btn {{
      height: 22px;
      padding: 0 9px;
      border: 1px solid transparent;
      border-radius: 99px;
      background: transparent;
      font-size: 11px;
      font-weight: 500;
      color: var(--ink-2);
      cursor: pointer;
      transition: background 0.1s, border-color 0.1s, color 0.1s;
      line-height: 1;
    }}
    .filter-btn:hover {{ background: var(--surface-2); color: var(--ink); }}
    .filter-btn:focus-visible {{ outline: 2px solid var(--blue); outline-offset: 1px; }}
    .filter-btn.active {{
      background: var(--surface-2);
      border-color: var(--border);
      color: var(--ink);
      font-weight: 600;
    }}

    /* ── stage ──────────────────────────────────── */
    .stage {{
      flex: 1;
      min-height: 0;
      position: relative;
    }}

    /* ── run page ───────────────────────────────── */
    .run-page {{
      display: none;
      flex-direction: column;
      position: absolute;
      inset: 0;
    }}
    .run-page.active {{
      display: flex;
    }}

    /* ── run header strip ───────────────────────── */
    .run-bar {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 16px;
      height: 36px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      flex-shrink: 0;
      overflow: hidden;
    }}
    .run-id {{
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 500;
      color: var(--ink);
      white-space: nowrap;
      flex-shrink: 0;
    }}
    .run-slash {{
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
    }}
    .run-trajectory {{
      font-family: var(--mono);
      font-size: 11px;
      color: var(--ink-3);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 1 1 0;
      min-width: 0;
    }}
    .run-goal-text {{
      font-size: 12px;
      color: var(--ink-2);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      flex: 2 1 0;
      min-width: 0;
    }}
    .run-badge {{
      flex-shrink: 0;
    }}

    /* ── compare zone ───────────────────────────── */
    .compare {{
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1px 1fr;
      min-height: 0;
      background: var(--frame-bg);
    }}
    @media (max-width: 800px) {{
      .compare {{ grid-template-columns: 1fr; }}
      .col-divider {{ display: none; }}
    }}
    .col-divider {{
      background: var(--divider-clr);
    }}
    .shot {{
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .shot-cap {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 0 14px;
      height: 28px;
      background: #21262d;
      border-bottom: 1px solid var(--divider-clr);
      flex-shrink: 0;
    }}
    .shot-cap-label {{
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #8b949e;
    }}
    .shot-cap-step {{
      font-family: var(--mono);
      font-size: 10px;
      color: #6e7681;
    }}
    .shot-frame {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      overflow: auto;
      background: var(--frame-bg);
    }}
    .shot-frame img {{
      display: block;
      max-width: 100%;
      max-height: 100%;
      width: auto;
      height: auto;
      object-fit: contain;
      border-radius: 3px;
      cursor: zoom-in;
    }}
    .shot-frame img[src=""] {{ display: none; }}
    .shot-none {{
      font-family: var(--mono);
      font-size: 11px;
      color: #484f58;
      text-align: center;
    }}

    /* ── clean panel (no findings) ──────────────── */
    .clean-compare {{
      flex: 1;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--frame-bg);
      min-height: 0;
    }}
    .clean-inner {{
      text-align: center;
    }}
    .clean-mark {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 40px; height: 40px;
      border-radius: 50%;
      background: rgba(26,127,55,.15);
      color: #3fb950;
      font-size: 18px;
      margin-bottom: 12px;
    }}
    .clean-title {{
      font-size: 13px;
      font-weight: 600;
      color: #c9d1d9;
      margin: 0 0 4px;
    }}
    .clean-sub {{
      font-family: var(--mono);
      font-size: 11px;
      color: #6e7681;
    }}

    /* ── conviction panel ───────────────────────── */
    .conviction {{
      flex-shrink: 0;
      background: var(--surface);
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }}
    .conviction-head {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 0 16px;
      height: 34px;
      border-bottom: 1px solid var(--border-sub);
      flex-shrink: 0;
    }}
    .conviction-section-label {{
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--ink-3);
    }}
    .finding-tabs {{
      display: flex;
      gap: 4px;
      overflow-x: auto;
      flex: 1;
      scrollbar-width: none;
    }}
    .finding-tabs::-webkit-scrollbar {{ display: none; }}
    .finding-tab {{
      flex-shrink: 0;
      height: 20px;
      padding: 0 8px;
      border: 1px solid transparent;
      border-radius: 99px;
      background: transparent;
      font-size: 11px;
      font-weight: 500;
      color: var(--ink-2);
      cursor: pointer;
      transition: background 0.1s, border-color 0.1s;
      white-space: nowrap;
    }}
    .finding-tab:hover {{ background: var(--surface-2); }}
    .finding-tab.active {{
      background: var(--surface-2);
      border-color: var(--border);
      color: var(--ink);
      font-weight: 600;
    }}
    .finding-tab:focus-visible {{ outline: 2px solid var(--blue); outline-offset: 1px; }}

    .conviction-body {{
      padding: 10px 16px;
      overflow-y: auto;
      max-height: 210px;
    }}
    .finding-detail {{ display: none; }}
    .finding-detail.active {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px 20px; }}

    @media (max-width: 900px) {{
      .finding-detail.active {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      .finding-detail.active {{ grid-template-columns: 1fr; }}
    }}

    .fd-field strong {{
      display: block;
      font-size: 10px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--ink-3);
      margin-bottom: 3px;
    }}
    .fd-field p, .fd-field ol {{
      margin: 0;
      font-size: 12px;
      color: var(--ink-2);
      line-height: 1.55;
    }}
    .fd-field ol {{
      padding-left: 1.1rem;
    }}
    .fd-field ol li {{ margin-bottom: 2px; }}
    .fd-title {{
      display: flex;
      align-items: flex-start;
      gap: 8px;
      grid-column: 1 / -1;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border-sub);
      margin-bottom: 2px;
    }}
    .fd-title-text {{
      font-size: 13px;
      font-weight: 600;
      color: var(--ink);
      flex: 1;
      line-height: 1.4;
    }}

    /* ── clean conviction ───────────────────────── */
    .clean-conviction {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      gap: 10px 20px;
    }}

    /* ── badges ─────────────────────────────────── */
    .badge {{
      display: inline-flex;
      align-items: center;
      height: 18px;
      padding: 0 7px;
      border-radius: 99px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      flex-shrink: 0;
    }}
    .badge-critical {{ background: var(--red-bg);    color: var(--red);    }}
    .badge-high     {{ background: var(--orange-bg); color: var(--orange); }}
    .badge-medium   {{ background: var(--amber-bg);  color: var(--amber);  }}
    .badge-low      {{ background: var(--blue-bg);   color: var(--blue);   }}
    .badge-clean    {{ background: var(--green-bg);  color: var(--green);  }}
    .badge-type     {{ background: var(--surface-3); color: var(--ink-2);  }}

    /* ── filter-empty ───────────────────────────── */
    .filter-empty {{
      display: none;
      position: absolute;
      inset: 0;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 8px;
      background: var(--surface-2);
      font-size: 13px;
      color: var(--ink-3);
    }}
    .filter-empty.visible {{ display: flex; }}
    .filter-empty strong {{ color: var(--ink-2); font-weight: 600; }}

    /* ── lightbox ───────────────────────────────── */
    .lightbox {{
      position: fixed; inset: 0;
      background: rgba(1,4,9,.88);
      display: none;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 400;
      cursor: zoom-out;
    }}
    .lightbox.open {{ display: flex; }}
    .lightbox img {{
      max-width: min(96vw, 1400px);
      max-height: 92vh;
      object-fit: contain;
      border-radius: var(--r-sm);
      box-shadow: 0 24px 64px rgba(0,0,0,.6);
    }}
    .lightbox-close {{
      position: absolute;
      top: 16px; right: 20px;
      background: none; border: none;
      color: rgba(255,255,255,.6);
      font-size: 22px;
      cursor: pointer;
      line-height: 1;
      padding: 4px;
    }}
    .lightbox-close:hover {{ color: #fff; }}

    /* ── empty state ────────────────────────────── */
    .empty-state {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 8px;
      background: var(--surface-2);
    }}
    .empty-glyph {{
      font-size: 28px;
      color: var(--ink-3);
      margin-bottom: 4px;
    }}
    .empty-title {{
      font-size: 14px;
      font-weight: 600;
      color: var(--ink-2);
      margin: 0;
    }}
    .empty-body {{
      font-size: 12px;
      color: var(--ink-3);
      margin: 0;
    }}
  </style>
</head>
<body>
  <a class="skip" href="#stage">Skip to run</a>

  <!-- ── topbar ──────────────────────────────────── -->
  <header class="topbar" role="banner">
    <div class="topbar-brand">
      <div class="topbar-brand-dot"></div>
      Harness Report
    </div>
    <div class="topbar-sep"></div>
    <div class="topbar-stats" role="list" aria-label="Summary">
      <div class="tstat" role="listitem"><span class="tstat-v">{total}</span> runs</div>
      <div class="tstat" role="listitem"><span class="tstat-v">{clean}</span> clean</div>
      <div class="tstat" role="listitem"><span class="tstat-v">{report.total_findings}</span> findings</div>
      <div class="tstat" role="listitem"><span class="tstat-v red">{counts["critical"]}</span> critical</div>
      <div class="tstat" role="listitem"><span class="tstat-v orange">{counts["high"]}</span> high</div>
      <div class="tstat" role="listitem"><span class="tstat-v amber">{counts["medium"]}</span> med</div>
      <div class="tstat" role="listitem"><span class="tstat-v blue">{counts["low"]}</span> low</div>
    </div>
    <div class="topbar-meta">{generated}</div>
  </header>

  <!-- ── controls bar ────────────────────────────── -->
  <div class="controls" role="toolbar" aria-label="Navigation and filters">
    <nav class="pager" aria-label="Run navigation">
      <button class="pager-btn" id="prev" type="button" aria-label="Previous run">&#8592;</button>
      <div class="pager-label" id="pager-status" aria-live="polite" aria-atomic="true">
        <strong>—</strong> <span class="pager-hint">&#8592; &#8594; keys</span>
      </div>
      <button class="pager-btn" id="next" type="button" aria-label="Next run">&#8594;</button>
    </nav>
    <div class="controls-divider"></div>
    <div class="filters" role="group" aria-label="Filter by status">
      <button class="filter-btn active" data-filter="all"      type="button">All</button>
      <button class="filter-btn"        data-filter="issues"   type="button">Issues</button>
      <button class="filter-btn"        data-filter="clean"    type="button">Clean</button>
      <button class="filter-btn"        data-filter="critical" type="button">Critical</button>
      <button class="filter-btn"        data-filter="high"     type="button">High</button>
      <button class="filter-btn"        data-filter="medium"   type="button">Medium</button>
      <button class="filter-btn"        data-filter="low"      type="button">Low</button>
    </div>
  </div>

  <!-- ── stage ───────────────────────────────────── -->
  <main class="stage" id="stage">
    <div class="filter-empty" id="filter-empty" role="status">
      <strong>No runs match this filter</strong>
      <span>Select a different filter to continue.</span>
    </div>
    {pages_html}
  </main>

  <!-- ── lightbox ────────────────────────────────── -->
  <div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Enlarged screenshot">
    <button class="lightbox-close" id="lightbox-close" aria-label="Close">&#215;</button>
    <img src="" alt="" id="lightbox-img">
  </div>

  <script>
  (function () {{
    const pages = Array.from(document.querySelectorAll(".run-page"));
    const prevBtn   = document.getElementById("prev");
    const nextBtn   = document.getElementById("next");
    const statusEl  = document.getElementById("pager-status");
    const emptyEl   = document.getElementById("filter-empty");
    const lightbox  = document.getElementById("lightbox");
    const lbImg     = document.getElementById("lightbox-img");
    const lbClose   = document.getElementById("lightbox-close");
    const filterBtns= Array.from(document.querySelectorAll(".filter-btn"));

    let activeFilter = "all";
    let visible = [];
    let cur = 0;
    let syncing = false;

    function matchFilter(p) {{
      const hasIssues  = p.dataset.hasIssues === "true";
      const severities = (p.dataset.severities || "").split(",").filter(Boolean);
      if (activeFilter === "all")    return true;
      if (activeFilter === "issues") return hasIssues;
      if (activeFilter === "clean")  return !hasIssues;
      return severities.includes(activeFilter);
    }}

    function refresh() {{
      visible = pages.filter(p => {{
        const ok = matchFilter(p);
        p.style.display = "none";
        return ok;
      }});
      cur = Math.min(cur, Math.max(0, visible.length - 1));
      show(cur);
    }}

    function show(i) {{
      pages.forEach(p => {{ p.style.display = "none"; p.classList.remove("active"); }});
      if (!visible.length) {{
        emptyEl.classList.add("visible");
        statusEl.innerHTML = "<strong>—</strong>";
        prevBtn.disabled = nextBtn.disabled = true;
        return;
      }}
      emptyEl.classList.remove("visible");
      cur = Math.max(0, Math.min(i, visible.length - 1));
      const p = visible[cur];
      p.style.display = "flex";
      p.classList.add("active");

      const id = p.dataset.label || ("Run " + (cur + 1));
      statusEl.innerHTML =
        "<strong>" + id + "</strong>" +
        " <span style='color:var(--ink-3)'>" + (cur+1) + "/" + visible.length + "</span>" +
        " <span class='pager-hint'>&#8592; &#8594;</span>";
      prevBtn.disabled = cur === 0;
      nextBtn.disabled = cur === visible.length - 1;

      const detail = p.querySelector(".finding-detail.active") || p.querySelector(".finding-detail");
      if (detail) loadScreenshots(p, detail);
      bindScroll(p);
    }}

    function loadScreenshots(page, detail) {{
      const bPath = detail.dataset.before || "";
      const aPath = detail.dataset.after  || "";
      const bImg  = page.querySelector(".shot-before .shot-frame img");
      const aImg  = page.querySelector(".shot-after  .shot-frame img");
      const bNone = page.querySelector(".shot-before .shot-none");
      const aNone = page.querySelector(".shot-after  .shot-none");
      if (bImg) {{ bImg.src = bPath; bImg.style.display = bPath ? "" : "none"; }}
      if (aImg) {{ aImg.src = aPath; aImg.style.display = aPath ? "" : "none"; }}
      if (bNone) bNone.style.display = bPath ? "none" : "";
      if (aNone) aNone.style.display = aPath ? "none" : "";
    }}

    function bindScroll(page) {{
      const bf = page.querySelector(".shot-before .shot-frame");
      const af = page.querySelector(".shot-after  .shot-frame");
      if (!bf || !af) return;
      const sync = (src, tgt) => {{
        if (syncing) return;
        syncing = true;
        const r = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
        tgt.scrollTop = r * (tgt.scrollHeight - tgt.clientHeight);
        syncing = false;
      }};
      bf.onscroll = () => sync(bf, af);
      af.onscroll = () => sync(af, bf);
    }}

    prevBtn.addEventListener("click", () => show(cur - 1));
    nextBtn.addEventListener("click", () => show(cur + 1));

    document.addEventListener("keydown", e => {{
      if (lightbox.classList.contains("open")) {{
        if (e.key === "Escape") closeLightbox();
        return;
      }}
      if (e.target.closest("button.finding-tab, input, select, textarea")) return;
      if (e.key === "ArrowLeft")  {{ e.preventDefault(); show(cur - 1); }}
      if (e.key === "ArrowRight") {{ e.preventDefault(); show(cur + 1); }}
    }});

    filterBtns.forEach(btn => btn.addEventListener("click", () => {{
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.dataset.filter;
      cur = 0;
      refresh();
    }}));

    document.querySelectorAll(".finding-tab").forEach(tab => tab.addEventListener("click", () => {{
      const page = tab.closest(".run-page");
      const idx  = tab.dataset.idx;
      page.querySelectorAll(".finding-tab").forEach(t => {{ t.classList.remove("active"); t.setAttribute("aria-pressed","false"); }});
      page.querySelectorAll(".finding-detail").forEach(d => d.classList.remove("active"));
      tab.classList.add("active"); tab.setAttribute("aria-pressed","true");
      const detail = page.querySelector(".finding-detail[data-idx='" + idx + "']");
      if (detail) {{ detail.classList.add("active"); loadScreenshots(page, detail); }}
    }}));

    document.querySelectorAll(".shot-frame img").forEach(img => img.addEventListener("click", () => {{
      if (!img.src || img.src.endsWith("//")) return;
      lbImg.src = img.src; lbImg.alt = img.alt;
      lightbox.classList.add("open");
    }}));

    function closeLightbox() {{
      lightbox.classList.remove("open");
      lbImg.src = "";
    }}
    lightbox.addEventListener("click", e => {{ if (e.target === lightbox) closeLightbox(); }});
    lbClose.addEventListener("click", closeLightbox);

    refresh();
  }})();
  </script>
</body>
</html>"""


# ─── run page ─────────────────────────────────────────────────────────────────

def _run_page(index: int, run: TestRunReport) -> str:
    has_issues = run.total_findings > 0
    severities = ",".join(sorted({f.severity for f in run.findings}))
    label = html.escape(run.test_case_id)

    status_badge = _status_badge(run)
    compare_section = _compare_section(run)
    conviction_section = _conviction_section(run)

    return f"""
    <div class="run-page"
         data-index="{index}"
         data-label="{label}"
         data-has-issues="{"true" if has_issues else "false"}"
         data-severities="{html.escape(severities)}">
      <div class="run-bar">
        <span class="run-id">{html.escape(run.test_case_id)}</span>
        <span class="run-slash">/</span>
        <span class="run-trajectory">{html.escape(run.description)}</span>
        <span class="run-goal-text" title="{html.escape(run.goal)}">{html.escape(run.goal)}</span>
        <span class="run-badge">{status_badge}</span>
      </div>
      {compare_section}
      {conviction_section}
    </div>"""


def _status_badge(run: TestRunReport) -> str:
    top = _top_severity(run)
    if top is None:
        return '<span class="badge badge-clean">clean</span>'
    n = len(run.findings)
    extra = f" +{n - 1}" if n > 1 else ""
    return f'<span class="badge badge-{html.escape(top)}">{html.escape(top)}{html.escape(extra)}</span>'


# ─── compare ──────────────────────────────────────────────────────────────────

def _compare_section(run: TestRunReport) -> str:
    if not run.findings:
        return f"""
      <div class="clean-compare">
        <div class="clean-inner">
          <div class="clean-mark">&#10003;</div>
          <p class="clean-title">No findings</p>
          <p class="clean-sub">{run.total_steps_verified} steps verified — all clear</p>
        </div>
      </div>"""

    f0 = run.findings[0]
    before_src = html.escape(f0.screenshot_before) if f0.screenshot_before else ""
    after_src  = html.escape(f0.screenshot_after)  if f0.screenshot_after  else ""
    step_label = f"step {f0.step_index}"

    before_img  = f'<img src="{before_src}" alt="Before screenshot" loading="lazy">' if before_src else ""
    after_img   = f'<img src="{after_src}"  alt="After screenshot"  loading="lazy">' if after_src  else ""
    before_none = "" if before_src else '<p class="shot-none">No screenshot</p>'
    after_none  = "" if after_src  else '<p class="shot-none">No screenshot</p>'

    return f"""
      <div class="compare">
        <figure class="shot shot-before">
          <div class="shot-cap">
            <span class="shot-cap-label">Before</span>
            <span class="shot-cap-step">{html.escape(step_label)}</span>
          </div>
          <div class="shot-frame">{before_img}{before_none}</div>
        </figure>
        <div class="col-divider" aria-hidden="true"></div>
        <figure class="shot shot-after">
          <div class="shot-cap">
            <span class="shot-cap-label">After</span>
            <span class="shot-cap-step">{html.escape(step_label)}</span>
          </div>
          <div class="shot-frame">{after_img}{after_none}</div>
        </figure>
      </div>"""


# ─── conviction ───────────────────────────────────────────────────────────────

def _conviction_section(run: TestRunReport) -> str:
    if not run.findings:
        return f"""
      <div class="conviction">
        <div class="conviction-head">
          <span class="conviction-section-label">Conviction</span>
        </div>
        <div class="conviction-body">
          <div class="clean-conviction">
            <div class="fd-field"><strong>Result</strong><p>No defects reported.</p></div>
            <div class="fd-field"><strong>Coverage</strong><p>{run.total_steps_verified} steps verified</p></div>
            <div class="fd-field"><strong>Trajectory</strong><p>{html.escape(run.description)}</p></div>
            <div class="fd-field"><strong>Run ID</strong><p style="font-family:var(--mono);font-size:11px">{html.escape(run.run_id)}</p></div>
          </div>
        </div>
      </div>"""

    tabs_html = ""
    details_html = ""
    for i, f in enumerate(run.findings):
        active = " active" if i == 0 else ""
        pressed = "true" if i == 0 else "false"
        short = html.escape(f.title[:52])
        tabs_html += (
            f'<button class="finding-tab{active}" type="button" '
            f'data-idx="{i}" aria-pressed="{pressed}">{short}</button>'
        )
        details_html += _finding_detail(i, f, active)

    tabs_section = (
        f'<div class="finding-tabs" role="group" aria-label="Findings">{tabs_html}</div>'
        if len(run.findings) > 1 else ""
    )

    return f"""
      <div class="conviction">
        <div class="conviction-head">
          <span class="conviction-section-label">Conviction</span>
          {tabs_section}
        </div>
        <div class="conviction-body">
          {details_html}
        </div>
      </div>"""


def _finding_detail(index: int, f: ReportFinding, active_class: str) -> str:
    steps_html = (
        "\n".join(f"<li>{html.escape(s)}</li>" for s in f.reproduction_steps)
        or "<li>No reproduction steps recorded.</li>"
    )
    return f"""
          <div class="finding-detail{active_class}"
               data-idx="{index}"
               data-before="{html.escape(f.screenshot_before)}"
               data-after="{html.escape(f.screenshot_after)}">
            <div class="fd-title">
              <span class="fd-title-text">{html.escape(f.title)}</span>
              <span class="badge badge-{html.escape(f.severity)}">{html.escape(f.severity)}</span>
              <span class="badge badge-type">{html.escape(f.bug_type)}</span>
            </div>
            <div class="fd-field">
              <strong>Step {f.step_index}</strong>
              <p>{html.escape(f.instruction)}</p>
            </div>
            <div class="fd-field">
              <strong>Description</strong>
              <p>{html.escape(f.description)}</p>
            </div>
            <div class="fd-field">
              <strong>Evidence</strong>
              <p>{html.escape(f.evidence)}</p>
            </div>
            <div class="fd-field">
              <strong>Reproduction</strong>
              <ol>{steps_html}</ol>
            </div>
          </div>"""
