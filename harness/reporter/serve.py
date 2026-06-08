"""
Serve the HTML report and open it in the user's browser.
"""

from __future__ import annotations

import http.server
import logging
import socket
import webbrowser
from functools import partial
from pathlib import Path

from harness.reporter.collector import HarnessReport, load_report
from harness.reporter.render import render_html

logger = logging.getLogger(__name__)


def _free_port(preferred: int = 8765) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


def open_report(
    output_dir: Path,
    *,
    port: int = 8765,
    open_browser: bool = True,
    block: bool = True,
) -> tuple[HarnessReport, Path, int]:
    """
    Load verifier claims, render report.html, serve output_dir, open browser.

    Returns (report, html_path, port).  When block=True, runs until Ctrl+C.
    """
    output_dir = output_dir.resolve()
    report = load_report(output_dir)
    html_path = render_html(report, output_dir)

    port = _free_port(port)
    handler = partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(output_dir),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)

    url = f"http://127.0.0.1:{port}/report.html"
    logger.info("Report: %d finding(s) from %d run(s)", report.total_findings, report.total_runs)
    logger.info("HTML written to %s", html_path)
    logger.info("Serving at %s", url)

    if open_browser:
        webbrowser.open(url)

    print("\n" + "=" * 60)
    print("  Harness Verification Report")
    print(f"  Runs     : {report.total_runs}")
    print(f"  Findings : {report.total_findings}")
    counts = report.severity_counts
    print(f"    Critical : {counts['critical']}")
    print(f"    High     : {counts['high']}")
    print(f"    Medium   : {counts['medium']}")
    print(f"    Low      : {counts['low']}")
    print(f"  URL      : {url}")
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60 + "\n")

    if block:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down report server")
            server.shutdown()

    return report, html_path, port
