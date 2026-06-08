"""
Dump the rendered DOM HTML for a page (after JS runs).

Usage (from repo root):
    uv run python planner/inspect_rendered_html.py

Change URL and OUTPUT below. App must be reachable.
Writes rendered HTML to OUTPUT; also prints byte count and path to stdout.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from browser.session import BrowserSession

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
URL = "http://localhost:5230"
OUTPUT = "rendered.html"  # relative to repo root, or use an absolute path
HEADLESS = True
PRINT_TO_STDOUT = False  # True = also dump HTML to terminal (can be huge)


async def main() -> None:
    out_path = Path(OUTPUT)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    async with BrowserSession.create(headless=HEADLESS) as session:
        print(f"Navigating to {URL} ...")
        await session.navigate(URL)
        html = await session.page.content()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Rendered HTML: {len(html):,} bytes → {out_path}")
    if PRINT_TO_STDOUT:
        print("\n" + "=" * 72 + "\n")
        print(html)


if __name__ == "__main__":
    asyncio.run(main())
