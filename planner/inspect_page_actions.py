"""
Inspect what the planner sees on a page — raw DOM scan vs LLM/action selection.

Usage (from repo root):
    uv run python planner/inspect_page_actions.py

Change URL below. App must be reachable (local server or Docker instance).
Set USE_LLM=False to skip the LLM call and only show dom_group fallback actions.
"""

import asyncio
import json
import sys
from pathlib import Path

# Repo root on sys.path so `browser.*` / `planner.*` / `oracles.*` import cleanly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from browser.session import BrowserSession
from planner.action_identifier import ActionIdentifier, dom_group

# ---------------------------------------------------------------------------
# Config — edit these
# ---------------------------------------------------------------------------
URL = "http://localhost:5230"
USE_LLM = True
HEADLESS = True


def _llm_element_payload(elements: list[dict]) -> list[dict]:
    """Same field subset serialized in LLMOracle.analyze_page_actions()."""
    return [
        {
            "selector": e.get("selector", ""),
            "role": e.get("role", ""),
            "label": e.get("label", ""),
            "type": e.get("type", ""),
        }
        for e in elements
        if e.get("selector")
    ]


def _print_section(title: str, data) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(json.dumps(data, indent=2))


async def main() -> None:
    llm = None
    if USE_LLM:
        try:
            from oracles.llm import LLMOracle
            llm = LLMOracle.from_env()
        except Exception as e:
            print(f"LLM unavailable ({e}); falling back to dom_group only.")

    async with BrowserSession.create(headless=HEADLESS) as session:
        print(f"Navigating to {URL} ...")
        await session.navigate(URL)

        state = await session.capture_state()
        elements = await session.get_interactive_elements()

        # 1) Full DOM scan — output of get_interactive_elements()
        _print_section(
            f"DOM SCAN — get_interactive_elements()  ({len(elements)} element(s))",
            elements,
        )

        # 2) Subset actually sent to the LLM (selector, role, label, type only)
        llm_elements = _llm_element_payload(elements)
        _print_section(
            f"LLM INPUT — interactive elements passed to analyze_page_actions()  "
            f"({len(llm_elements)} element(s))",
            llm_elements,
        )

        # 3) Final workflows BFS would try — ActionIdentifier.identify()
        identifier = ActionIdentifier(llm_client=llm)
        actions = await identifier.identify(
            elements, state.a11y_tree, state.screenshot
        )
        action_payload = [
            {
                "name": a.name,
                "description": a.description,
                "expected_outcome": a.expected_outcome,
                "steps": [
                    {"type": s.type, "selector": s.selector, "value": s.value}
                    for s in a.steps
                ],
            }
            for a in actions
        ]
        _print_section(
            f"SELECTED ACTIONS — ActionIdentifier.identify()  ({len(actions)} action(s))",
            action_payload,
        )

        if not llm:
            dom_only = dom_group(elements)
            _print_section(
                "DOM FALLBACK ONLY — dom_group(elements)",
                [
                    {
                        "name": a.name,
                        "steps": [
                            {"type": s.type, "selector": s.selector}
                            for s in a.steps
                        ],
                    }
                    for a in dom_only
                ],
            )

        print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())
