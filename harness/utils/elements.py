"""DOM element serialization for LLM prompts."""

import json


def serialize_elements(elements: list[dict], limit: int = 40) -> str:
    """JSON list of selector/role/label/type for LLM element lists."""
    return json.dumps(
        [
            {
                "selector": e.get("selector", ""),
                "role": e.get("role", ""),
                "label": e.get("label", ""),
                "type": e.get("type", ""),
            }
            for e in elements[:limit]
            if e.get("selector")
        ],
        indent=2,
    )
