"""LLM response parsing helpers shared across planner, executor, verifier, and oracles."""


def strip_code_fence(raw: str) -> str:
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return raw
