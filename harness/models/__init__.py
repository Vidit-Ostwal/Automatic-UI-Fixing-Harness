"""Shared domain types used across the harness."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class BugType(str, Enum):
    VISUAL = "visual"
    LOGIC = "logic"


class DetectedBy(str, Enum):
    HEURISTIC = "heuristic"
    LLM = "llm"
    BOTH = "both"


@dataclass
class PageState:
    url: str
    screenshot: bytes
    a11y_tree: dict
    console_errors: list[str] = field(default_factory=list)
    network_failures: list[dict] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class Finding:
    id: str
    title: str
    bug_type: BugType
    severity: Severity
    steps: list[str]
    detected_by: DetectedBy
    reasoning: str
    trajectory_id: str = ""
    screenshot_before: Optional[bytes] = None
    screenshot_after: Optional[bytes] = None
    console_errors: list[str] = field(default_factory=list)
    network_errors: list[dict] = field(default_factory=list)


@dataclass
class SuppressedNoise:
    description: str
    reason: str


__all__ = [
    "BugType",
    "DetectedBy",
    "Finding",
    "PageState",
    "Severity",
    "SuppressedNoise",
]
