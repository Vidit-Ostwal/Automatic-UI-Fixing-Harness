"""
Finding collector — accumulates Finding and SuppressedNoise objects
during a single executor run.

One collector instance lives per executor trajectory. After all executors
finish, their collectors are passed to the merger.
"""

from dataclasses import dataclass, field
from models import Finding, SuppressedNoise


@dataclass
class CollectorStats:
    total: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0
    visual: int = 0
    logic: int = 0
    suppressed: int = 0


class FindingCollector:
    """
    Accumulates findings and suppressed noise for one executor trajectory.

    Methods
    -------
    add(finding, trajectory_id)   Record a Finding; optionally tag with trajectory.
    suppress(description, reason) Record a suppressed noise item.
    findings                      All Finding objects added so far.
    noise                         All SuppressedNoise objects added so far.
    stats                         Quick summary counts.
    """

    def __init__(self, trajectory_id: str = ""):
        self._trajectory_id = trajectory_id
        self._findings: list[Finding] = []
        self._noise: list[SuppressedNoise] = []

    def add(self, finding: Finding, trajectory_id: str = "") -> None:
        """Add a finding, stamping it with trajectory_id if not already set."""
        tid = trajectory_id or self._trajectory_id
        if tid and not finding.trajectory_id:
            finding.trajectory_id = tid
        self._findings.append(finding)

    def suppress(self, description: str, reason: str) -> None:
        """Record a suppressed noise item so the report can explain it."""
        self._noise.append(SuppressedNoise(description=description, reason=reason))

    @property
    def findings(self) -> list[Finding]:
        return list(self._findings)

    @property
    def noise(self) -> list[SuppressedNoise]:
        return list(self._noise)

    def stats(self) -> CollectorStats:
        s = CollectorStats(
            total=len(self._findings),
            suppressed=len(self._noise),
        )
        for f in self._findings:
            sev = f.severity.value
            if sev == "critical": s.critical += 1
            elif sev == "high":   s.high += 1
            elif sev == "medium": s.medium += 1
            elif sev == "low":    s.low += 1

            bt = f.bug_type.value
            if bt == "visual": s.visual += 1
            elif bt == "logic": s.logic += 1
        return s

    def __len__(self) -> int:
        return len(self._findings)
