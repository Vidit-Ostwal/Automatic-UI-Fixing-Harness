"""
Merger — combines findings from N parallel executor collectors into one
deduplicated list.

Deduplication fingerprint: (normalised title, bug_type, severity).
When two findings share a fingerprint, the one with more evidence
(screenshots, steps, console errors) is kept.
"""

import re
from models import Finding, SuppressedNoise
from reporter.collector import FindingCollector


def _fingerprint(f: Finding) -> str:
    """
    Stable identity key for a finding.
    Strips trailing UUIDs from auto-generated titles and normalises whitespace.
    """
    title = re.sub(r"\b[0-9a-f]{6}\b", "", f.title).strip().lower()
    title = re.sub(r"\s+", " ", title)
    return f"{title}|{f.bug_type.value}|{f.severity.value}"


def _evidence_score(f: Finding) -> int:
    """Higher = more evidence. Used to pick the best duplicate."""
    score = 0
    if f.screenshot_before: score += 2
    if f.screenshot_after:  score += 2
    score += len(f.steps)
    score += len(f.console_errors)
    score += len(f.network_errors)
    if f.reasoning:         score += 1
    return score


def deduplicate(findings: list[Finding]) -> list[Finding]:
    """
    Collapse findings with the same fingerprint, keeping the one with
    the most evidence.  Deterministic: stable sort before dedup.
    """
    seen: dict[str, Finding] = {}
    for f in findings:
        key = _fingerprint(f)
        if key not in seen or _evidence_score(f) > _evidence_score(seen[key]):
            seen[key] = f
    # Return in original severity order: critical → high → medium → low.
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(seen.values(), key=lambda f: order.get(f.severity.value, 9))


def merge(collectors: list[FindingCollector]) -> tuple[list[Finding], list[SuppressedNoise]]:
    """
    Merge all collectors into (deduplicated_findings, all_noise).

    Parameters
    ----------
    collectors   One FindingCollector per executor trajectory.

    Returns
    -------
    findings     Deduplicated, severity-sorted list of all findings.
    noise        All suppressed noise items (not deduplicated — each is informative).
    """
    all_findings: list[Finding] = []
    all_noise: list[SuppressedNoise] = []

    for collector in collectors:
        all_findings.extend(collector.findings)
        all_noise.extend(collector.noise)

    return deduplicate(all_findings), all_noise
