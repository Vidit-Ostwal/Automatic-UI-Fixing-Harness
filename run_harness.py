"""
run_harness.py — top-level entry point for the autonomous UI test harness.

Usage:
    uv run python run_harness.py [options]

Two-phase execution:
  Phase 1  PLANNER   — one Docker instance, BFS exploration, emit trajectories.json
  Phase 2  EXECUTORS — N parallel Docker instances, each running one trajectory,
                       oracles firing at every step

Configuration (env vars, all optional):
    DEPTH_N            BFS depth          (default: 3)
    MAX_TRAJECTORIES   executor cap       (default: 20)
    MAX_PARALLEL       concurrent Docker  (default: 4)
    OUTPUT_DIR         report directory   (default: output/)
    ANTHROPIC_API_KEY  } at least one
    OPENAI_API_KEY     } must be set
    LLM_PROVIDER       anthropic | openai (auto-detected)

Flags:
    --skip-planner     Skip phase 1; load trajectories from OUTPUT_DIR/trajectories.json
    --no-llm           Disable LLM oracle (deterministic checks only)
    --depth N          Override DEPTH_N
    --max-trajectories Override MAX_TRAJECTORIES
    --output DIR       Override OUTPUT_DIR
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging — set up before any local imports so all modules inherit the config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("harness")

from browser.session import BrowserSession
from docker_manager import DockerInstance
from executor.runner import RunnerConfig, TrajectoryRunner, steps_from_trajectory
from oracles.diff import DiffOracle
from oracles.logic import LogicOracle
from oracles.visual import VisualOracle
from planner.action_identifier import ActionIdentifier
from planner.bfs_explorer import BFSExplorer
from planner.trajectory_extractor import extract_trajectories, trajectories_to_json
from reporter.collector import FindingCollector
from reporter.merger import merge
from reporter.render import RunReport, render_html, render_json


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class HarnessConfig:
    def __init__(self, args: argparse.Namespace):
        self.depth_n          = args.depth          or _env_int("DEPTH_N", 3)
        self.max_trajectories = args.max_trajectories or _env_int("MAX_TRAJECTORIES", 20)
        self.max_parallel     = _env_int("MAX_PARALLEL", 4)
        self.output_dir       = Path(args.output or os.environ.get("OUTPUT_DIR", "output"))
        self.planner_only     = args.planner_only
        self.skip_planner     = args.skip_planner
        self.no_llm           = args.no_llm
        self.run_id           = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _build_llm_oracle(no_llm: bool):
    if no_llm:
        return None
    try:
        from oracles.llm import LLMOracle
        return LLMOracle.from_env()
    except EnvironmentError as e:
        logger.warning("LLM oracle disabled: %s", e)
        return None


# ---------------------------------------------------------------------------
# Phase 1 — Planner
# ---------------------------------------------------------------------------

async def run_planner(config: HarnessConfig, llm_oracle) -> list[dict]:
    """
    Spin up one Docker instance, BFS-explore the app, return serialised
    trajectories. Also writes trajectories.json to the output dir.
    """
    logger.info("=== PHASE 1: PLANNER (depth=%d) ===", config.depth_n)

    async with DockerInstance.start() as docker:
        logger.info("Planner instance ready at %s", docker.url)

        identifier = ActionIdentifier(llm_client=llm_oracle)

        async with BrowserSession.create() as session:
            explorer = BFSExplorer(
                session=session,
                action_identifier=identifier,
                max_depth=config.depth_n,
                max_actions_per_node=8,
            )
            result = await explorer.explore(docker.url)

    logger.info(
        "Planner done: %d states discovered, %d edges",
        len(result.nodes),
        sum(len(v) for v in result.graph.values()),
    )

    trajectories = extract_trajectories(result.graph, result.root_hash)
    logger.info("Extracted %d trajectories", len(trajectories))

    serialised = trajectories_to_json(trajectories)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = config.output_dir / "trajectories.json"
    traj_path.write_text(json.dumps(serialised, indent=2))
    logger.info("Trajectories saved to %s", traj_path)

    return serialised


# ---------------------------------------------------------------------------
# Phase 2 — single executor
# ---------------------------------------------------------------------------

async def run_one_trajectory(
    trajectory: dict,
    config: HarnessConfig,
    llm_oracle,
    semaphore: asyncio.Semaphore,
) -> FindingCollector:
    """
    Acquire a concurrency slot, spin up a Docker instance, run the trajectory,
    return the collector. Never raises — exceptions produce an empty collector
    with a logged warning.
    """
    async with semaphore:
        traj_id = trajectory.get("id", "T-???")
        logger.info("Executor %s starting", traj_id)
        try:
            async with DockerInstance.start() as docker:
                steps = steps_from_trajectory(trajectory)
                runner = TrajectoryRunner(
                    steps=steps,
                    config=RunnerConfig(
                        trajectory_id=traj_id,
                        app_url=docker.url,
                    ),
                    visual_oracle=VisualOracle(llm_oracle=llm_oracle),
                    logic_oracle=LogicOracle(),
                    diff_oracle=DiffOracle(llm_oracle=llm_oracle),
                    llm_oracle=llm_oracle,
                )
                collector = await runner.run()
                logger.info(
                    "Executor %s finished: %d finding(s)", traj_id, len(collector)
                )
                return collector
        except Exception as e:
            logger.warning("Executor %s failed: %s", traj_id, e)
            return FindingCollector(trajectory_id=traj_id)


# ---------------------------------------------------------------------------
# Phase 2 — all executors in parallel
# ---------------------------------------------------------------------------

async def run_executors(
    trajectories: list[dict],
    config: HarnessConfig,
    llm_oracle,
) -> list[FindingCollector]:
    capped = trajectories[: config.max_trajectories]
    logger.info(
        "=== PHASE 2: EXECUTORS (%d trajectories, max %d parallel) ===",
        len(capped), config.max_parallel,
    )

    semaphore = asyncio.Semaphore(config.max_parallel)
    tasks = [
        run_one_trajectory(t, config, llm_oracle, semaphore)
        for t in capped
    ]
    collectors = await asyncio.gather(*tasks)
    return list(collectors)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    collectors: list[FindingCollector],
    config: HarnessConfig,
    trajectories: list[dict],
    duration: float,
) -> None:
    findings, noise = merge(collectors)

    report = RunReport(
        run_id=config.run_id,
        app_url="http://localhost (see trajectories.json for per-instance URLs)",
        findings=findings,
        suppressed_noise=noise,
        trajectories_explored=len(trajectories),
        duration_seconds=duration,
    )

    json_path = render_json(report, config.output_dir)
    html_path = render_html(report, config.output_dir)

    logger.info("Report written:")
    logger.info("  JSON → %s", json_path)
    logger.info("  HTML → %s", html_path)

    # Print summary to stdout.
    total    = len(findings)
    critical = sum(1 for f in findings if f.severity.value == "critical")
    high     = sum(1 for f in findings if f.severity.value == "high")
    medium   = sum(1 for f in findings if f.severity.value == "medium")
    low      = sum(1 for f in findings if f.severity.value == "low")

    print("\n" + "=" * 60)
    print(f"  Run ID : {config.run_id}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Trajectories explored: {len(trajectories)}")
    print(f"  Findings: {total} total")
    print(f"    Critical : {critical}")
    print(f"    High     : {high}")
    print(f"    Medium   : {medium}")
    print(f"    Low      : {low}")
    print(f"  Suppressed noise: {len(noise)}")
    print(f"  Report: {html_path}")
    print("=" * 60 + "\n")

    return critical, high


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_trajectories(trajectories: list[dict]) -> None:
    """Pretty-print discovered trajectories to stdout."""
    print(f"\n{'='*60}")
    print(f"  BFS exploration complete — {len(trajectories)} trajectory/trajectories found")
    print(f"{'='*60}")
    for i, t in enumerate(trajectories, 1):
        steps = t.get("steps", [])
        desc  = t.get("description", t.get("id", ""))
        print(f"\n  [{i:02d}] {t.get('id', '')}  ({len(steps)} step(s))")
        if desc:
            print(f"       {desc}")
        for j, s in enumerate(steps, 1):
            action   = s.get("action", "?")
            selector = s.get("selector", "")
            sel_hint = f"  →  {selector}" if selector else ""
            print(f"         {j}. {action}{sel_hint}")
    print(f"\n  trajectories.json written to output dir\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autonomous UI test harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--planner-only", action="store_true",
                   help="Run BFS exploration only; write trajectories.json and exit")
    p.add_argument("--skip-planner", action="store_true",
                   help="Skip BFS exploration; load trajectories.json from output dir")
    p.add_argument("--no-llm",       action="store_true",
                   help="Disable LLM oracle (deterministic checks only)")
    p.add_argument("--depth",        type=int, default=None,
                   help="BFS depth override (env: DEPTH_N)")
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Executor cap override (env: MAX_TRAJECTORIES)")
    p.add_argument("--output",       default=None,
                   help="Output directory override (env: OUTPUT_DIR)")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    config = HarnessConfig(args)

    logger.info("Harness run %s | depth=%d | max_traj=%d | max_parallel=%d",
                config.run_id, config.depth_n, config.max_trajectories, config.max_parallel)

    llm_oracle = _build_llm_oracle(config.no_llm)

    start = time.monotonic()

    # Phase 1 — Planner.
    if config.skip_planner:
        traj_path = config.output_dir / "trajectories.json"
        if not traj_path.exists():
            logger.error("--skip-planner set but %s not found", traj_path)
            return 2
        trajectories = json.loads(traj_path.read_text())
        logger.info("Loaded %d trajectories from %s", len(trajectories), traj_path)
    else:
        trajectories = await run_planner(config, llm_oracle)

    if not trajectories:
        logger.warning("No trajectories to execute — check BFS depth and app URL")
        trajectories = []

    # Early exit when only BFS exploration was requested.
    if config.planner_only:
        _print_trajectories(trajectories)
        return 0

    # Phase 2 — Executors.
    collectors = await run_executors(trajectories, config, llm_oracle)

    duration = time.monotonic() - start

    # Phase 3 — Report.
    critical, high = generate_report(collectors, config, trajectories, duration)

    # Exit 1 if any critical or high findings — useful for CI.
    return 1 if (critical + high) > 0 else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
