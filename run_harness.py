"""
run_harness.py — top-level entry point for the autonomous UI test harness.

Usage:
    uv run python run_harness.py [options]

Four-phase execution:
  Phase 1  PLANNER     — BFS exploration, emit trajectories.json
  Phase 2  GOAL WRITER — LLM converts each trajectory into plain-English goals,
                         emit trajectories_goal.json
  Phase 3  EXECUTORS   — N parallel GoalExecutors, one per goal
  Phase 3  VERIFIERS   — run simultaneously, consuming executor StepMessages
                         (findings written to output/verifier_claims/*/claims.json)
  Phase 4  REPORT      — after a full run, render report.html and open in browser
                         (also available standalone via --report)

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
    --goals-only       Read existing trajectories.json, write trajectories_goal.json, exit
    --run-goals        Read existing trajectories_goal.json, run goal executors, exit
    --rollout N        Fresh executor run: clear executor_runs/ and verifier_claims/,
                       then run executors + verifiers N times (default: 1 when flag set)
    --report           Read verifier claims, open HTML report in browser (standalone)
    --no-llm           Disable LLM oracle (deterministic checks only)
    --depth N          Override DEPTH_N
    --max-trajectories Override MAX_TRAJECTORIES
    --output DIR       Override OUTPUT_DIR
"""

import argparse
import asyncio
import json
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from harness.config import env_bool, env_int, env_str, load_env

load_env()

# ---------------------------------------------------------------------------
# Logging — set up before any local imports so all modules inherit the config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)-18s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("harness")

from harness.browser import BrowserSession
from harness.docker import DockerInstance
from harness.executor.goal_executor import ExecutorResult, GoalExecutor
from harness.executor.step_message import QUEUE_DONE_SENTINEL, StepMessage
from harness.models import Finding
from harness.verifier.agent import VerifierAgent
from harness.planner import BFSExplorer, extract_trajectories, trajectories_to_json, write_trajectory_goals
from harness.reporter import open_report


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class HarnessConfig:
    def __init__(self, args: argparse.Namespace):
        self.depth_n          = args.depth          or env_int("DEPTH_N")
        self.max_trajectories = args.max_trajectories or env_int("MAX_TRAJECTORIES")
        self.max_parallel     = env_int("MAX_PARALLEL")
        self.max_actions_per_node = env_int("MAX_ACTIONS_PER_NODE")
        self.output_dir       = Path(args.output or env_str("OUTPUT_DIR"))
        self.planner_only     = args.planner_only
        self.skip_planner     = args.skip_planner
        self.goals_only       = args.goals_only
        self.run_goals        = args.run_goals
        self.report           = args.report
        self.rollout          = args.rollout
        self.no_llm           = args.no_llm
        self.verbose_bfs      = args.verbose_bfs or env_bool("BFS_VERBOSE")
        self.run_id           = uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# LLM client factory
# ---------------------------------------------------------------------------

def _build_llm_oracle(no_llm: bool):
    if no_llm:
        return None
    try:
        from harness.oracles.llm import LLMOracle
        return LLMOracle.from_env()
    except EnvironmentError as e:
        logger.warning("LLM oracle disabled: %s", e)
        return None


# ---------------------------------------------------------------------------
# Phase 1 — Planner
# ---------------------------------------------------------------------------

def _save_trajectory_screenshots(
    trajectories: list,
    nodes: dict,
    screenshots_root: Path,
) -> None:
    """
    For each trajectory create a subfolder and write one PNG per step:
      00_start.png             — page before the first action
      01_<action_name>.png     — page after step 1
      02_<action_name>.png     — page after step 2
      ...
    Files are ordered by prefix so any file browser shows the sequence.
    """
    from harness.planner.trajectory_extractor import Trajectory

    for traj in trajectories:
        traj_id   = traj.id if hasattr(traj, "id") else traj.get("id", "T-???")
        traj_steps = traj.steps if hasattr(traj, "steps") else traj.get("steps", [])

        folder = screenshots_root / traj_id
        folder.mkdir(exist_ok=True)

        # Step 0 — initial state (from_hash of the first step).
        if traj_steps:
            first_hash = traj_steps[0].get("from_hash", "") if isinstance(traj_steps[0], dict) else traj_steps[0].from_hash
            node = nodes.get(first_hash, {})
            png  = node.get("screenshot")
            if png:
                (folder / "00_start.png").write_bytes(png)

        # One file per step — named by index + action.
        for i, step in enumerate(traj_steps, start=1):
            action   = step.get("action", "step") if isinstance(step, dict) else step.action
            to_hash  = step.get("to_hash", "")    if isinstance(step, dict) else step.to_hash
            node     = nodes.get(to_hash, {})
            png      = node.get("screenshot")
            if png:
                safe_name = action[:40].replace("/", "_").replace(" ", "_")
                (folder / f"{i:02d}_{safe_name}.png").write_bytes(png)


async def run_planner(config: HarnessConfig, llm_oracle) -> tuple[list[dict], list[Finding]]:
    """
    Spin up one Docker instance, BFS-explore the app, return serialised
    trajectories and any crashes/findings discovered during exploration.
    Also writes trajectories.json to the output dir.
    """
    logger.info("=== PHASE 1: PLANNER (depth=%d) ===", config.depth_n)

    async with DockerInstance.start() as docker:
        logger.info("Planner instance ready at %s", docker.url)

        async with BrowserSession.create() as session:
            explorer = BFSExplorer(
                session=session,
                llm_oracle=llm_oracle,
                max_depth=config.depth_n,
                max_actions_per_node=config.max_actions_per_node,
                verbose=config.verbose_bfs,
            )
            result = await explorer.explore(docker.url)

    logger.info(
        "Planner done: %d states discovered, %d edges, %d exploration findings",
        len(result.nodes),
        sum(len(v) for v in result.graph.values()),
        len(result.findings),
    )

    trajectories = extract_trajectories(result.graph, result.root_hash)
    logger.info("Extracted %d trajectories", len(trajectories))

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-trajectory screenshot folders.
    # output/screenshots/T-001/
    #   00_start.png
    #   01_create_account.png
    #   02_create_new_memo.png
    #   ...
    screenshots_root = config.output_dir / "screenshots"
    screenshots_root.mkdir(exist_ok=True)
    _save_trajectory_screenshots(trajectories, result.nodes, screenshots_root)
    logger.info("Per-trajectory screenshots saved to %s", screenshots_root)

    serialised = trajectories_to_json(trajectories, screenshots_root=str(screenshots_root))

    traj_path = config.output_dir / "trajectories.json"
    traj_path.write_text(json.dumps(serialised, indent=2))
    logger.info("Trajectories saved to %s", traj_path)

    return serialised, list(result.findings)


# ---------------------------------------------------------------------------
# Phase 1b — Goal writer
# ---------------------------------------------------------------------------

async def run_goal_writer(
    trajectories: list[dict],
    config: HarnessConfig,
    llm_oracle,
) -> list[dict]:
    """
    For each trajectory, ask the LLM to produce a plain-English goal document
    (goal, instructions, success_criteria) and write trajectories_goal.json.
    Safe to call without an LLM — heuristic fallback is always used then.
    """
    logger.info("=== PHASE 2: GOAL WRITER (%d trajectories) ===", len(trajectories))
    goals = await write_trajectory_goals(trajectories, llm_oracle, output_dir=config.output_dir)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    goal_path = config.output_dir / "trajectories_goal.json"
    goal_path.write_text(json.dumps(goals, indent=2))
    logger.info("Trajectory goals saved to %s", goal_path)
    return goals


# ---------------------------------------------------------------------------
# Phase 3 — Goal executors + verifiers (run in parallel)
# ---------------------------------------------------------------------------

def _clear_executor_artifacts(output_dir: Path) -> None:
    """Remove prior executor runs and verifier claims before a fresh rollout."""
    for name in ("executor_runs", "verifier_claims"):
        path = output_dir / name
        if path.exists():
            shutil.rmtree(path)
            logger.info("Cleared %s", path)


async def _run_verifier_consumer(
    queue: "asyncio.Queue[StepMessage]",
    goals_by_id: dict,
    total_executors: int,
    output_dir: Path,
    llm_oracle,
) -> list[Path]:
    """
    Consume StepMessages from the queue, routing each to the correct
    VerifierAgent.  One VerifierAgent is created per (test_case_id, run_id).
    Returns the list of claims.json paths written.
    """
    verifiers: dict[str, VerifierAgent] = {}
    claims_paths: list[Path] = []
    done_count = 0

    while done_count < total_executors:
        msg: StepMessage = await queue.get()

        if msg.action_name == QUEUE_DONE_SENTINEL:
            key = f"{msg.test_case_id}_{msg.run_id}"
            if key in verifiers:
                path = verifiers[key].finalize()
                claims_paths.append(path)
            done_count += 1
            logger.info(
                "Verifier: %s/%s done (%d/%d executors finished)",
                msg.test_case_id, msg.run_id, done_count, total_executors,
            )
        else:
            key = f"{msg.test_case_id}_{msg.run_id}"
            if key not in verifiers:
                goal = goals_by_id.get(msg.test_case_id, {"id": msg.test_case_id})
                verifiers[key] = VerifierAgent(goal, output_dir, llm_oracle)
                logger.info(
                    "Verifier: created agent for %s/%s", msg.test_case_id, msg.run_id
                )
            await verifiers[key].verify_step(msg)

        queue.task_done()

    return claims_paths


async def _run_one_goal(
    goal: dict,
    step_queue: "asyncio.Queue[StepMessage]",
    semaphore: asyncio.Semaphore,
    config: "HarnessConfig",
    llm_oracle,
) -> ExecutorResult:
    async with semaphore:
        executor = GoalExecutor(
            goal=goal,
            step_queue=step_queue,
            output_dir=config.output_dir,
            llm_oracle=llm_oracle,
        )
        return await executor.run()


async def _run_goal_executors_once(
    goals: list[dict],
    config: "HarnessConfig",
    llm_oracle,
    *,
    rollout_index: int = 1,
    rollout_total: int = 1,
) -> tuple[list[ExecutorResult], list[Path]]:
    """Single pass: run GoalExecutors and VerifierAgents in parallel."""
    capped = goals[: config.max_trajectories]
    if rollout_total > 1:
        logger.info(
            "=== PHASE 3: EXECUTORS + VERIFIERS (rollout %d/%d, %d goals, max %d parallel) ===",
            rollout_index, rollout_total, len(capped), config.max_parallel,
        )
    else:
        logger.info(
            "=== PHASE 3: EXECUTORS + VERIFIERS (%d goals, max %d parallel) ===",
            len(capped), config.max_parallel,
        )

    goals_by_id = {g["id"]: g for g in goals if "id" in g}
    step_queue: asyncio.Queue[StepMessage] = asyncio.Queue()
    semaphore = asyncio.Semaphore(config.max_parallel)

    executor_tasks = [
        asyncio.create_task(
            _run_one_goal(g, step_queue, semaphore, config, llm_oracle)
        )
        for g in capped
    ]
    verifier_task = asyncio.create_task(
        _run_verifier_consumer(
            step_queue,
            goals_by_id=goals_by_id,
            total_executors=len(capped),
            output_dir=config.output_dir,
            llm_oracle=llm_oracle,
        )
    )

    results: list[ExecutorResult] = await asyncio.gather(*executor_tasks)
    claims_paths: list[Path] = await verifier_task

    succeeded = sum(1 for r in results if r.completed)
    logger.info(
        "GoalExecutors rollout %d/%d done: %d/%d goals completed successfully",
        rollout_index, rollout_total, succeeded, len(results),
    )
    if claims_paths:
        logger.info("Verifier claims (rollout %d/%d):", rollout_index, rollout_total)
        for p in claims_paths:
            logger.info("  → %s", p)

    return results, claims_paths


async def run_goal_executors(
    goals: list[dict],
    config: "HarnessConfig",
    llm_oracle,
) -> tuple[list[ExecutorResult], list[Path]]:
    """
    Run GoalExecutors + VerifierAgents for each goal, bounded by max_parallel.

    When --rollout N is set, clears executor_runs/ and verifier_claims/ first,
    then repeats the full executor+verifier pass N times.
    Writes executor_trajectories.json summarising all runs.
    Returns (executor results, verifier claims.json paths).
    """
    if not goals:
        logger.warning("GoalExecutors: no goals to run")
        return [], []

    rollout_total = config.rollout if config.rollout is not None else 1
    if rollout_total < 1:
        logger.error("--rollout must be >= 1 (got %d)", rollout_total)
        raise ValueError(f"rollout must be >= 1, got {rollout_total}")

    if config.rollout is not None:
        _clear_executor_artifacts(config.output_dir)

    all_results: list[ExecutorResult] = []
    all_claims: list[Path] = []
    summary: list[dict] = []

    for rollout_index in range(1, rollout_total + 1):
        results, claims_paths = await _run_goal_executors_once(
            goals,
            config,
            llm_oracle,
            rollout_index=rollout_index,
            rollout_total=rollout_total,
        )
        all_results.extend(results)
        all_claims.extend(claims_paths)
        for r in results:
            entry = {
                "rollout":         rollout_index,
                "test_case_id":    r.test_case_id,
                "run_id":          r.run_id,
                "final_state":     r.final_state,
                "completed":       r.completed,
                "steps_executed":  r.steps_executed,
                "steps_succeeded": r.steps_succeeded,
                "run_dir":         str(r.run_dir.relative_to(config.output_dir)),
                "error":           r.error,
            }
            summary.append(entry)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "executor_trajectories.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Executor summary written to %s", summary_path)

    return all_results, all_claims


# ---------------------------------------------------------------------------
# Run summary
# ---------------------------------------------------------------------------

def _count_severities(
    claims_paths: list[Path],
    bfs_findings: list[Finding] | None = None,
) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0}
    for path in claims_paths:
        if not path.exists():
            continue
        for raw in json.loads(path.read_text()).get("findings", []):
            counts["total"] += 1
            sev = raw.get("severity", "low")
            if sev in counts:
                counts[sev] += 1
    for finding in bfs_findings or []:
        counts["total"] += 1
        sev = finding.severity.value
        if sev in counts:
            counts[sev] += 1
    return counts


def _print_run_summary(
    config: HarnessConfig,
    trajectories: list[dict],
    claims_paths: list[Path],
    bfs_findings: list[Finding] | None,
    duration: float,
) -> tuple[int, int]:
    counts = _count_severities(claims_paths, bfs_findings)
    print("\n" + "=" * 60)
    print(f"  Run ID : {config.run_id}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Trajectories: {len(trajectories)}")
    print(f"  Verifier claims: {len(claims_paths)} file(s)")
    print(f"  Findings: {counts['total']} total")
    print(f"    Critical : {counts['critical']}")
    print(f"    High     : {counts['high']}")
    print(f"    Medium   : {counts['medium']}")
    print(f"    Low      : {counts['low']}")
    if claims_paths:
        print("  Claims:")
        for p in claims_paths:
            print(f"    → {p}")
    print("=" * 60 + "\n")
    return counts["critical"], counts["high"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _open_report_after_run(output_dir: Path) -> None:
    """Render and serve the HTML report when verifier claims exist."""
    claims_root = output_dir / "verifier_claims"
    if not claims_root.is_dir() or not any(claims_root.glob("*/claims.json")):
        logger.info("No verifier claims — skipping report")
        return
    logger.info("=== PHASE 4: REPORT ===")
    open_report(output_dir)


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
    p.add_argument("--goals-only", action="store_true",
                   help="Read existing trajectories.json, write trajectories_goal.json, and exit")
    p.add_argument("--run-goals", action="store_true",
                   help="Read existing trajectories_goal.json, run goal executors, and exit")
    p.add_argument("--report", action="store_true",
                   help="Read verifier claims from output dir, open HTML report in browser")
    p.add_argument("--no-llm",       action="store_true",
                   help="Disable LLM oracle (deterministic checks only)")
    p.add_argument("--verbose-bfs",  action="store_true",
                   help="Emit step-level BFS debug logs (also: BFS_VERBOSE=1)")
    p.add_argument("--depth",        type=int, default=None,
                   help="BFS depth override (env: DEPTH_N)")
    p.add_argument("--max-trajectories", type=int, default=None,
                   help="Executor cap override (env: MAX_TRAJECTORIES)")
    p.add_argument("--rollout", type=int, default=None, metavar="N",
                   help="Clear executor_runs/ and verifier_claims/, then run N rollout(s)")
    p.add_argument("--output",       default=None,
                   help="Output directory override (env: OUTPUT_DIR)")
    return p.parse_args()


async def _main() -> int:
    args = _parse_args()
    config = HarnessConfig(args)

    rollout_label = (
        str(config.rollout) if config.rollout is not None else "off"
    )
    logger.info(
        "Harness run %s | depth=%d | max_traj=%d | max_parallel=%d | rollout=%s",
        config.run_id, config.depth_n, config.max_trajectories,
        config.max_parallel, rollout_label,
    )

    # --report: standalone report viewer — no harness phases run.
    if config.report:
        claims_root = config.output_dir / "verifier_claims"
        if not claims_root.is_dir():
            logger.error("--report set but %s not found — run the harness first", claims_root)
            return 2
        open_report(config.output_dir)
        return 0

    llm_oracle = _build_llm_oracle(config.no_llm)

    start = time.monotonic()

    # --goals-only: standalone goal-writing from an existing trajectories.json.
    if config.goals_only:
        traj_path = config.output_dir / "trajectories.json"
        if not traj_path.exists():
            logger.error("--goals-only set but %s not found", traj_path)
            return 2
        trajectories = json.loads(traj_path.read_text())
        logger.info("Loaded %d trajectories from %s", len(trajectories), traj_path)
        await run_goal_writer(trajectories, config, llm_oracle)
        return 0

    # --run-goals: skip planner + goal writer; run executors + verifiers only.
    if config.run_goals:
        goal_path = config.output_dir / "trajectories_goal.json"
        if not goal_path.exists():
            logger.error("--run-goals set but %s not found", goal_path)
            return 2
        goals = json.loads(goal_path.read_text())
        logger.info("Loaded %d goals from %s", len(goals), goal_path)
        _, claims_paths = await run_goal_executors(goals, config, llm_oracle)
        duration = time.monotonic() - start
        critical, high = _print_run_summary(config, [], claims_paths, None, duration)
        exit_code = 1 if (critical + high) > 0 else 0
        _open_report_after_run(config.output_dir)
        return exit_code

    # Phase 1 — Planner.
    bfs_findings: list[Finding] = []
    if config.skip_planner:
        traj_path = config.output_dir / "trajectories.json"
        if not traj_path.exists():
            logger.error("--skip-planner set but %s not found", traj_path)
            return 2
        trajectories = json.loads(traj_path.read_text())
        logger.info("Loaded %d trajectories from %s", len(trajectories), traj_path)
    else:
        trajectories, bfs_findings = await run_planner(config, llm_oracle)

    if not trajectories:
        logger.warning("No trajectories to execute — check BFS depth and app URL")
        trajectories = []

    # Phase 2 — Goal writer (always runs after planner, including --skip-planner).
    goals = await run_goal_writer(trajectories, config, llm_oracle)

    # Early exit when only BFS exploration was requested.
    if config.planner_only:
        _print_trajectories(trajectories)
        return 0

    # Phase 3 — Executors + verifiers (parallel).
    _, claims_paths = await run_goal_executors(goals, config, llm_oracle)

    duration = time.monotonic() - start

    # Summary — verifier claims are the primary output; include BFS findings in counts.
    critical, high = _print_run_summary(
        config, trajectories, claims_paths, bfs_findings, duration
    )

    exit_code = 1 if (critical + high) > 0 else 0
    _open_report_after_run(config.output_dir)
    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
