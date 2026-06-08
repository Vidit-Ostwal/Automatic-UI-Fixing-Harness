"""
AdvancedBFSExplorer — parallel BFS exploration entry point.

Replaces the single-threaded BFSExplorer for Phase 1.
Returns an ExplorationResult identical in structure so the rest of the
harness pipeline (trajectory extraction, goal writer, executors) works
without modification.

Dispatch loop:
  - Seeds the coordinator with one bootstrap work item.
  - Continuously dequeues items and spawns worker tasks, bounded by
    a semaphore (max_parallel).
  - When the queue is momentarily empty but workers are still in-flight,
    waits up to 0.5s for new items to be enqueued by returning workers.
  - Stops when drain condition is met: queue empty AND in_flight == 0.
"""

import asyncio
import logging
from pathlib import Path

from harness.planner.bfs_explorer import ExplorationResult
from harness.advanced_planner.coordinator import WorkCoordinator
from harness.advanced_planner.shared_state import SharedVisitedState
from harness.advanced_planner.worker import run_worker

logger = logging.getLogger(__name__)

_STATE_FILE = Path("advanced_planner_state.json")


async def advanced_explore(
    llm_oracle,
    max_depth: int = 3,
    max_parallel: int = 9,
    state_file: Path = _STATE_FILE,
) -> ExplorationResult:
    """
    Parallel BFS exploration using multiple Docker instances.

    Each worker starts its own Docker container internally — no pre-started
    container is required. The bootstrap worker navigates to the fresh
    container root and seeds the BFS queue.

    Parameters
    ----------
    llm_oracle      LLMOracle for action identification (can be None).
    max_depth       BFS depth limit (default 3).
    max_parallel    Max concurrent Docker instances (default 9, suggested max 16).
    state_file      Path for the FileLock-protected shared dedup state.
    """
    logger.info(
        "AdvancedBFSExplorer: starting — max_depth=%d  max_parallel=%d",
        max_depth, max_parallel,
    )

    # Prune unused Docker networks left by crashed runs (prevents address pool exhaustion).
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "network", "prune", "-f",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception:
        pass

    # Clean up any leftover state from a prior run.
    if state_file.exists():
        state_file.unlink()
    lock_path = Path(str(state_file) + ".lock")
    if lock_path.exists():
        lock_path.unlink()

    shared_state = SharedVisitedState(state_file)
    coordinator = WorkCoordinator(max_depth=max_depth)
    semaphore = asyncio.Semaphore(max_parallel)

    await coordinator.seed_bootstrap()

    pending: set[asyncio.Task] = set()

    async def spawn(item):
        async with semaphore:
            await run_worker(item, coordinator, shared_state, llm_oracle, max_depth)

    while True:
        # Drain condition: nothing queued and no workers running.
        if coordinator.is_done() and not pending:
            break

        # Spawn as many workers as the semaphore allows.
        while semaphore._value > 0:
            item = coordinator.maybe_get_work()
            if item is None:
                break
            task = asyncio.create_task(spawn(item))
            pending.add(task)
            task.add_done_callback(pending.discard)

        if not pending:
            # Nothing running, nothing queued — done.
            break

        # Wait for at least one worker to finish (or 0.5s) before re-checking.
        done, _ = await asyncio.wait(pending, timeout=0.5, return_when=asyncio.FIRST_COMPLETED)
        pending -= done

    # Wait for any stragglers.
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # Clean up state file.
    try:
        if state_file.exists():
            state_file.unlink()
        if lock_path.exists():
            lock_path.unlink()
    except Exception:
        pass

    logger.info(
        "AdvancedBFSExplorer: done — %d states  %d edges  %d findings",
        len(coordinator.nodes),
        sum(len(v) for v in coordinator.graph.values()),
        len(coordinator.findings),
    )

    return ExplorationResult(
        root_hash=coordinator.root_hash or "",
        graph=coordinator.graph,
        nodes=coordinator.nodes,
        findings=coordinator.findings,
    )
