"""
WorkCoordinator — async BFS coordinator for parallel exploration.

Owns:
  - In-memory BFS queue (asyncio.Queue)
  - In-flight counter: items dequeued but not yet reported back
  - Accumulated exploration graph, node registry, and findings
  - Exploration memory snapshot sent to workers for LLM context

Drain condition: queue empty AND in_flight == 0 → BFS complete.
Coordinator crash is a hard failure — no checkpointing.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from harness.models import Finding

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    # Structured trajectory: list of action dicts to replay in order.
    # Empty for the bootstrap worker.
    trajectory: list[dict]
    # Expected state hash after replaying trajectory.
    # None for the bootstrap worker.
    expected_parent_hash: str | None
    # The action to execute after replay. None for the bootstrap worker
    # (bootstrap only identifies actions, doesn't execute one).
    action_to_execute: dict | None
    # BFS depth of the *child* state this worker will produce.
    depth: int
    # Hash of the parent node (used for graph edge recording).
    parent_hash: str | None = None


@dataclass
class WorkerResult:
    parent_hash: str | None          # None for bootstrap
    action_executed: dict | None     # None for bootstrap
    child_hash: str | None           # resulting state hash (None on failure)
    child_url: str | None
    child_a11y_tree: dict | None
    child_screenshot: bytes | None
    # Actions identified at child state (to enqueue as next-level work).
    child_actions: list[dict]
    # Full trajectory to reach child state (trajectory + action_executed).
    child_trajectory: list[dict]
    depth: int
    finding: Finding | None = None
    replay_failed: bool = False      # could not verify parent hash after retries
    action_failed: bool = False      # action execution failed or pre-empted by dedup


class WorkCoordinator:
    def __init__(self, max_depth: int = 3) -> None:
        self._queue: asyncio.Queue[WorkItem] = asyncio.Queue()
        self._in_flight: int = 0
        self._lock = asyncio.Lock()
        self._max_depth = max_depth

        # Accumulated exploration graph — same format as BFSExplorer output.
        self.graph: dict[str, list[dict]] = {}
        self.nodes: dict[str, dict] = {}
        self.findings: list[Finding] = []
        self.root_hash: str | None = None

        # Compact log for LLM context sent to each worker.
        self._exploration_entries: list[dict] = []

    async def seed_bootstrap(self) -> None:
        """Enqueue the single bootstrap work item (empty trajectory, no action)."""
        await self._queue.put(WorkItem(
            trajectory=[],
            expected_parent_hash=None,
            action_to_execute=None,
            depth=0,
        ))

    def maybe_get_work(self) -> WorkItem | None:
        """
        Non-blocking dequeue. Increments in_flight on success.
        Returns None when queue is currently empty.
        """
        try:
            item = self._queue.get_nowait()
            self._in_flight += 1
            return item
        except asyncio.QueueEmpty:
            return None

    def is_done(self) -> bool:
        """True when queue is empty and no workers are in-flight."""
        return self._queue.empty() and self._in_flight == 0

    async def submit_result(self, result: WorkerResult) -> None:
        """
        Ingest a worker result. Records graph edges, enqueues child work items.
        Always decrements in_flight so the drain condition stays accurate.
        """
        async with self._lock:
            self._in_flight -= 1

            if result.replay_failed:
                logger.warning(
                    "Coordinator: replay failed — action=%s parent=%s",
                    result.action_executed.get("name") if result.action_executed else "?",
                    (result.parent_hash or "")[:8],
                )
                return

            if result.action_failed:
                return

            # Bootstrap result: sets root hash, seeds queue with initial actions.
            if result.parent_hash is None:
                self.root_hash = result.child_hash
                if result.child_hash:
                    self.nodes[result.child_hash] = {
                        "url": result.child_url,
                        "screenshot": result.child_screenshot,
                        "a11y_tree": result.child_a11y_tree,
                    }
                    self.graph.setdefault(result.child_hash, [])

                logger.info(
                    "Coordinator: bootstrap done — root=%s  %d action(s) to explore",
                    (result.child_hash or "")[:8],
                    len(result.child_actions),
                )
                self._enqueue_children(
                    parent_hash=result.child_hash,
                    parent_trajectory=result.child_trajectory,
                    child_actions=result.child_actions,
                    parent_depth=0,
                )
                return

            # Normal result.
            if result.child_hash:
                edge = {
                    "action":           result.action_executed["name"],
                    "description":      result.action_executed.get("description", ""),
                    "steps":            result.action_executed.get("steps", []),
                    "expected_outcome": result.action_executed.get("expected_outcome", ""),
                    "to_hash":          result.child_hash,
                    "queued":           result.child_hash not in self.nodes,
                }
                self.graph.setdefault(result.parent_hash, []).append(edge)

                if result.child_hash not in self.nodes:
                    self.nodes[result.child_hash] = {
                        "url": result.child_url,
                        "screenshot": result.child_screenshot,
                        "a11y_tree": result.child_a11y_tree,
                    }
                    self.graph.setdefault(result.child_hash, [])

                self._exploration_entries.append({
                    "url": result.child_url or "",
                    "depth": result.depth,
                    "action": result.action_executed.get("name", ""),
                })

                # Enqueue children only if below depth limit and there are child actions.
                if result.depth + 1 < self._max_depth and result.child_actions:
                    self._enqueue_children(
                        parent_hash=result.child_hash,
                        parent_trajectory=result.child_trajectory,
                        child_actions=result.child_actions,
                        parent_depth=result.depth,
                    )

            if result.finding:
                self.findings.append(result.finding)

    def _enqueue_children(
        self,
        parent_hash: str | None,
        parent_trajectory: list[dict],
        child_actions: list[dict],
        parent_depth: int,
    ) -> None:
        for action in child_actions:
            self._queue.put_nowait(WorkItem(
                trajectory=parent_trajectory,
                expected_parent_hash=parent_hash,
                action_to_execute=action,
                depth=parent_depth + 1,
                parent_hash=parent_hash,
            ))
        if child_actions:
            logger.debug(
                "Coordinator: enqueued %d child action(s) at depth %d from parent=%s",
                len(child_actions), parent_depth + 1, (parent_hash or "")[:8],
            )

    def get_exploration_snapshot(self) -> str:
        """Compact exploration log sent to workers for LLM context."""
        if not self._exploration_entries:
            return ""
        lines = ["## Already-explored states (do NOT re-suggest these workflows)"]
        for e in self._exploration_entries[-60:]:
            lines.append(f'- d{e["depth"]} {e["url"]}  tried=[{e["action"]}]')
        return "\n".join(lines)
