"""
SharedVisitedState — FileLock-protected cross-worker deduplication.

Two checks prevent redundant work:
  1. claimed_actions: (parent_hash, action_name) pairs — prevent two workers
     from executing the same action from the same parent state simultaneously.
  2. visited_hashes: state hashes already fully processed — prevent enqueueing
     children of a state that another worker already handled.

Both checks are atomic read-modify-write operations under a FileLock.
"""

import json
import logging
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)


class SharedVisitedState:
    def __init__(self, state_file: Path) -> None:
        self._file = state_file
        self._lock = FileLock(str(state_file) + ".lock", timeout=10)
        if not state_file.exists():
            state_file.write_text(json.dumps({
                "visited_hashes": [],
                "claimed_actions": [],
            }))

    def _read(self) -> dict:
        try:
            return json.loads(self._file.read_text())
        except Exception:
            return {"visited_hashes": [], "claimed_actions": []}

    def _write(self, data: dict) -> None:
        self._file.write_text(json.dumps(data))

    def check_and_claim_action(self, parent_hash: str, action_name: str) -> bool:
        """
        Atomically check-and-claim a (parent_hash, action_name) pair.
        Returns True if freshly claimed (caller should proceed).
        Returns False if already claimed (caller should skip).
        """
        key = f"{parent_hash}::{action_name}"
        with self._lock:
            data = self._read()
            claimed = set(data.get("claimed_actions", []))
            if key in claimed:
                return False
            claimed.add(key)
            data["claimed_actions"] = list(claimed)
            self._write(data)
            return True

    def mark_visited(self, state_hash: str) -> None:
        with self._lock:
            data = self._read()
            visited = set(data.get("visited_hashes", []))
            visited.add(state_hash)
            data["visited_hashes"] = list(visited)
            self._write(data)

    def is_visited(self, state_hash: str) -> bool:
        with self._lock:
            data = self._read()
            return state_hash in set(data.get("visited_hashes", []))

    def reset(self) -> None:
        """Clear all state (called at the start of each run)."""
        with self._lock:
            self._write({"visited_hashes": [], "claimed_actions": []})
