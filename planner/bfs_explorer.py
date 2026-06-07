"""
BFS explorer — crawls the application's state space up to depth N.

Algorithm:
  1. Start at the given URL, capture state, hash it (root node).
  2. Get interactive elements → identify semantic actions.
  3. For each action: click it, capture resulting state, record the
     (current_hash → action → new_hash) edge, then restore to current state.
  4. Add newly discovered hashes to the BFS queue.
  5. Repeat until queue is empty or max_depth reached.

Restoration strategy:
  - After each click, attempt page.go_back().
  - If the resulting hash doesn't match the expected parent, re-navigate
    to the parent URL directly.
  - If restoration still fails, skip the remaining actions at this node
    (the state is unrecoverable without a server reset).

Output:
  graph  — {state_hash: [{"action", "selector", "to_hash"}]}
  nodes  — {state_hash: {"url", "screenshot", "a11y_tree"}}
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field

from browser.session import BrowserSession
from planner.action_identifier import ActionIdentifier
from planner.state_hasher import state_hash

logger = logging.getLogger(__name__)


@dataclass
class ExplorationResult:
    root_hash: str
    graph: dict[str, list[dict]] = field(default_factory=dict)
    nodes: dict[str, dict] = field(default_factory=dict)


class BFSExplorer:
    """
    Explores a web application via BFS, building a state-transition graph.

    Parameters
    ----------
    session          BrowserSession to drive.
    action_identifier ActionIdentifier for semantic action discovery.
    max_depth        How many clicks deep to explore (default 3).
    max_actions_per_node  Cap on actions tried per state (prevents explosion
                          on pages with many elements).
    """

    def __init__(
        self,
        session: BrowserSession,
        action_identifier: ActionIdentifier,
        max_depth: int = 3,
        max_actions_per_node: int = 8,
    ):
        self._session = session
        self._identifier = action_identifier
        self._max_depth = max_depth
        self._max_actions = max_actions_per_node

    async def explore(self, start_url: str) -> ExplorationResult:
        await self._session.navigate(start_url)
        root_state = await self._session.capture_state()
        root_hash = state_hash(root_state.url, root_state.a11y_tree)

        result = ExplorationResult(root_hash=root_hash)
        result.nodes[root_hash] = {
            "url": root_state.url,
            "screenshot": root_state.screenshot,
            "a11y_tree": root_state.a11y_tree,
        }

        # Queue entries: (hash, url, depth)
        queue: deque[tuple[str, str, int]] = deque()
        queue.append((root_hash, root_state.url, 0))
        visited: set[str] = {root_hash}

        while queue:
            current_hash, current_url, depth = queue.popleft()

            if depth >= self._max_depth:
                continue

            logger.info("BFS: depth=%d hash=%s url=%s", depth, current_hash[:8], current_url)

            # Restore to this node before discovering its actions.
            if not await self._restore(current_hash, current_url):
                logger.warning("BFS: could not restore to %s — skipping", current_hash[:8])
                continue

            # Identify available actions at this state.
            elements = await self._session.get_interactive_elements()
            current_state = await self._session.capture_state()
            actions = await self._identifier.identify(
                elements, current_state.screenshot
            )
            actions = actions[: self._max_actions]

            result.graph.setdefault(current_hash, [])

            for action in actions:
                # Restore to current node before trying each action.
                if not await self._restore(current_hash, current_url):
                    break

                try:
                    await self._session.click(action.representative_selector)
                except Exception as e:
                    logger.debug("BFS: click failed for %s: %s", action.name, e)
                    continue

                new_state = await self._session.capture_state()
                new_hash = state_hash(new_state.url, new_state.a11y_tree)

                edge = {
                    "action": action.name,
                    "selector": action.representative_selector,
                    "description": action.description,
                    "to_hash": new_hash,
                }
                result.graph[current_hash].append(edge)

                if new_hash not in visited:
                    visited.add(new_hash)
                    result.nodes[new_hash] = {
                        "url": new_state.url,
                        "screenshot": new_state.screenshot,
                        "a11y_tree": new_state.a11y_tree,
                    }
                    queue.append((new_hash, new_state.url, depth + 1))
                    logger.info(
                        "BFS: new state discovered via %s → %s",
                        action.name, new_hash[:8],
                    )

        return result

    # ------------------------------------------------------------------
    # State restoration
    # ------------------------------------------------------------------

    async def _restore(self, target_hash: str, target_url: str) -> bool:
        """
        Attempt to get back to the state identified by target_hash.

        Strategy:
          1. If already there, no-op.
          2. Try go_back() — works for simple navigation.
          3. If hash still doesn't match, re-navigate to target_url.
          4. If hash still doesn't match, give up (server state changed).
        """
        current_state = await self._session.capture_state()
        current_hash = state_hash(current_state.url, current_state.a11y_tree)
        if current_hash == target_hash:
            return True

        # Try browser back.
        try:
            await self._session.page.go_back(timeout=3000)
            await self._session._wait_stable()
            current_state = await self._session.capture_state()
            current_hash = state_hash(current_state.url, current_state.a11y_tree)
            if current_hash == target_hash:
                return True
        except Exception:
            pass

        # Fall back to direct URL navigation.
        try:
            await self._session.navigate(target_url)
            current_state = await self._session.capture_state()
            current_hash = state_hash(current_state.url, current_state.a11y_tree)
            if current_hash == target_hash:
                return True
        except Exception:
            pass

        # State is unrecoverable (server-side change, auth wall, etc.).
        logger.warning(
            "BFS: restoration failed — target=%s current=%s url=%s",
            target_hash[:8], current_hash[:8], target_url,
        )
        return False
