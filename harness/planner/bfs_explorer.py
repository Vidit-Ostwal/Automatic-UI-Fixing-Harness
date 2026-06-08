"""
BFS explorer — crawls the application's state space up to depth N.

Algorithm:
  1. Start at the given URL, capture state, hash it (root node).
  2. Ask ActionIdentifier for the complete workflows available on this page.
     Each workflow may be multi-step: fill a form, click submit, etc.
  3. For each workflow: restore to the node's URL, execute all steps,
     capture the resulting state, record the edge.
  4. Add newly discovered hashes to the BFS queue.
  5. Repeat until queue is empty or max_depth reached.

Forward-only model:
  - Server-side mutations (create memo, pin, archive) are permanent within
    one Docker run. Attempting to restore to a pre-mutation state via
    go_back() would fail because the server still has the created data.
  - Instead, every new state discovered is queued for further exploration.
    "List with 1 memo" is a richer state than "empty list" — it unlocks
    pin, archive, edit, delete — so we explore forward from it.

URL-level restoration:
  - Before each action, _url_restore() ensures the browser is at the current
    node's URL path. This lets multiple actions be tried from the same
    starting page (e.g. language toggle AND signup on the auth page).
  - Matches on URL path only — not hash — so mutating actions that change
    server state are handled correctly (same URL, different content).

State explosion prevention:
  - visited_hashes:       Never re-queue a hash already explored.
  - visited_url_actions:  At a given URL path, each action name is tried
                          at most once. Prevents "create_memo" being retried
                          at every subsequent list state.

Crash detection:
  - LogicOracle fires check_no_crash() after every action during exploration.
    Cheap (no LLM), catches crashes/blank pages immediately rather than
    waiting for executor replay.

Verbose mode (verbose=True or BFS_VERBOSE=1):
  - Logs page title, every element found, every action tried, every step
    attempted, the before/after state for each transition, and why states
    were skipped or re-queued.

Output:
  graph    — {state_hash: [{"action", "steps", "description", "to_hash"}]}
  nodes    — {state_hash: {"url", "screenshot", "a11y_tree"}}
  findings — list of Finding objects detected during exploration
"""

import asyncio
import logging
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from harness.config import env_bool
from harness.browser import BrowserSession
from harness.utils.step_runner import dismiss_overlays, execute_steps, wait_for_navigation
from harness.models import Finding
from harness.oracles.logic import LogicOracle
from harness.planner.action_identifier import ActionIdentifier, SemanticAction
from harness.planner.state_hasher import state_hash
from harness.utils.url import normalise_url

if TYPE_CHECKING:
    from harness.oracles.llm import LLMOracle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANSI colour helpers — disabled automatically when stderr is not a TTY
# (log file redirect, CI pipe, etc.)
# ---------------------------------------------------------------------------
_TTY = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

def _ansi(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

def _bold(t: str)   -> str: return _ansi("1",     t)
def _dim(t: str)    -> str: return _ansi("2",     t)
def _cyan(t: str)   -> str: return _ansi("1;36",  t)
def _green(t: str)  -> str: return _ansi("1;32",  t)
def _yellow(t: str) -> str: return _ansi("33",    t)
def _red(t: str)    -> str: return _ansi("1;31",  t)
def _gray(t: str)   -> str: return _ansi("90",    t)


MEMORY_FILE = Path("exploration_memory.md")


class ExplorationMemory:
    """
    Lightweight exploration log written to exploration_memory.md.

    Cleared at the start of every BFS run. After each node is fully
    explored the caller calls record_node(); the summary is then passed
    into ActionIdentifier so the LLM can reason about what is still
    unexplored instead of re-suggesting the same workflows.
    """

    def __init__(self, path: Path = MEMORY_FILE) -> None:
        self._path = path
        self._entries: list[dict] = []
        self._path.write_text("# BFS Exploration Memory\n\n_Run in progress…_\n")

    def record_node(
        self,
        url: str,
        page_title: str,
        depth: int,
        actions_tried: list[str],
        new_states_found: int,
    ) -> None:
        # Normalise multi-line titles (calendar grids, etc.) to a single line.
        clean_title = page_title.replace("\n", " · ").strip()[:60]
        self._entries.append(
            dict(
                url=url,
                page_title=clean_title,
                depth=depth,
                actions_tried=actions_tried,
                new_states=new_states_found,
            )
        )
        self._flush()

    def summary(self) -> str:
        """Compact text fed to the LLM before it suggests new workflows."""
        if not self._entries:
            return ""
        lines = ["## Already-explored states (do NOT re-suggest these workflows)"]
        for e in self._entries:
            tried = ", ".join(e["actions_tried"]) if e["actions_tried"] else "none"
            if len(tried) > 120:
                tried = tried[:117] + "…"
            lines.append(
                f'- d{e["depth"]} {e["url"]}  "{e["page_title"]}"'
                f'  tried=[{tried}]  new={e["new_states"]}'
            )
        return "\n".join(lines)

    def display_lines(self) -> list[str]:
        """Short per-entry lines for the terminal sub-box (no newlines)."""
        if not self._entries:
            return []
        out = []
        for e in self._entries:
            tried = ", ".join(e["actions_tried"]) if e["actions_tried"] else "none"
            if len(tried) > 70:
                tried = tried[:67] + "…"
            out.append(
                f'd{e["depth"]} {e["url"]}  '
                f'"{e["page_title"]}"  '
                f'tried=[{tried}]  new={e["new_states"]}'
            )
        return out

    def _flush(self) -> None:
        lines = [
            "# BFS Exploration Memory\n",
            f"Nodes explored so far: **{len(self._entries)}**\n\n",
        ]
        for e in self._entries:
            tried = ", ".join(e["actions_tried"]) if e["actions_tried"] else "none"
            lines.append(f"### {e['page_title']} (depth={e['depth']})\n")
            lines.append(f"- **URL**: {e['url']}\n")
            lines.append(f"- **Actions tried**: {tried}\n")
            lines.append(f"- **New states found**: {e['new_states']}\n\n")
        self._path.write_text("".join(lines))


@dataclass
class ExplorationResult:
    root_hash: str
    graph: dict[str, list[dict]] = field(default_factory=dict)
    nodes: dict[str, dict] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)


def _page_title(a11y_tree: dict) -> str:
    """Extract first heading or landmark name from a11y tree for log context."""
    def _walk(node: dict, depth: int = 0) -> str | None:
        if not node:
            return None
        role = node.get("role", "")
        name = node.get("name", "").strip()
        if role == "heading" and name:
            return name
        if role in ("main", "banner", "navigation") and name:
            return f"[{role}] {name}"
        if depth < 4:
            for child in node.get("children", []):
                found = _walk(child, depth + 1)
                if found:
                    return found
        return None
    return _walk(a11y_tree) or "(untitled)"


class BFSExplorer:
    """
    Explores a web application via BFS, building a state-transition graph.

    Parameters
    ----------
    session              BrowserSession to drive.
    llm_oracle           Optional LLMOracle for workflow analysis.
                         Without it, DOM heuristics are used.
    max_depth            How many workflow-steps deep to explore (default 3).
    max_actions_per_node Cap on workflows tried per state (default 8).
    verbose              Emit step-level debug logs. Also enabled by
                         setting env var BFS_VERBOSE=1.
    """

    def __init__(
        self,
        session: BrowserSession,
        llm_oracle: "LLMOracle | None" = None,
        max_depth: int = 3,
        max_actions_per_node: int = 8,
        verbose: bool = False,
    ):
        self._session = session
        self._llm = llm_oracle
        self._identifier = ActionIdentifier(llm_client=llm_oracle)
        self._logic = LogicOracle()
        self._max_depth = max_depth
        self._max_actions = max_actions_per_node
        self._verbose = verbose or env_bool("BFS_VERBOSE")
        self._box_depth = 0  # updated per-node for indentation

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    _BOX_WIDTH = 70  # inner box width (before indent)

    def _ind(self) -> str:
        """Two spaces per depth level — makes tree nesting visible."""
        return "  " * self._box_depth

    def _vlog(self, msg: str, *args) -> None:
        if self._verbose:
            logger.info(msg, *args)
        else:
            logger.debug(msg, *args)

    def _box_open(self, depth: int, hash8: str, url: str, page: str, replay: list = None) -> None:
        self._box_depth = depth
        ind = self._ind()
        page_line = page.replace("\n", " · ").strip()[:55]
        bar_len = max(4, self._BOX_WIDTH - 26 - len(ind))
        header = f"{ind}{_cyan('┌──')} {_bold(hash8)}  {_gray(f'd={depth}')}  {_cyan('─' * bar_len)}"
        logger.info("BFS: %s", header)
        logger.info("BFS: %s%s %s %s", ind, _cyan("│"), _gray("url :"), url)
        logger.info("BFS: %s%s %s %s", ind, _cyan("│"), _gray("page:"), page_line)
        if replay:
            path = " → ".join(a.name for a in replay)
            logger.info("BFS: %s%s %s %s", ind, _cyan("│"), _gray("via :"), _yellow(path))

    def _box_memory(self, memory) -> None:
        lines = memory.display_lines()
        if not lines:
            return
        ind = self._ind()
        logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"┌─[memory: {len(lines)} state(s) explored]"))
        for line in lines:
            logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"│ {line}"))
        logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray("└─[memory]"))

    def _box_elements(self, elements: list[dict]) -> None:
        ind = self._ind()
        total = len(elements)
        suffix = f"  {_gray(f'(showing 10 of {total})')}" if total > 10 else ""
        logger.info("BFS: %s%s %s%s", ind, _cyan("│"), _gray(f"elements: {total}"), suffix)
        for el in elements[:10]:
            logger.info(
                "BFS: %s%s   %s  %s  %s",
                ind, _cyan("│"),
                _gray(f"{el.get('role', '?'):<10}"),
                _dim(f"{repr(el.get('label', ''))[:28]:<30}"),
                _gray(el.get("selector", "?")),
            )
        if total > 10:
            logger.info("BFS: %s%s   %s", ind, _cyan("│"), _gray(f"… and {total - 10} more"))

    def _box_llm_open(self, label: str) -> None:
        ind = self._ind()
        logger.info("BFS: %s%s %s", ind, _gray("│"), _gray(f" ┌─[LLM: {label}]"))

    def _box_llm_close(self) -> None:
        ind = self._ind()
        logger.info("BFS: %s%s %s", ind, _gray("│"), _gray(" └─[LLM: done]"))

    def _box_actions(self, actions: list) -> None:
        ind = self._ind()
        if actions:
            names = ", ".join(_bold(a.name) for a in actions)
            logger.info("BFS: %s%s %s %s", ind, _cyan("│"), _yellow(f"actions ({len(actions)}):"), names)
        else:
            logger.info("BFS: %s%s %s", ind, _cyan("│"), _yellow("actions: none  (dead end)"))

    def _box_trying(self, idx: int, total: int, action) -> None:
        ind = self._ind()
        label = f"[{idx}/{total}] {_bold(action.name)}  ({len(action.steps)} step(s))"
        logger.info("BFS: %s%s %s", ind, _cyan("├──"), label)

    def _box_steps(self, action) -> None:
        ind = self._ind()
        for s in action.steps:
            val_part = f", {s.value!r}" if s.value else ""
            logger.info("BFS: %s%s   %s", ind, _cyan("│"), _dim(f"{s.type}({s.selector[:50]}{val_part})"))

    def _box_skip(self, name: str, reason: str) -> None:
        ind = self._ind()
        logger.info("BFS: %s%s %s %s", ind, _cyan("│"), _yellow(f"skip '{name}'"), _gray(f"— {reason}"))

    def _box_result(self, ok: bool, detail: str, from8: str, to8: str, queued_info: str = "") -> None:
        ind = self._ind()
        if ok:
            if "queued" in queued_info:
                status = _green(f"✓  {detail}  |  {from8} → {to8}{queued_info}")
            else:
                status = _yellow(f"↩  {detail}  |  {from8} → {to8}{queued_info}")
            logger.info("BFS: %s%s %s", ind, _cyan("│"), status)
        else:
            short = detail.split("\n")[0][:120]
            logger.info("BFS: %s%s %s", ind, _cyan("│"), _red(f"✗  {short}"))

    def _box_close(self) -> None:
        ind = self._ind()
        logger.info("BFS: %s%s", ind, _cyan("└" + "─" * self._BOX_WIDTH))

    # ------------------------------------------------------------------
    # Main exploration loop
    # ------------------------------------------------------------------

    async def explore(self, start_url: str) -> ExplorationResult:
        memory = ExplorationMemory()

        await self._session.navigate(start_url)
        root_state = await self._session.capture_state()
        root_elements = await self._session.get_interactive_elements()
        root_hash = state_hash(root_state.url, root_state.a11y_tree, root_elements)
        root_title = _page_title(root_state.a11y_tree)

        logger.info("BFS: root  hash=%s  url=%s  title=%r", root_hash[:8], root_state.url, root_title)

        result = ExplorationResult(root_hash=root_hash)
        result.nodes[root_hash] = {
            "url": root_state.url,
            "screenshot": root_state.screenshot,
            "a11y_tree": root_state.a11y_tree,
        }

        # Queue entries: (hash, url, depth, replay)
        #
        # replay is a list of SemanticAction objects that must be executed
        # after navigating to `url` in order to reach this UI state.
        # It is empty when a URL change marks the restoration point (the new
        # URL fully identifies the state).  It is non-empty for same-URL
        # transitions where only the UI changed (popup opened, filter applied,
        # dropdown expanded) — those states can only be recovered by re-clicking
        # the sequence of buttons that produced them.
        queue: deque[tuple[str, str, int, list]] = deque()
        queue.append((root_hash, root_state.url, 0, []))
        self._current_queue = queue  # shared ref so _try_alternative_fills can enqueue

        # Hash-level dedup: never re-explore a state already visited.
        visited_hashes: set[str] = {root_hash}

        # (url_path, replay_fingerprint) → set of action names already tried.
        # Including the replay key means "create_memo at / with popup open"
        # is tracked separately from "create_memo at / without popup".
        visited_url_actions: dict[tuple[str, tuple], set[str]] = {}

        while queue:
            current_hash, current_url, depth, current_replay = queue.popleft()

            if depth >= self._max_depth:
                logger.info(
                    "BFS: %sskip  %s — depth limit (%d) reached",
                    "  " * depth, current_hash[:8], self._max_depth,
                )
                continue

            # Restore this node's exact UI state before identifying actions.
            await self._restore_node(current_url, current_replay)
            current_state = await self._session.capture_state()
            page_title = _page_title(current_state.a11y_tree)
            elements = await self._session.get_interactive_elements()
            mem_summary = memory.summary()

            # ── open node box ──────────────────────────────────────────
            self._box_open(depth, current_hash[:8], current_url, page_title, current_replay)
            self._box_memory(memory)
            if self._verbose:
                self._box_elements(elements)

            # Identify actions — bracket with a sub-box so LLM logs are visually grouped.
            self._box_llm_open("identifying actions")
            all_actions = await self._identifier.identify(
                elements,
                current_state.a11y_tree,
                current_state.screenshot,
                explored_context=mem_summary,
            )
            self._box_llm_close()
            actions = all_actions[: self._max_actions]

            if len(all_actions) > self._max_actions:
                ind = self._ind()
                logger.info(
                    "BFS: %s%s %s",
                    ind, _cyan("│"),
                    _yellow(f"selecting top {self._max_actions} of {len(all_actions)} identified actions"),
                )
            self._box_actions(actions)

            result.graph.setdefault(current_hash, [])
            url_path = normalise_url(current_url)
            replay_key = tuple(a.name for a in current_replay)
            seen_actions_at_url = visited_url_actions.setdefault((url_path, replay_key), set())

            total_actions = len(actions)
            tried_idx = 0
            new_states_this_node = 0
            actions_tried_this_node: list[str] = []
            for action in actions:
                if action.name in seen_actions_at_url:
                    self._box_skip(action.name, f"already tried at {url_path}")
                    continue

                seen_actions_at_url.add(action.name)
                actions_tried_this_node.append(action.name)
                tried_idx += 1
                self._box_trying(tried_idx, total_actions, action)

                # Restore this node's exact UI state before each action so
                # one action's side-effects don't bleed into the next.
                if not await self._restore_node(current_url, current_replay, reload=True):
                    self._box_skip(action.name, "could not restore node state")
                    continue

                if self._verbose:
                    self._box_steps(action)

                success, fail_reason, had_fill_issues = await self._execute_action(action)
                if not success:
                    self._box_result(False, fail_reason, current_hash[:8], "")
                    continue

                await self._wait_for_navigation(current_url)

                new_state = await self._session.capture_state()
                new_elements = await self._session.get_interactive_elements()
                new_hash = state_hash(new_state.url, new_state.a11y_tree, new_elements)
                new_title = _page_title(new_state.a11y_tree)

                url_changed = normalise_url(new_state.url) != normalise_url(current_url)
                if url_changed:
                    transition = f"url {normalise_url(current_url)} → {normalise_url(new_state.url)}"
                    # URL change is a hard restoration point — no replay needed.
                    new_replay = []
                elif new_hash != current_hash:
                    transition = f"content changed  page={new_title[:40]!r}"
                    # Same URL, UI changed — must replay the action to restore.
                    new_replay = list(current_replay) + [action]
                else:
                    transition = "no change"
                    new_replay = list(current_replay)

                if new_hash not in visited_hashes:
                    replay_hint = f" [replay:{len(new_replay)}]" if new_replay else ""
                    queued_info = f"  → queued at depth {depth + 1}{replay_hint}"
                    new_states_this_node += 1
                else:
                    queued_info = "  → already known (edge recorded)"
                self._box_result(True, transition, current_hash[:8], new_hash[:8], queued_info)

                crash = self._logic.check_no_crash(new_state, action.name)
                if crash:
                    crash.trajectory_id = f"BFS-{current_hash[:6]}"
                    result.findings.append(crash)
                    logger.warning("BFS: %s│   CRASH after '%s': %s", self._ind(), action.name, crash.title)

                edge = {
                    "action":           action.name,
                    "description":      action.description,
                    "steps":            [{"type": s.type, "selector": s.selector, "value": s.value}
                                         for s in action.steps],
                    "expected_outcome": action.expected_outcome,
                    "to_hash":          new_hash,
                    "queued":           new_hash not in visited_hashes,
                }
                result.graph[current_hash].append(edge)

                if new_hash not in visited_hashes:
                    visited_hashes.add(new_hash)
                    result.nodes[new_hash] = {
                        "url":        new_state.url,
                        "screenshot": new_state.screenshot,
                        "a11y_tree":  new_state.a11y_tree,
                    }
                    queue.append((new_hash, new_state.url, depth + 1, new_replay))

                # Only retry with alternative fills when the original fill had
                # issues (a value was rejected and needed an LLM alternative).
                # A clean same-URL state change means the action succeeded — no
                # retry needed.
                if not url_changed and new_hash != current_hash and had_fill_issues:
                    await self._try_alternative_fills(
                        action, current_url, current_replay,
                        visited_hashes, result, depth,
                        current_hash, new_hash,
                    )

            # ── close node box ─────────────────────────────────────────
            self._box_close()

            # Record this node in the exploration memory so subsequent LLM
            # calls know what has already been tried here.
            memory.record_node(
                url=current_url,
                page_title=page_title,
                depth=depth,
                actions_tried=actions_tried_this_node,
                new_states_found=new_states_this_node,
            )

        logger.info(
            "BFS: done — %d states, %d edges, %d findings",
            len(result.nodes),
            sum(len(v) for v in result.graph.values()),
            len(result.findings),
        )
        return result

    # ------------------------------------------------------------------
    # Post-action navigation wait
    # ------------------------------------------------------------------

    async def _wait_for_navigation(self, prev_url: str, timeout_ms: int = 2000) -> None:
        await wait_for_navigation(self._session, prev_url, timeout_ms)

    # ------------------------------------------------------------------
    # URL-level restoration
    # ------------------------------------------------------------------

    async def _restore_node(
        self,
        url: str,
        replay: list,
        reload: bool = False,
    ) -> bool:
        """
        Restore the browser to the exact UI state of a BFS node.

        1. Navigate to `url` (with optional reload to flush prior UI state).
        2. Re-execute each action in `replay` to recreate ephemeral UI state
           (open popups, expanded dropdowns, applied filters) that a bare
           URL navigation cannot restore.

        Returns False if the URL navigate fails (auth guard redirect) or if
        any replay step fails (element gone, overlay blocked, etc.).
        """
        if not await self._url_restore(url, reload=reload):
            return False

        for step_action in replay:
            ind = self._ind()
            logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"→ replay  {step_action.name}"))
            success, reason, _ = await self._execute_action(step_action)
            if not success:
                logger.info(
                    "BFS: %s%s %s",
                    ind, _cyan("│"),
                    _yellow(f"replay failed ({step_action.name}): {reason.split(chr(10))[0][:60]}"),
                )
                return False
            await self._wait_for_navigation(url)

        return True

    async def _dismiss_overlays(self) -> None:
        await dismiss_overlays(self._session)

    async def _url_restore(self, target_url: str, reload: bool = False) -> bool:
        """
        Ensure the browser is at target_url before attempting an action.

        reload=True (used before each action):
          If already at the target URL, does a full page reload so each action
          starts from a clean DOM — no open dropdowns, no transient UI state
          carried over from the previous action.  Server-side mutations (created
          memos, etc.) are permanent and survive the reload, which is correct.

        reload=False (used at the top of each BFS node):
          Already-at-target is a no-op — we just navigated there and don't want
          an extra round-trip before capturing state.

        Returns False only when the app auth-guards the target URL and redirects
        us elsewhere (detected in ~300 ms rather than waiting for full networkidle).
        """
        target_norm = normalise_url(target_url)
        current_norm = normalise_url(self._session.page.url)

        if current_norm == target_norm:
            if not reload:
                return True
            # Reload to flush UI state from the previous action.
            ind = self._ind()
            logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"→ reloading  {target_norm}"))
            try:
                await self._session.page.goto(target_url, wait_until="load", timeout=10000)
                await self._session._wait_stable()
            except Exception:
                pass
            return True

        ind = self._ind()
        logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"→ going to  {target_norm}"))

        # Try go_back first (cheap; works when previous action navigated away).
        try:
            await self._session.page.go_back(timeout=3000)
            await self._session._wait_stable()
            if normalise_url(self._session.page.url) == target_norm:
                return True
        except Exception:
            pass

        # Direct navigate with domcontentloaded so auth redirects are caught fast.
        logger.info("BFS: %s%s %s", ind, _cyan("│"), _gray(f"→ navigating directly to  {target_norm}"))
        try:
            await self._session.page.goto(target_url, wait_until="load", timeout=10000)
        except Exception as e:
            logger.debug("BFS: url_restore goto failed: %s", e)
            return False

        # Brief pause for SPA auth redirect (~200 ms).
        await asyncio.sleep(0.3)
        actual = normalise_url(self._session.page.url)
        if actual != target_norm:
            logger.info(
                "BFS: %s%s %s",
                ind, _cyan("│"),
                _yellow(f"restore failed — {target_norm} redirected to {actual} (auth guard)"),
            )
            return False

        await self._session._wait_stable()
        return True

    # ------------------------------------------------------------------
    # Alternative-fill retry
    # ------------------------------------------------------------------

    async def _execute_action_with_overrides(
        self,
        action: SemanticAction,
        fill_overrides: dict[str, str],
    ) -> tuple[bool, str]:
        """Execute action with specific fill values substituted."""
        from copy import deepcopy
        patched = deepcopy(action)
        for step in patched.steps:
            if step.type == "fill" and step.selector in fill_overrides:
                step.value = fill_overrides[step.selector]
        success, reason, _ = await self._execute_action(patched)
        return success, reason

    async def _try_alternative_fills(
        self,
        action: SemanticAction,
        current_url: str,
        current_replay: list,
        visited_hashes: set,
        result,
        depth: int,
        parent_hash: str,
        error_hash: str,
        max_attempts: int = 4,
    ) -> None:
        """
        Called when an action with fill steps produced a same-URL state change
        (likely a form error). Retries up to max_attempts times, passing all
        previously tried fill sets to the LLM so it generates increasingly
        different values. Queues any new distinct state discovered.
        """
        if self._llm is None:
            return

        steps_dicts = [
            {"type": s.type, "selector": s.selector, "value": s.value}
            for s in action.steps
        ]
        previous_attempts: list[dict[str, str]] = []
        seen_result_hashes: set[str] = {error_hash}

        for attempt_num in range(1, max_attempts + 1):
            screenshot = await self._session.page.screenshot()

            self._box_llm_open(f"LLM: alt-fill attempt {attempt_num}/{max_attempts}")
            from harness.planner.prompts import suggest_alternative_fills

            alt_fills = await suggest_alternative_fills(
                self._llm, action.name, steps_dicts, screenshot,
                previous_attempts=previous_attempts,
            )
            self._box_llm_close()

            if not alt_fills:
                logger.info(
                    "BFS: %s%s %s", self._ind(), _cyan("│"),
                    _gray(f"alt-fill [{attempt_num}]: LLM returned no alternatives, stopping"),
                )
                return

            logger.info(
                "BFS: %s%s %s", self._ind(), _cyan("│"),
                _yellow(f"alt-fill [{attempt_num}]: retrying '{action.name}' with {alt_fills}"),
            )

            if not await self._restore_node(current_url, current_replay, reload=True):
                logger.info(
                    "BFS: %s%s %s", self._ind(), _cyan("│"),
                    _gray(f"alt-fill [{attempt_num}]: restore failed, stopping"),
                )
                return

            success, reason = await self._execute_action_with_overrides(action, alt_fills)
            if not success:
                logger.info(
                    "BFS: %s%s %s", self._ind(), _cyan("│"),
                    _gray(f"alt-fill [{attempt_num}]: execution failed — {reason.split(chr(10))[0][:80]}"),
                )
                previous_attempts.append(alt_fills)
                continue

            await self._wait_for_navigation(current_url)
            alt_state = await self._session.capture_state()
            alt_elements = await self._session.get_interactive_elements()
            alt_hash = state_hash(alt_state.url, alt_state.a11y_tree, alt_elements)
            alt_title = _page_title(alt_state.a11y_tree)

            previous_attempts.append(alt_fills)

            if alt_hash in seen_result_hashes:
                logger.info(
                    "BFS: %s%s %s", self._ind(), _cyan("│"),
                    _gray(f"alt-fill [{attempt_num}]: same error state again, trying next"),
                )
                seen_result_hashes.add(alt_hash)
                continue

            seen_result_hashes.add(alt_hash)

            if alt_hash in visited_hashes:
                logger.info(
                    "BFS: %s%s %s", self._ind(), _cyan("│"),
                    _gray(f"alt-fill [{attempt_num}]: reached already-known state {alt_hash[:8]}, stopping"),
                )
                return

            url_changed = normalise_url(alt_state.url) != normalise_url(current_url)
            alt_replay = [] if url_changed else list(current_replay) + [action]
            replay_hint = f" [replay:{len(alt_replay)}]" if alt_replay else ""
            logger.info(
                "BFS: %s%s %s", self._ind(), _cyan("│"),
                _green(f"alt-fill [{attempt_num}]: ✓ new state {alt_hash[:8]}  {alt_title[:40]!r}  → queued{replay_hint}"),
            )

            visited_hashes.add(alt_hash)
            result.nodes[alt_hash] = {
                "url":        alt_state.url,
                "screenshot": alt_state.screenshot,
                "a11y_tree":  alt_state.a11y_tree,
            }
            result.graph[parent_hash].append({
                "action":           f"{action.name}_alt{attempt_num}",
                "description":      f"Alternative fills (attempt {attempt_num}): {alt_fills}",
                "steps":            [
                    {"type": s.type, "selector": s.selector,
                     "value": alt_fills.get(s.selector, s.value) if s.type == "fill" else s.value}
                    for s in action.steps
                ],
                "expected_outcome": action.expected_outcome,
                "to_hash":          alt_hash,
                "queued":           True,
            })
            queue_ref = getattr(self, "_current_queue", None)
            if queue_ref is not None:
                queue_ref.append((alt_hash, alt_state.url, depth + 1, alt_replay))
            return

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def _execute_action(self, action: SemanticAction) -> tuple[bool, str, bool]:
        """Delegate to the shared step_runner (same logic, single source of truth)."""
        steps_dicts = [
            {"type": s.type, "selector": s.selector, "value": s.value or ""}
            for s in action.steps
        ]
        return await execute_steps(self._session, steps_dicts, self._llm)
