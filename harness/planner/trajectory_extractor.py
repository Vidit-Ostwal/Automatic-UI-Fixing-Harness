"""
Trajectory extractor — converts a BFS state graph into a flat list of
test trajectories (root-to-leaf paths).

Each trajectory is a sequence of (action_name, state_hash) pairs that
an executor can replay against a fresh Docker instance.

Same-hash edges (UI/data changed but BFS did not re-queue the state) are
still included as trajectory steps — one pass per action name at that hash.
"""

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    id: str
    description: str
    steps: list[dict]          # [{"action": str, "from_hash": str, "to_hash": str}]
    leaf_hash: str


def _can_traverse_edge(
    edge: dict,
    current_hash: str,
    visited_in_path: set[str],
    actions_at_hash: dict[str, set[str]],
) -> bool:
    """
    Decide whether an edge can extend the current DFS path.

    - New hash not yet in path → allowed (normal forward edge).
    - Same hash (mutation / visual change, not re-explored) → allowed once
      per action name at this hash so executors still replay the step.
    - Return to an earlier hash in the path → blocked (cycle).
    """
    to_hash = edge["to_hash"]
    action = edge["action"]

    if to_hash == current_hash:
        return action not in actions_at_hash.get(current_hash, set())

    return to_hash not in visited_in_path


def extract_trajectories(
    graph: dict[str, list[dict]],
    root_hash: str,
) -> list[Trajectory]:
    """
    DFS over the state graph to collect every root-to-leaf path.

    graph  — {state_hash: [{"action": str, "selector": str, "to_hash": str}, ...]}
    root_hash — hash of the starting state

    A node is a leaf when it has no outgoing edges OR all its children
    have already appeared in the current path (cycle guard), except that
    same-hash mutation edges remain traversable once per action.
    """
    trajectories: list[Trajectory] = []
    counter = [0]

    def dfs(
        current_hash: str,
        path: list[dict],
        visited_in_path: set[str],
        actions_at_hash: dict[str, set[str]],
    ) -> None:
        edges = graph.get(current_hash, [])
        unvisited_edges = [
            e for e in edges
            if _can_traverse_edge(e, current_hash, visited_in_path, actions_at_hash)
        ]

        if not unvisited_edges:
            # Leaf node — record this path as a trajectory.
            if path:
                counter[0] += 1
                traj_id = f"T-{counter[0]:03d}"
                description = " → ".join(s["action"] for s in path)
                trajectories.append(
                    Trajectory(
                        id=traj_id,
                        description=description,
                        steps=list(path),
                        leaf_hash=current_hash,
                    )
                )
            return

        for edge in unvisited_edges:
            path.append({
                "action":      edge["action"],
                "description": edge.get("description", ""),
                "steps":       edge.get("steps", []),
                "from_hash":   current_hash,
                "to_hash":     edge["to_hash"],
                "queued":      edge.get("queued", True),
            })
            next_actions = {h: set(names) for h, names in actions_at_hash.items()}
            next_actions.setdefault(current_hash, set()).add(edge["action"])

            if edge["to_hash"] == current_hash:
                dfs(current_hash, path, visited_in_path, next_actions)
            else:
                dfs(
                    edge["to_hash"],
                    path,
                    visited_in_path | {edge["to_hash"]},
                    next_actions,
                )
            path.pop()

    dfs(root_hash, [], {root_hash}, {})
    return trajectories


def trajectories_to_json(
    trajectories: list[Trajectory],
    screenshots_root: str = "",
) -> list[dict]:
    """
    Serialise trajectories to a JSON-compatible list of dicts.

    If screenshots_root is provided, each trajectory gains a
    'screenshots_folder' key pointing to its per-trajectory folder, and
    each step gains 'screenshot_before' / 'screenshot_after' keys with
    the paths of the sequentially-named PNGs written there.

    Folder layout (written by _save_trajectory_screenshots in run_harness):
      screenshots_root/T-001/
        00_start.png
        01_<action>.png
        02_<action>.png
        ...
    """
    result = []
    for t in trajectories:
        traj_id = t.id

        steps = []
        screenshots: list[str] = []
        for i, s in enumerate(t.steps, start=1):
            step = dict(s)
            if screenshots_root:
                folder      = f"screenshots/{traj_id}"
                action      = s.get("action", "step")[:40].replace("/", "_").replace(" ", "_")
                prev_action = "start" if i == 1 else t.steps[i - 2].get("action", "step")[:40].replace("/", "_").replace(" ", "_")
                before = f"{folder}/{i-1:02d}_{prev_action}.png"
                after  = f"{folder}/{i:02d}_{action}.png"
                step["screenshot_before"] = before
                step["screenshot_after"]  = after
                if i == 1:
                    screenshots.append(before)
                screenshots.append(after)
            steps.append(step)

        entry: dict = {
            "id":          traj_id,
            "description": t.description,
            "steps":       steps,
            "leaf_hash":   t.leaf_hash,
        }
        if screenshots_root:
            entry["screenshots_folder"] = f"screenshots/{traj_id}"
            entry["screenshots"]        = screenshots
        result.append(entry)
    return result
