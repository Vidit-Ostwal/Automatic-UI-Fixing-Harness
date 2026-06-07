"""
Trajectory extractor — converts a BFS state graph into a flat list of
test trajectories (root-to-leaf paths).

Each trajectory is a sequence of (action_name, state_hash) pairs that
an executor can replay against a fresh Docker instance.
"""

from dataclasses import dataclass, field


@dataclass
class Trajectory:
    id: str
    description: str
    steps: list[dict]          # [{"action": str, "from_hash": str, "to_hash": str}]
    leaf_hash: str


def extract_trajectories(
    graph: dict[str, list[dict]],
    root_hash: str,
) -> list[Trajectory]:
    """
    DFS over the state graph to collect every root-to-leaf path.

    graph  — {state_hash: [{"action": str, "selector": str, "to_hash": str}, ...]}
    root_hash — hash of the starting state

    A node is a leaf when it has no outgoing edges OR all its children
    have already appeared in the current path (cycle guard).
    """
    trajectories: list[Trajectory] = []
    counter = [0]

    def dfs(current_hash: str, path: list[dict], visited_in_path: set[str]) -> None:
        edges = graph.get(current_hash, [])
        unvisited_edges = [e for e in edges if e["to_hash"] not in visited_in_path]

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
                "action": edge["action"],
                "selector": edge.get("selector", ""),
                "from_hash": current_hash,
                "to_hash": edge["to_hash"],
            })
            dfs(edge["to_hash"], path, visited_in_path | {edge["to_hash"]})
            path.pop()

    dfs(root_hash, [], {root_hash})
    return trajectories


def trajectories_to_json(trajectories: list[Trajectory]) -> list[dict]:
    """Serialise trajectories to a JSON-compatible list of dicts."""
    return [
        {
            "id": t.id,
            "description": t.description,
            "steps": t.steps,
            "leaf_hash": t.leaf_hash,
        }
        for t in trajectories
    ]
