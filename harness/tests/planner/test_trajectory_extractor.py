"""
Tests for planner/trajectory_extractor.py

All tests are pure unit tests — no browser required.
Uses hardcoded state graphs to verify path extraction logic.
"""

import pytest
from harness.planner.trajectory_extractor import (
    Trajectory,
    extract_trajectories,
    trajectories_to_json,
)


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------

# Linear graph: root → A → B  (single path)
LINEAR_GRAPH = {
    "root": [{"action": "click_login", "selector": "#login", "to_hash": "A"}],
    "A":    [{"action": "submit_form", "selector": "#submit", "to_hash": "B"}],
    "B":    [],
}

# Branching graph: root → A and root → B  (two paths)
BRANCHING_GRAPH = {
    "root": [
        {"action": "create_memo", "selector": "#new", "to_hash": "A"},
        {"action": "open_search", "selector": "#search", "to_hash": "B"},
    ],
    "A": [],
    "B": [],
}

# Deep graph: root → A → C and root → B → D  (two paths, two levels deep)
DEEP_GRAPH = {
    "root": [
        {"action": "go_home",    "selector": "#home",   "to_hash": "A"},
        {"action": "go_explore", "selector": "#explore","to_hash": "B"},
    ],
    "A": [{"action": "create_memo", "selector": "#new", "to_hash": "C"}],
    "B": [{"action": "search",      "selector": "#q",   "to_hash": "D"}],
    "C": [],
    "D": [],
}

# Cyclic graph: root → A → root  (cycle must not produce infinite loop)
CYCLIC_GRAPH = {
    "root": [{"action": "open_modal", "selector": "#modal", "to_hash": "A"}],
    "A":    [{"action": "close_modal","selector": "#close", "to_hash": "root"}],
}

# Graph with self-loop: root → root
SELF_LOOP_GRAPH = {
    "root": [{"action": "refresh", "selector": "#refresh", "to_hash": "root"}],
}

# Empty graph (only root, no edges)
EMPTY_GRAPH: dict = {"root": []}

# Disconnected graph: root → A, but B has no incoming edges
DISCONNECTED_GRAPH = {
    "root": [{"action": "go_A", "selector": "#a", "to_hash": "A"}],
    "A": [],
    "B": [{"action": "go_C", "selector": "#c", "to_hash": "C"}],
    "C": [],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_linear_graph_one_trajectory():
    trajs = extract_trajectories(LINEAR_GRAPH, "root")
    assert len(trajs) == 1
    assert trajs[0].steps[0]["action"] == "click_login"
    assert trajs[0].steps[1]["action"] == "submit_form"


def test_linear_graph_leaf_hash():
    trajs = extract_trajectories(LINEAR_GRAPH, "root")
    assert trajs[0].leaf_hash == "B"


def test_branching_graph_two_trajectories():
    trajs = extract_trajectories(BRANCHING_GRAPH, "root")
    assert len(trajs) == 2
    actions = {t.steps[0]["action"] for t in trajs}
    assert actions == {"create_memo", "open_search"}


def test_deep_graph_two_trajectories_two_steps_each():
    trajs = extract_trajectories(DEEP_GRAPH, "root")
    assert len(trajs) == 2
    for t in trajs:
        assert len(t.steps) == 2


def test_cyclic_graph_no_infinite_loop():
    """Cycle detection must prevent infinite recursion."""
    trajs = extract_trajectories(CYCLIC_GRAPH, "root")
    # root→A is a valid path; A→root is a back-edge (cycle) so it's a leaf.
    assert len(trajs) >= 1
    for t in trajs:
        assert len(t.steps) <= 2


def test_self_loop_graph_terminates():
    trajs = extract_trajectories(SELF_LOOP_GRAPH, "root")
    assert len(trajs) == 1
    assert trajs[0].steps[0]["action"] == "refresh"
    assert trajs[0].steps[0]["from_hash"] == trajs[0].steps[0]["to_hash"]


def test_same_hash_mutation_included_in_trajectory():
    """Actions that stay on the same hash (not re-queued) still become steps."""
    graph = {
        "root": [
            {"action": "create_memo", "selector": "#new", "to_hash": "root", "queued": False},
        ],
    }
    trajs = extract_trajectories(graph, "root")
    assert len(trajs) == 1
    assert trajs[0].steps[0]["action"] == "create_memo"
    assert trajs[0].steps[0]["to_hash"] == "root"
    assert trajs[0].steps[0]["queued"] is False


def test_empty_graph_no_trajectories():
    """A graph with only the root and no edges → no trajectories."""
    trajs = extract_trajectories(EMPTY_GRAPH, "root")
    assert trajs == []


def test_disconnected_graph_only_reachable_paths():
    """Unreachable nodes (B, C) must not appear in trajectories."""
    trajs = extract_trajectories(DISCONNECTED_GRAPH, "root")
    all_hashes = {h for t in trajs for s in t.steps for h in (s["from_hash"], s["to_hash"])}
    assert "B" not in all_hashes
    assert "C" not in all_hashes


def test_trajectory_ids_are_unique():
    trajs = extract_trajectories(DEEP_GRAPH, "root")
    ids = [t.id for t in trajs]
    assert len(ids) == len(set(ids))


def test_trajectory_description_reflects_actions():
    trajs = extract_trajectories(LINEAR_GRAPH, "root")
    desc = trajs[0].description
    assert "click_login" in desc
    assert "submit_form" in desc


def test_trajectory_steps_have_required_fields():
    trajs = extract_trajectories(DEEP_GRAPH, "root")
    for traj in trajs:
        for step in traj.steps:
            assert "action" in step
            assert "from_hash" in step
            assert "to_hash" in step
            assert "steps" in step


def test_trajectories_to_json_serialisable():
    trajs = extract_trajectories(DEEP_GRAPH, "root")
    import json
    result = trajectories_to_json(trajs)
    # Should not raise.
    serialised = json.dumps(result)
    parsed = json.loads(serialised)
    assert len(parsed) == len(trajs)


def test_trajectories_to_json_structure():
    trajs = extract_trajectories(LINEAR_GRAPH, "root")
    result = trajectories_to_json(trajs)
    assert result[0]["id"] == trajs[0].id
    assert result[0]["description"] == trajs[0].description
    assert result[0]["leaf_hash"] == trajs[0].leaf_hash
    assert isinstance(result[0]["steps"], list)
