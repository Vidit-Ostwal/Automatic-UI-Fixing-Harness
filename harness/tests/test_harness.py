"""
Tests for run_harness.py

All tests mock Docker, browser, and LLM dependencies — nothing spawns real
processes or network connections.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harness.models import BugType, DetectedBy, Finding, Severity


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        depth=None,
        max_trajectories=None,
        output=None,
        planner_only=False,
        skip_planner=False,
        goals_only=False,
        run_goals=False,
        no_llm=False,
        verbose_bfs=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _critical_finding() -> Finding:
    return Finding(
        id="F-CRIT", title="Crash on submit", bug_type=BugType.LOGIC,
        severity=Severity.CRITICAL, steps=["click submit"],
        detected_by=DetectedBy.HEURISTIC, reasoning="pageerror",
    )


def _high_finding() -> Finding:
    return Finding(
        id="F-HIGH", title="Overlap detected", bug_type=BugType.VISUAL,
        severity=Severity.HIGH, steps=["viewport 375"],
        detected_by=DetectedBy.HEURISTIC, reasoning="geometry",
    )


def _fake_traj(n=3) -> list[dict]:
    return [{"id": f"T-{i:03d}", "steps": [], "description": f"traj {i}"} for i in range(n)]


def _fake_goals(n=3) -> list[dict]:
    return [
        {
            "id": f"T-{i:03d}",
            "goal": f"Verify workflow {i}",
            "instructions": [f"Step {i}"],
            "success_criteria": ["No errors"],
        }
        for i in range(n)
    ]


from run_harness import HarnessConfig, _env_int, _count_severities, _print_run_summary


def test_harness_config_defaults():
    args = _make_args()
    cfg = HarnessConfig(args)
    assert cfg.depth_n == 3
    assert cfg.max_trajectories == 20
    assert cfg.output_dir == Path("output")
    assert cfg.skip_planner is False
    assert cfg.no_llm is False


def test_harness_config_args_override_defaults():
    args = _make_args(depth=5, max_trajectories=7, output="/tmp/out")
    cfg = HarnessConfig(args)
    assert cfg.depth_n == 5
    assert cfg.max_trajectories == 7
    assert cfg.output_dir == Path("/tmp/out")


def test_harness_config_env_vars_override(monkeypatch):
    monkeypatch.setenv("DEPTH_N", "9")
    monkeypatch.setenv("MAX_TRAJECTORIES", "11")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/envout")
    cfg = HarnessConfig(_make_args())
    assert cfg.depth_n == 9
    assert cfg.max_trajectories == 11
    assert cfg.output_dir == Path("/tmp/envout")


def test_harness_config_run_id_is_unique():
    cfg1 = HarnessConfig(_make_args())
    cfg2 = HarnessConfig(_make_args())
    assert cfg1.run_id != cfg2.run_id


def test_env_int_falls_back_on_bad_value(monkeypatch):
    monkeypatch.setenv("DEPTH_N", "not-a-number")
    assert _env_int("DEPTH_N", 3) == 3


def test_count_severities_from_claims_and_bfs(tmp_path):
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps({
        "findings": [
            {"severity": "critical", "title": "A"},
            {"severity": "high", "title": "B"},
        ],
    }))
    bfs_low = Finding(
        id="L", title="low", bug_type=BugType.VISUAL, severity=Severity.LOW,
        steps=[], detected_by=DetectedBy.LLM, reasoning="",
    )
    counts = _count_severities([claims_path], [bfs_low])
    assert counts["critical"] == 1
    assert counts["high"] == 1
    assert counts["low"] == 1
    assert counts["total"] == 3


def test_print_run_summary_returns_critical_high_counts(tmp_path, capsys):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    cfg.run_id = "abc123"
    claims_path = tmp_path / "claims.json"
    claims_path.write_text(json.dumps({
        "findings": [{"severity": "critical", "title": "Crash"}],
    }))
    critical, high = _print_run_summary(cfg, [], [claims_path], None, 1.0)
    assert critical == 1
    assert high == 0
    assert "abc123" in capsys.readouterr().out


from run_harness import run_planner


@pytest.mark.asyncio
async def test_run_planner_writes_trajectories_json(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))

    with (
        patch("run_harness.DockerInstance") as MockDocker,
        patch("run_harness.BrowserSession") as MockBrowser,
        patch("run_harness.BFSExplorer") as MockExplorer,
        patch("run_harness.extract_trajectories") as mock_extract,
        patch("run_harness.trajectories_to_json") as mock_to_json,
    ):
        docker_cm = AsyncMock()
        docker_cm.__aenter__ = AsyncMock(return_value=MagicMock(url="http://localhost:9999"))
        docker_cm.__aexit__ = AsyncMock(return_value=False)
        MockDocker.start.return_value = docker_cm

        browser_cm = AsyncMock()
        browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        browser_cm.__aexit__ = AsyncMock(return_value=False)
        MockBrowser.create.return_value = browser_cm

        explorer_inst = MagicMock()
        explorer_inst.explore = AsyncMock(return_value=MagicMock(
            nodes={"h1": {"url": "/", "screenshot": b"png", "a11y_tree": {}}},
            graph={"h1": []},
            root_hash="h1",
            findings=[],
        ))
        MockExplorer.return_value = explorer_inst

        mock_extract.return_value = [MagicMock(id="T-001", steps=[])]
        mock_to_json.return_value = [{"id": "T-001", "steps": []}]

        result, findings = await run_planner(cfg, llm_oracle=None)

    assert (tmp_path / "trajectories.json").exists()
    assert result == [{"id": "T-001", "steps": []}]
    assert findings == []


@pytest.mark.asyncio
async def test_run_planner_returns_empty_list_when_no_trajectories(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))

    with (
        patch("run_harness.DockerInstance") as MockDocker,
        patch("run_harness.BrowserSession") as MockBrowser,
        patch("run_harness.BFSExplorer") as MockExplorer,
        patch("run_harness.extract_trajectories", return_value=[]),
        patch("run_harness.trajectories_to_json", return_value=[]),
    ):
        docker_cm = AsyncMock()
        docker_cm.__aenter__ = AsyncMock(return_value=MagicMock(url="http://localhost:9999"))
        docker_cm.__aexit__ = AsyncMock(return_value=False)
        MockDocker.start.return_value = docker_cm

        browser_cm = AsyncMock()
        browser_cm.__aenter__ = AsyncMock(return_value=MagicMock())
        browser_cm.__aexit__ = AsyncMock(return_value=False)
        MockBrowser.create.return_value = browser_cm

        explorer_inst = MagicMock()
        explorer_inst.explore = AsyncMock(return_value=MagicMock(
            nodes={}, graph={}, root_hash="h1", findings=[]
        ))
        MockExplorer.return_value = explorer_inst

        result, findings = await run_planner(cfg, llm_oracle=None)

    assert result == []
    assert findings == []


from harness.executor.goal_executor import ExecutorResult
from run_harness import run_goal_executors


def _executor_result(test_case_id: str, run_dir: Path) -> ExecutorResult:
    return ExecutorResult(
        test_case_id=test_case_id,
        run_id="run123",
        completed=True,
        steps_executed=1,
        steps_succeeded=1,
        final_state="success",
        run_dir=run_dir,
    )


@pytest.mark.asyncio
async def test_run_goal_executors_caps_at_max_trajectories(tmp_path):
    cfg = HarnessConfig(_make_args(max_trajectories=2, output=str(tmp_path)))
    goals = _fake_goals(5)
    run_dir = tmp_path / "executor_runs" / "T-000_run123"

    with (
        patch(
            "run_harness._run_one_goal",
            new=AsyncMock(side_effect=lambda g, *a, **k: _executor_result(g["id"], run_dir)),
        ) as mock_run,
        patch("run_harness._run_verifier_consumer", new=AsyncMock(return_value=[])),
    ):
        results, claims = await run_goal_executors(goals, cfg, llm_oracle=None)

    assert mock_run.call_count == 2
    assert len(results) == 2
    assert claims == []


@pytest.mark.asyncio
async def test_run_goal_executors_empty_goals(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    results, claims = await run_goal_executors([], cfg, llm_oracle=None)
    assert results == []
    assert claims == []


from run_harness import _main


@pytest.mark.asyncio
async def test_main_returns_0_on_clean_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner", new=AsyncMock(return_value=(_fake_traj(1), []))),
        patch("run_harness.run_goal_writer", new=AsyncMock(return_value=_fake_goals(1))),
        patch("run_harness.run_goal_executors", new=AsyncMock(return_value=([], []))),
    ):
        code = await _main()

    assert code == 0


@pytest.mark.asyncio
async def test_main_returns_1_on_critical_finding(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    claims_path = tmp_path / "verifier_claims" / "T-000_abc" / "claims.json"
    claims_path.parent.mkdir(parents=True)
    claims_path.write_text(json.dumps({
        "test_case_id": "T-000",
        "findings": [{
            "id": "V-1", "title": "Crash", "bug_type": "logic",
            "severity": "critical", "reproduction_steps": [],
            "description": "bad", "evidence": "crash",
        }],
    }))

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner", new=AsyncMock(return_value=(_fake_traj(1), []))),
        patch("run_harness.run_goal_writer", new=AsyncMock(return_value=_fake_goals(1))),
        patch("run_harness.run_goal_executors", new=AsyncMock(return_value=([], [claims_path]))),
    ):
        code = await _main()

    assert code == 1


@pytest.mark.asyncio
async def test_main_returns_2_when_skip_planner_file_missing(tmp_path):
    with patch("run_harness._parse_args", return_value=_make_args(
        skip_planner=True, no_llm=True, output=str(tmp_path)
    )):
        code = await _main()

    assert code == 2


@pytest.mark.asyncio
async def test_main_skip_planner_loads_from_file(tmp_path):
    trajectories = _fake_traj(2)
    (tmp_path / "trajectories.json").write_text(json.dumps(trajectories))

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            skip_planner=True, no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner") as mock_planner,
        patch("run_harness.run_goal_writer", new=AsyncMock(return_value=_fake_goals(2))),
        patch("run_harness.run_goal_executors", new=AsyncMock(return_value=([], []))),
    ):
        code = await _main()

    mock_planner.assert_not_called()
    assert code == 0


@pytest.mark.asyncio
async def test_main_no_llm_flag_disables_llm_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-key")

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner", new=AsyncMock(return_value=([], []))),
        patch("run_harness.run_goal_writer", new=AsyncMock(return_value=[])),
        patch("run_harness.run_goal_executors", new=AsyncMock(return_value=([], []))) as mock_exec,
    ):
        await _main()

    assert mock_exec.call_args[0][2] is None
