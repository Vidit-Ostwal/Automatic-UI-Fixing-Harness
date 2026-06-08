"""
Tests for run_harness.py

All tests mock Docker, browser, and LLM dependencies — nothing spawns real
processes or network connections.

Covers:
  - HarnessConfig reads CLI args and env-var defaults
  - _env_int falls back on bad values
  - run_planner writes trajectories.json and returns list
  - run_executors respects max_trajectories cap
  - run_executors returns one collector per trajectory
  - run_one_trajectory catches executor exceptions and returns empty collector
  - generate_report calls render_json and render_html
  - generate_report returns correct critical / high counts
  - _main returns exit code 1 on critical findings, 0 on clean run
  - _main returns exit code 2 when --skip-planner file is missing
  - _main --skip-planner loads trajectories.json without calling run_planner
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import BugType, DetectedBy, Finding, Severity
from reporter.collector import FindingCollector
from reporter.render import RunReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        depth=None,
        max_trajectories=None,
        output=None,
        planner_only=False,
        skip_planner=False,
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


def _low_finding() -> Finding:
    return Finding(
        id="F-LOW", title="Overflow x", bug_type=BugType.VISUAL,
        severity=Severity.LOW, steps=[],
        detected_by=DetectedBy.HEURISTIC, reasoning="overflow",
    )


def _fake_traj(n=3) -> list[dict]:
    return [{"id": f"T-{i:03d}", "steps": [], "description": f"traj {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# HarnessConfig
# ---------------------------------------------------------------------------

from run_harness import HarnessConfig, _env_int


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
    monkeypatch.setenv("DEPTH_N", "4")
    monkeypatch.setenv("MAX_TRAJECTORIES", "10")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/env_out")
    args = _make_args()
    cfg = HarnessConfig(args)
    assert cfg.depth_n == 4
    assert cfg.max_trajectories == 10


def test_harness_config_run_id_is_unique():
    cfg1 = HarnessConfig(_make_args())
    cfg2 = HarnessConfig(_make_args())
    assert cfg1.run_id != cfg2.run_id


def test_env_int_returns_default_on_bad_value(monkeypatch):
    monkeypatch.setenv("DEPTH_N", "not-a-number")
    assert _env_int("DEPTH_N", 3) == 3


def test_env_int_returns_int_on_valid_value(monkeypatch):
    monkeypatch.setenv("DEPTH_N", "7")
    assert _env_int("DEPTH_N", 3) == 7


def test_env_int_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("DEPTH_N", raising=False)
    assert _env_int("DEPTH_N", 3) == 3


# ---------------------------------------------------------------------------
# run_planner
# ---------------------------------------------------------------------------

from run_harness import run_planner


@pytest.mark.asyncio
async def test_run_planner_writes_trajectories_json(tmp_path):
    """run_planner should persist trajectories.json inside output_dir."""
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    trajectories = _fake_traj(2)

    with (
        patch("run_harness.DockerInstance") as MockDocker,
        patch("run_harness.BrowserSession") as MockBrowser,
        patch("run_harness.BFSExplorer") as MockExplorer,
        patch("run_harness.extract_trajectories") as mock_extract,
        patch("run_harness.trajectories_to_json", return_value=trajectories),
    ):
        # Set up docker context manager.
        docker_cm = AsyncMock()
        docker_cm.__aenter__ = AsyncMock(return_value=MagicMock(url="http://localhost:9999"))
        docker_cm.__aexit__ = AsyncMock(return_value=False)
        MockDocker.start.return_value = docker_cm

        # Set up browser context manager.
        browser_cm = AsyncMock()
        session = MagicMock()
        browser_cm.__aenter__ = AsyncMock(return_value=session)
        browser_cm.__aexit__ = AsyncMock(return_value=False)
        MockBrowser.create.return_value = browser_cm

        # Set up BFS explorer.
        fake_result = MagicMock()
        fake_result.nodes = {"h1": {}, "h2": {}}
        fake_result.graph = {"h1": ["h2"]}
        fake_result.root_hash = "h1"
        explorer_inst = MagicMock()
        explorer_inst.explore = AsyncMock(return_value=fake_result)
        MockExplorer.return_value = explorer_inst

        mock_extract.return_value = []

        result, bfs_collector = await run_planner(cfg, llm_oracle=None)

    traj_file = tmp_path / "trajectories.json"
    assert traj_file.exists(), "trajectories.json should be written"
    saved = json.loads(traj_file.read_text())
    assert saved == trajectories
    assert result == trajectories


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

        result, bfs_collector = await run_planner(cfg, llm_oracle=None)

    assert result == []


# ---------------------------------------------------------------------------
# run_executors
# ---------------------------------------------------------------------------

from run_harness import run_executors, run_one_trajectory


@pytest.mark.asyncio
async def test_run_executors_caps_at_max_trajectories(tmp_path):
    cfg = HarnessConfig(_make_args(max_trajectories=2, output=str(tmp_path)))
    trajectories = _fake_traj(5)
    called_ids = []

    async def fake_one(traj, llm_oracle, sem):
        called_ids.append(traj["id"])
        return FindingCollector(trajectory_id=traj["id"])

    with patch("run_harness.run_one_trajectory", side_effect=fake_one):
        collectors = await run_executors(trajectories, cfg, llm_oracle=None)

    assert len(collectors) == 2
    assert len(called_ids) == 2


@pytest.mark.asyncio
async def test_run_executors_returns_one_collector_per_trajectory(tmp_path):
    cfg = HarnessConfig(_make_args(max_trajectories=3, output=str(tmp_path)))
    trajectories = _fake_traj(3)

    async def fake_one(traj, llm_oracle, sem):
        return FindingCollector(trajectory_id=traj["id"])

    with patch("run_harness.run_one_trajectory", side_effect=fake_one):
        collectors = await run_executors(trajectories, cfg, llm_oracle=None)

    assert len(collectors) == 3


@pytest.mark.asyncio
async def test_run_executors_empty_trajectories(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    collectors = await run_executors([], cfg, llm_oracle=None)
    assert collectors == []


# ---------------------------------------------------------------------------
# run_one_trajectory — exception safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_one_trajectory_catches_exceptions(tmp_path):
    """A crashing executor must return an empty FindingCollector, not raise."""
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    sem = asyncio.Semaphore(1)
    traj = {"id": "T-ERR", "steps": []}

    with patch("run_harness.DockerInstance") as MockDocker:
        docker_cm = AsyncMock()
        docker_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("docker boom"))
        docker_cm.__aexit__ = AsyncMock(return_value=False)
        MockDocker.start.return_value = docker_cm

        collector = await run_one_trajectory(traj, llm_oracle=None, semaphore=sem)

    assert isinstance(collector, FindingCollector)
    assert len(collector) == 0


@pytest.mark.asyncio
async def test_run_one_trajectory_success_returns_collector(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    sem = asyncio.Semaphore(1)
    traj = {"id": "T-OK", "steps": []}

    expected_collector = FindingCollector(trajectory_id="T-OK")

    with (
        patch("run_harness.DockerInstance") as MockDocker,
        patch("run_harness.TrajectoryRunner") as MockRunner,
    ):
        docker_cm = AsyncMock()
        docker_cm.__aenter__ = AsyncMock(return_value=MagicMock(url="http://localhost:9000"))
        docker_cm.__aexit__ = AsyncMock(return_value=False)
        MockDocker.start.return_value = docker_cm

        runner_inst = MagicMock()
        runner_inst.run = AsyncMock(return_value=expected_collector)
        MockRunner.return_value = runner_inst

        collector = await run_one_trajectory(traj, llm_oracle=None, semaphore=sem)

    assert collector is expected_collector


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

from run_harness import generate_report


def test_generate_report_calls_render_json_and_html(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    cfg.run_id = "abc123"
    collectors = [FindingCollector()]

    with (
        patch("run_harness.merge", return_value=([], [])) as mock_merge,
        patch("run_harness.render_json", return_value=tmp_path / "report.json") as mock_json,
        patch("run_harness.render_html", return_value=tmp_path / "report.html") as mock_html,
    ):
        generate_report(collectors, cfg, trajectories=_fake_traj(2), duration=5.0)

    mock_json.assert_called_once()
    mock_html.assert_called_once()


def test_generate_report_returns_critical_high_counts(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    cfg.run_id = "abc123"

    c = FindingCollector()
    c.add(_critical_finding())
    c.add(_high_finding())
    c.add(_low_finding())

    with (
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
    ):
        critical, high = generate_report([c], cfg, trajectories=[], duration=1.0)

    assert critical == 1
    assert high == 1


def test_generate_report_zero_counts_on_no_findings(tmp_path):
    cfg = HarnessConfig(_make_args(output=str(tmp_path)))
    cfg.run_id = "abc123"

    with (
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
    ):
        critical, high = generate_report([FindingCollector()], cfg, trajectories=[], duration=0.5)

    assert critical == 0
    assert high == 0


# ---------------------------------------------------------------------------
# _main integration
# ---------------------------------------------------------------------------

from run_harness import _main


@pytest.mark.asyncio
async def test_main_returns_0_on_clean_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner", new=AsyncMock(return_value=(_fake_traj(1), FindingCollector()))),
        patch("run_harness.run_executors", new=AsyncMock(return_value=[FindingCollector()])),
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
    ):
        code = await _main()

    assert code == 0


@pytest.mark.asyncio
async def test_main_returns_1_on_critical_finding(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))

    c = FindingCollector()
    c.add(_critical_finding())

    with (
        patch("run_harness._parse_args", return_value=_make_args(
            no_llm=True, output=str(tmp_path)
        )),
        patch("run_harness.run_planner", new=AsyncMock(return_value=(_fake_traj(1), FindingCollector()))),
        patch("run_harness.run_executors", new=AsyncMock(return_value=[c])),
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
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
        patch("run_harness.run_executors", new=AsyncMock(return_value=[FindingCollector()])),
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
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
        patch("run_harness.run_planner", new=AsyncMock(return_value=([], FindingCollector()))),
        patch("run_harness.run_executors", new=AsyncMock(return_value=[])) as mock_exec,
        patch("run_harness.render_json", return_value=tmp_path / "report.json"),
        patch("run_harness.render_html", return_value=tmp_path / "report.html"),
    ):
        await _main()

    # llm_oracle argument should be None when --no-llm
    _, kwargs = mock_exec.call_args
    assert kwargs.get("llm_oracle") is None or mock_exec.call_args[0][2] is None
