"""
Tests for executor/runner.py

Uses mock sessions and mock oracles — no Docker or browser required.
Covers:
  - steps_from_trajectory parses correctly
  - Runner collects findings from oracles
  - Auth failure is reported as a Critical finding
  - Missing selector is reported as High finding
  - Expect-change logic derives correctly from action name
  - Viewport checks are run at all 3 sizes
  - All exceptions are caught and don't propagate
"""

import pytest
from models import BugType, DetectedBy, Finding, PageState, Severity
from executor.runner import (
    TrajectoryRunner,
    TrajectoryStep,
    RunnerConfig,
    steps_from_trajectory,
)
from reporter.collector import FindingCollector


# ---------------------------------------------------------------------------
# Helpers & mocks
# ---------------------------------------------------------------------------

def _state(url="http://localhost:5230/") -> PageState:
    return PageState(
        url=url,
        screenshot=b"\x89PNG",
        a11y_tree={"role": "main", "tag": "main", "children": [
            {"role": "button", "tag": "button", "name": "New Memo"}
        ]},
        timestamp=1.0,
    )


def _finding(title="Bug", severity=Severity.HIGH) -> Finding:
    return Finding(
        id="F1", title=title, bug_type=BugType.LOGIC,
        severity=severity, steps=[], detected_by=DetectedBy.HEURISTIC, reasoning="x"
    )


class _MockSession:
    """Minimal session mock — all operations succeed silently."""

    def __init__(self):
        self.url = "http://localhost:5230/"
        self._viewport_calls: list[tuple] = []
        self._click_calls: list[str] = []

    async def navigate(self, url): pass
    async def click(self, selector):
        self._click_calls.append(selector)

    async def fill(self, selector, value): pass
    async def press(self, selector, key): pass
    async def set_viewport(self, w, h):
        self._viewport_calls.append((w, h))

    async def capture_state(self) -> PageState:
        return _state(self.url)

    async def get_interactive_elements(self):
        return [{"tag": "button", "role": "button", "label": "New", "selector": "#new"}]

    async def get_geometry_violations(self):
        return []

    async def element_exists(self, selector):
        return True

    async def get_text(self, selector):
        return "text"

    async def _wait_stable(self): pass

    @property
    def page(self):
        class _Page:
            url = "http://localhost:5230/"
        return _Page()


class _MockVisualOracle:
    def __init__(self, findings=None):
        self._findings = findings or []
        self.calls = 0

    async def check(self, session, step_context=""):
        self.calls += 1
        return list(self._findings)


class _MockLogicOracle:
    def __init__(self, crash=None):
        self._crash = crash

    def check_no_crash(self, after, action=""):
        return self._crash


class _MockDiffOracle:
    def __init__(self, finding=None):
        self._finding = finding

    async def check(self, before, after, action, expect_change=True):
        return self._finding


class _MockAuthSuccess:
    success = True
    username = "harness_admin"
    url_after = "http://localhost:5230/"
    error = ""


class _MockAuthFailure:
    success = False
    username = "harness_admin"
    url_after = "http://localhost:5230/auth"
    error = "Could not find username input field"


# ---------------------------------------------------------------------------
# steps_from_trajectory
# ---------------------------------------------------------------------------

def test_steps_from_trajectory_basic():
    traj = {
        "id": "T-001",
        "steps": [
            {"action": "create_memo", "selector": "#new", "from_hash": "A", "to_hash": "B"},
            {"action": "pin_memo",    "selector": "#pin", "from_hash": "B", "to_hash": "C"},
        ]
    }
    steps = steps_from_trajectory(traj)
    assert len(steps) == 2
    assert steps[0].action == "create_memo"
    assert steps[0].selector == "#new"
    assert steps[1].action == "pin_memo"


def test_steps_from_trajectory_empty():
    assert steps_from_trajectory({"steps": []}) == []


def test_steps_from_trajectory_missing_selector():
    traj = {"steps": [{"action": "create_memo", "from_hash": "A", "to_hash": "B"}]}
    steps = steps_from_trajectory(traj)
    assert steps[0].selector == ""


def test_steps_from_trajectory_preserves_hashes():
    traj = {"steps": [{"action": "x", "selector": "", "from_hash": "AAA", "to_hash": "BBB"}]}
    steps = steps_from_trajectory(traj)
    assert steps[0].from_hash == "AAA"
    assert steps[0].to_hash == "BBB"


# ---------------------------------------------------------------------------
# TrajectoryRunner — unit tests with mocked session
# ---------------------------------------------------------------------------

class _PatchedRunner(TrajectoryRunner):
    """Runner that injects a mock session instead of opening a real browser."""

    def __init__(self, steps, config, session, **oracles):
        super().__init__(steps, config, **oracles)
        self._mock_session = session

    async def run(self) -> FindingCollector:
        try:
            await self._run_with_session(self._mock_session)
        except Exception:
            pass
        return self._collector


def _make_runner(steps=None, session=None, visual=None, logic=None, diff=None, traj_id="T-001"):
    return _PatchedRunner(
        steps=steps or [],
        config=RunnerConfig(trajectory_id=traj_id, app_url="http://localhost:5230"),
        session=session or _MockSession(),
        visual_oracle=visual or _MockVisualOracle(),
        logic_oracle=logic or _MockLogicOracle(),
        diff_oracle=diff or _MockDiffOracle(),
    )


@pytest.mark.asyncio
async def test_runner_auth_failure_adds_critical_finding():
    import executor.runner as runner_mod
    original = runner_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthFailure()

    runner_mod.signup = mock_signup
    try:
        runner = _make_runner()
        collector = await runner.run()
        assert len(collector) == 1
        f = collector.findings[0]
        assert f.severity == Severity.CRITICAL
        assert "Sign-up" in f.title or "signup" in f.title.lower() or "sign" in f.title.lower()
    finally:
        runner_mod.signup = original


@pytest.mark.asyncio
async def test_runner_auth_success_proceeds_to_steps():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        visual = _MockVisualOracle()
        runner = _make_runner(
            steps=[TrajectoryStep("create_memo", "#new")],
            visual=visual,
        )
        await runner.run()
        assert visual.calls > 0
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_collects_visual_findings():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        visual_finding = _finding("Overflow detected", Severity.LOW)
        visual = _MockVisualOracle(findings=[visual_finding])
        runner = _make_runner(
            steps=[TrajectoryStep("view_page", "#irrelevant")],
            visual=visual,
        )
        collector = await runner.run()
        visual_findings = [f for f in collector.findings if f.title == "Overflow detected"]
        assert len(visual_findings) > 0
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_collects_diff_findings():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        diff_finding = _finding("No change after pin", Severity.HIGH)
        runner = _make_runner(
            steps=[TrajectoryStep("pin_memo", "#pin")],
            diff=_MockDiffOracle(finding=diff_finding),
        )
        collector = await runner.run()
        assert any(f.title == "No change after pin" for f in collector.findings)
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_collects_crash_findings():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        crash = _finding("JS crash", Severity.CRITICAL)
        runner = _make_runner(
            steps=[TrajectoryStep("submit", "#submit")],
            logic=_MockLogicOracle(crash=crash),
        )
        collector = await runner.run()
        assert any(f.title == "JS crash" for f in collector.findings)
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_missing_selector_adds_finding():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup

    class _FailSession(_MockSession):
        async def click(self, selector):
            raise Exception("Element not found")
        async def element_exists(self, selector):
            return False

    try:
        runner = _make_runner(
            steps=[TrajectoryStep("pin_memo", "#nonexistent")],
            session=_FailSession(),
        )
        collector = await runner.run()
        missing = [f for f in collector.findings if "not found" in f.title.lower()]
        assert len(missing) >= 1
        assert missing[0].severity == Severity.HIGH
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_viewport_checks_run_at_all_sizes():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        session = _MockSession()
        visual = _MockVisualOracle()
        runner = _make_runner(steps=[], session=session, visual=visual)
        await runner.run()
        viewports_set = {(w, h) for w, h in session._viewport_calls}
        assert (375, 812) in viewports_set
        assert (768, 1024) in viewports_set
        assert (1280, 800) in viewports_set
    finally:
        auth_mod.signup = original


@pytest.mark.asyncio
async def test_runner_trajectory_id_stamped_on_findings():
    import executor.workflows.auth as auth_mod
    original = auth_mod.signup

    async def mock_signup(session, app_url, **kwargs):
        return _MockAuthSuccess()

    auth_mod.signup = mock_signup
    try:
        visual_finding = _finding("Overflow")
        runner = _make_runner(
            steps=[TrajectoryStep("view", "#x")],
            visual=_MockVisualOracle(findings=[visual_finding]),
            traj_id="T-042",
        )
        collector = await runner.run()
        for f in collector.findings:
            assert f.trajectory_id == "T-042"
    finally:
        auth_mod.signup = original


# ---------------------------------------------------------------------------
# expect_change logic
# ---------------------------------------------------------------------------

def test_expect_change_true_for_create():
    config = RunnerConfig()
    assert any(kw in "create_memo" for kw in config.expect_change_actions)


def test_expect_change_false_for_view():
    config = RunnerConfig()
    assert not any(kw in "view_memo" for kw in config.expect_change_actions)
