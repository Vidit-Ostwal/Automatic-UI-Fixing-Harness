"""
Tests for docker/manager.py

All tests mock subprocess calls — no Docker daemon required.
Covers:
  - find_free_port returns a valid port number
  - compose file written with correct port and container name
  - DockerInstance.url is derived from port
  - Multiple instances get different ports
  - _run helper returns correct returncode/stdout/stderr
  - _wait_healthy times out and raises TimeoutError
  - _wait_healthy succeeds when endpoint returns 200
"""

import asyncio
import socket
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import httpx

from harness.docker import DockerInstance, CONTAINER_PORT, IMAGE_NAME, find_free_port
from harness.docker.manager import _write_compose_file, _run, _wait_healthy


# ---------------------------------------------------------------------------
# find_free_port
# ---------------------------------------------------------------------------

def test_find_free_port_returns_integer():
    port = find_free_port()
    assert isinstance(port, int)


def test_find_free_port_in_valid_range():
    port = find_free_port()
    assert 1024 <= port <= 65535


def test_find_free_port_is_actually_free():
    port = find_free_port()
    # Should be able to bind to this port immediately.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))


def test_find_free_port_returns_different_ports():
    """Two consecutive calls should (almost certainly) return different ports."""
    ports = {find_free_port() for _ in range(5)}
    # At least 3 of 5 should be distinct (very conservative bound).
    assert len(ports) >= 3


# ---------------------------------------------------------------------------
# _write_compose_file
# ---------------------------------------------------------------------------

def test_write_compose_file_creates_file(tmp_path):
    path = _write_compose_file(54321, "memos-buggy-abc", str(tmp_path))
    assert path.exists()
    assert path.name == "docker-compose.yml"


def test_write_compose_file_contains_host_port(tmp_path):
    _write_compose_file(54321, "memos-buggy-abc", str(tmp_path))
    content = (tmp_path / "docker-compose.yml").read_text()
    assert "54321" in content


def test_write_compose_file_contains_container_port(tmp_path):
    _write_compose_file(54321, "memos-buggy-abc", str(tmp_path))
    content = (tmp_path / "docker-compose.yml").read_text()
    assert str(CONTAINER_PORT) in content


def test_write_compose_file_contains_container_name(tmp_path):
    _write_compose_file(54321, "memos-buggy-test", str(tmp_path))
    content = (tmp_path / "docker-compose.yml").read_text()
    assert "memos-buggy-test" in content


def test_write_compose_file_contains_image_name(tmp_path):
    _write_compose_file(54321, "memos-buggy-abc", str(tmp_path))
    content = (tmp_path / "docker-compose.yml").read_text()
    assert IMAGE_NAME in content


def test_write_compose_file_different_ports_for_two_instances(tmp_path):
    p1 = tmp_path / "inst1"
    p2 = tmp_path / "inst2"
    p1.mkdir(); p2.mkdir()
    _write_compose_file(10001, "memos-buggy-aaa", str(p1))
    _write_compose_file(10002, "memos-buggy-bbb", str(p2))
    content1 = (p1 / "docker-compose.yml").read_text()
    content2 = (p2 / "docker-compose.yml").read_text()
    assert "10001" in content1 and "10001" not in content2
    assert "10002" in content2 and "10002" not in content1
    assert "memos-buggy-aaa" in content1
    assert "memos-buggy-bbb" in content2


# ---------------------------------------------------------------------------
# _run
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_zero_on_success():
    code, out, err = await _run(["echo", "hello"])
    assert code == 0
    assert "hello" in out


@pytest.mark.asyncio
async def test_run_returns_nonzero_on_failure():
    code, out, err = await _run(["false"])
    assert code != 0


@pytest.mark.asyncio
async def test_run_captures_stdout():
    code, out, err = await _run(["echo", "test-output"])
    assert "test-output" in out


@pytest.mark.asyncio
async def test_run_captures_stderr():
    code, out, err = await _run(["sh", "-c", "echo error >&2"])
    assert "error" in err


# ---------------------------------------------------------------------------
# _wait_healthy
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wait_healthy_succeeds_on_200(respx_mock):
    import respx
    respx_mock.get("http://localhost:9999/healthz").mock(
        return_value=httpx.Response(200)
    )
    await _wait_healthy("http://localhost:9999", timeout=5)


@pytest.mark.asyncio
async def test_wait_healthy_raises_on_timeout():
    # Use a port that isn't listening — will always fail.
    with pytest.raises(TimeoutError):
        await _wait_healthy("http://localhost:19999", timeout=2)


# ---------------------------------------------------------------------------
# DockerInstance attributes
# ---------------------------------------------------------------------------

def test_docker_instance_url_format():
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    cf = Path(tmp.name) / "docker-compose.yml"
    cf.write_text("services: {}")
    inst = DockerInstance(port=55000, run_id="abc123", compose_file=cf, tmp_dir=tmp)
    assert inst.url == "http://localhost:55000"
    assert inst.port == 55000
    assert inst.run_id == "abc123"
    assert "abc123" in inst.project
    tmp.cleanup()


def test_docker_instance_project_name_contains_run_id():
    import tempfile
    tmp = tempfile.TemporaryDirectory()
    cf = Path(tmp.name) / "docker-compose.yml"
    cf.write_text("services: {}")
    inst = DockerInstance(port=55001, run_id="xyz789", compose_file=cf, tmp_dir=tmp)
    assert "xyz789" in inst.project
    tmp.cleanup()
