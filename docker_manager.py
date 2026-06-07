"""
Docker manager — spins up and tears down isolated Memos instances.

Each instance gets:
  - A dynamically chosen free port (no conflicts with other instances)
  - A unique container name  (memos-buggy-<run_id>)
  - A unique compose project (memos-<run_id>)
  - A temporary compose file written to /tmp

Multiple instances can run in parallel without any port or name collision.

Usage:
    async with DockerInstance.start() as instance:
        print(instance.url)   # http://localhost:<port>
        # run Playwright tests against instance.url
    # container is stopped and removed on exit
"""

import asyncio
import socket
import tempfile
import uuid
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# How long to wait for the app to become healthy (seconds).
HEALTH_TIMEOUT = 120
HEALTH_INTERVAL = 2
HEALTH_PATH = "/healthz"

# Docker image name — must match what was loaded via `docker load`.
IMAGE_NAME = "memos-buggy:latest"

# Internal container port (fixed by the image).
CONTAINER_PORT = 5230


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

def find_free_port() -> int:
    """
    Ask the OS for a free TCP port.
    Binds briefly to port 0 so the kernel picks an available port,
    then releases it. There is a small TOCTOU window — in practice this
    is fine because Docker binds the port within milliseconds.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Compose file template
# ---------------------------------------------------------------------------

_COMPOSE_TEMPLATE = """\
services:
  memos-buggy:
    image: {image}
    container_name: {container_name}
    ports:
      - "{host_port}:{container_port}"
    environment:
      MEMOS_MODE: prod
    tmpfs:
      - /var/opt/memos
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:{container_port}/healthz"]
      interval: 5s
      timeout: 3s
      retries: 20
"""


def _write_compose_file(
    host_port: int,
    container_name: str,
    tmp_dir: str,
) -> Path:
    content = _COMPOSE_TEMPLATE.format(
        image=IMAGE_NAME,
        container_name=container_name,
        host_port=host_port,
        container_port=CONTAINER_PORT,
    )
    path = Path(tmp_dir) / "docker-compose.yml"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Async subprocess helpers
# ---------------------------------------------------------------------------

async def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def _wait_healthy(url: str, timeout: int = HEALTH_TIMEOUT) -> None:
    """Poll the health endpoint until it returns 200 or timeout is exceeded."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url + HEALTH_PATH, timeout=3)
                if r.status_code == 200:
                    logger.info("Docker: %s is healthy", url)
                    return
            except Exception:
                pass
            await asyncio.sleep(HEALTH_INTERVAL)
    raise TimeoutError(
        f"App at {url} did not become healthy within {timeout}s"
    )


# ---------------------------------------------------------------------------
# DockerInstance
# ---------------------------------------------------------------------------

class DockerInstance:
    """
    A running Memos container bound to a unique host port.

    Attributes
    ----------
    url           Base URL for the app, e.g. http://localhost:54321
    port          Host port
    run_id        Short unique ID for this instance
    compose_file  Path to the temp compose file
    project       Docker Compose project name (isolates this instance)
    """

    def __init__(
        self,
        port: int,
        run_id: str,
        compose_file: Path,
        tmp_dir: object,  # tempfile.TemporaryDirectory handle
    ):
        self.port = port
        self.run_id = run_id
        self.compose_file = compose_file
        self.url = f"http://localhost:{port}"
        self.project = f"memos-{run_id}"
        self._tmp_dir = tmp_dir

    @classmethod
    @asynccontextmanager
    async def start(cls) -> AsyncIterator["DockerInstance"]:
        """
        Async context manager that starts a fresh container and tears it
        down on exit — even if an exception is raised.

        Usage:
            async with DockerInstance.start() as instance:
                await session.navigate(instance.url)
        """
        run_id = uuid.uuid4().hex[:8]
        port = find_free_port()
        container_name = f"memos-buggy-{run_id}"

        tmp_dir = tempfile.TemporaryDirectory(prefix=f"memos-harness-{run_id}-")
        compose_file = _write_compose_file(port, container_name, tmp_dir.name)

        instance = cls(port, run_id, compose_file, tmp_dir)

        logger.info(
            "Docker: starting instance %s on port %d (container: %s)",
            run_id, port, container_name,
        )

        try:
            await instance._up()
            await _wait_healthy(instance.url)
            yield instance
        finally:
            await instance._down()
            tmp_dir.cleanup()
            logger.info("Docker: instance %s stopped and removed", run_id)

    async def _up(self) -> None:
        code, out, err = await _run(
            ["docker", "compose", "-p", self.project,
             "-f", str(self.compose_file), "up", "-d"],
        )
        if code != 0:
            raise RuntimeError(
                f"docker compose up failed (exit {code}):\n{err}"
            )

    async def _down(self) -> None:
        code, out, err = await _run(
            ["docker", "compose", "-p", self.project,
             "-f", str(self.compose_file), "down", "--volumes", "--remove-orphans"],
        )
        if code != 0:
            logger.warning(
                "docker compose down returned %d for project %s:\n%s",
                code, self.project, err,
            )
