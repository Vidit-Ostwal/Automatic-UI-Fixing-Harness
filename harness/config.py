"""
Central harness configuration.

Defaults are defined once in DEFAULTS. Values in .env (repo root) override them
when load_env() runs at startup. See .env.example for the full list of keys.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_LOADED = False

# Single source of default values — keep in sync with .env.example
DEFAULTS: dict[str, str] = {
    # Harness run
    "DEPTH_N": "3",
    "MAX_TRAJECTORIES": "20",
    "MAX_PARALLEL": "4",
    "OUTPUT_DIR": "output",
    "MAX_ACTIONS_PER_NODE": "8",
    "BFS_VERBOSE": "0",
    "EXECUTOR_INSTRUCTION_RETRIES": "3",
    # LLM provider (anthropic | openai | local, or leave empty for auto-detect)
    "LLM_PROVIDER": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_MODEL": "claude-sonnet-4-6",
    "OPENAI_MODEL": "gpt-4o",
    "LOCAL_LLM_URL": "",
    "LOCAL_LLM_MODEL": "Qwen/Qwen3.5-9B",
    # Docker / app under test
    "DOCKER_IMAGE": "memos-buggy:latest",
    "CONTAINER_PORT": "5230",
    "HEALTH_PATH": "/healthz",
    "HEALTH_TIMEOUT": "120",
    # Browser
    "BROWSER_HEADLESS": "true",
    "BROWSER_VIEWPORT_WIDTH": "1280",
    "BROWSER_VIEWPORT_HEIGHT": "800",
    # Report server
    "REPORT_PORT": "8765",
}


def load_env(env_path: Path | None = None) -> None:
    """Load .env from the repo root (no-op if already loaded or file missing)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    path = env_path or (_REPO_ROOT / ".env")
    if path.is_file():
        try:
            from dotenv import load_dotenv
            load_dotenv(path, override=False)
        except ImportError:
            _load_env_manual(path)
    _ENV_LOADED = True


def _load_env_manual(path: Path) -> None:
    """Minimal .env parser when python-dotenv is unavailable."""
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def env_str(key: str) -> str:
    return os.environ.get(key, DEFAULTS.get(key, ""))


def env_int(key: str) -> int:
    raw = env_str(key)
    try:
        return int(raw)
    except ValueError:
        return int(DEFAULTS.get(key, "0"))


def env_bool(key: str) -> bool:
    return env_str(key).lower() in ("1", "true", "yes", "on")


def repo_root() -> Path:
    return _REPO_ROOT
