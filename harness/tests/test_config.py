from pathlib import Path

from harness.config import DEFAULTS, env_bool, env_int, env_str, load_env


def test_defaults_match_env_example_keys():
    example = Path(__file__).resolve().parents[2] / ".env.example"
    example_keys = set()
    for line in example.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        example_keys.add(line.partition("=")[0].strip())
    assert example_keys == set(DEFAULTS.keys())


def test_env_int_uses_default(monkeypatch):
    monkeypatch.delenv("DEPTH_N", raising=False)
    assert env_int("DEPTH_N") == 3


def test_env_bool_true_values(monkeypatch):
    monkeypatch.setenv("BFS_VERBOSE", "1")
    assert env_bool("BFS_VERBOSE") is True


def test_load_env_reads_file(tmp_path, monkeypatch):
    import harness.config as config_mod

    env_file = tmp_path / ".env"
    env_file.write_text("DEPTH_N=7\n")
    monkeypatch.delenv("DEPTH_N", raising=False)
    config_mod._ENV_LOADED = False
    load_env(env_file)
    assert env_int("DEPTH_N") == 7


def test_env_str_empty_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert env_str("LLM_PROVIDER") == ""
