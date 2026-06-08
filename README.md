# Autonomous UI Test Harness

Black-box UI defect discovery for locally hosted web apps. The harness drives a real browser (Playwright), explores the app via BFS, converts discovered paths into plain-English test goals, executes them in parallel, and verifies each step with an LLM-based verifier to produce structured findings and an HTML report.

The target app is a buggy **Memos** instance shipped as a Docker image (`memos-buggy:latest`), but the architecture is app-agnostic: exploration, goal writing, execution, and verification are LLM-assisted with deterministic fallbacks (DOM heuristics, crash detection).

---

## How it works

```
Phase 1 — PLANNER
  One Docker instance → BFS exploration → state graph → trajectories.json
  + per-trajectory screenshots in screenshots/T-NNN/

Phase 2 — GOAL WRITER
  LLM converts each trajectory into plain-English goals → trajectories_goal.json

Phase 3 — EXECUTORS + VERIFIERS  (parallel)
  N Docker instances, one per goal → GoalExecutor runs each instruction
  VerifierAgent consumes step events → verifier_claims/T-NNN_<run_id>/claims.json

Phase 4 — REPORT  (standalone, via --report)
  Load verifier claims → render report.html → serve locally and open in browser
```

**Key design note:** Executors are **goal-driven**, not raw trajectory replay. Each executor reads only `trajectories_goal.json` (plain-English instructions) and uses the LLM to resolve them to concrete UI actions at runtime. BFS trajectories inform goal writing via screenshots and step metadata, but are not replayed directly.

---

## Project structure

```
AutomaticUIFixingHarness/
├── run_harness.py              # Sole CLI entry point
├── pyproject.toml              # Python dependencies (uv/pip)
├── exploration_memory.md       # Live BFS exploration log (regenerated each run)
├── harness/
│   ├── browser/                # Playwright session wrapper
│   ├── docker/                 # Docker compose lifecycle
│   ├── planner/                # BFS, action ID, trajectory extraction, goal writer
│   ├── executor/                 # GoalExecutor + step queue messages
│   ├── verifier/                 # VerifierAgent (per-step bug detection)
│   ├── oracles/                  # LLM transport + LogicOracle (crash checks)
│   ├── reporter/                 # Load claims → render HTML → serve
│   ├── utils/                    # step_runner, fill_retry, elements, url helpers
│   ├── models/                   # Finding, PageState, Severity, etc.
│   └── tests/                    # pytest suite
└── output/                       # Default run artifacts (see Output artifacts)
```

---

## Setup

### Requirements

- **Python 3.13+**
- **Docker** (for app instances)
- **uv** (recommended package manager)

### 1. Load the Docker image (one-time)

```bash
docker load -i memos-buggy.image.tar.gz
```

### 2. Install Python dependencies

```bash
uv sync
uv run playwright install chromium
```

### 3. Configure LLM provider (optional but recommended)

```bash
# Cloud — Anthropic (best quality)
export ANTHROPIC_API_KEY=sk-ant-...

# Cloud — OpenAI
export OPENAI_API_KEY=sk-...

# Local — no API key needed (default fallback when no cloud key is set)
export LLM_PROVIDER=local
export LOCAL_LLM_URL=http://your-host:port
export LOCAL_LLM_MODEL=Qwen/Qwen3.5-9B
```

---

## Configuration

All configuration is via **environment variables** and **CLI flags**. There are no config files or `.env` templates in the repo (though you can source your own `.env` before running).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DEPTH_N` | `3` | BFS exploration depth (max hops from root state) |
| `MAX_TRAJECTORIES` | `20` | Cap on goals executed in Phase 3 |
| `MAX_PARALLEL` | `4` | Max concurrent Docker + executor instances |
| `OUTPUT_DIR` | `output` | Root output directory for all artifacts |
| `LLM_PROVIDER` | auto | Force provider: `anthropic` \| `openai` \| `local` |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic |
| `OPENAI_API_KEY` | — | Required when using OpenAI |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Override Anthropic model |
| `OPENAI_MODEL` | `gpt-4o` | Override OpenAI model |
| `LOCAL_LLM_URL` | `http://20.150.215.227` | Local OpenAI-compatible endpoint base URL |
| `LOCAL_LLM_MODEL` | `Qwen/Qwen3.5-9B` | Local model name |
| `BFS_VERBOSE` | unset | Set to `1` for step-level BFS debug logs |

**LLM provider priority** (first match wins):

1. **Anthropic** — `LLM_PROVIDER=anthropic` or `ANTHROPIC_API_KEY` set
2. **OpenAI** — `LLM_PROVIDER=openai` or `OPENAI_API_KEY` set
3. **Local Qwen** — `LLM_PROVIDER=local` or fallback when no cloud API key is present

CLI flags `--depth`, `--max-trajectories`, and `--output` override `DEPTH_N`, `MAX_TRAJECTORIES`, and `OUTPUT_DIR` respectively.

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--planner-only` | off | Run Phases 1–2 only (BFS + goal writer); print trajectories and exit |
| `--skip-planner` | off | Skip Phase 1; load existing `trajectories.json` from output dir |
| `--goals-only` | off | Load `trajectories.json` → write `trajectories_goal.json` → exit |
| `--run-goals` | off | Load `trajectories_goal.json` → run executors + verifiers only |
| `--report` | off | Load verifier claims → render + serve `report.html` in browser |
| `--no-llm` | off | Disable LLM oracle (deterministic BFS only; executors cannot resolve instructions) |
| `--verbose-bfs` | off | Step-level BFS logs (equivalent to `BFS_VERBOSE=1`) |
| `--depth N` | env / `3` | BFS depth override |
| `--max-trajectories N` | env / `20` | Executor cap override |
| `--rollout N` | off | Clear `executor_runs/` + `verifier_claims/`, then run N executor passes |
| `--output DIR` | env / `output` | Output directory override |

### Internal defaults (not exposed via CLI or env)

| Setting | Value | Location |
|---|---|---|
| Docker image | `memos-buggy:latest` | `harness/docker/manager.py` |
| Container port | `5230` | same |
| Health check path | `/healthz` | same |
| Health timeout | `120s` | same |
| Max actions per BFS node | `8` | `run_harness.py` → `BFSExplorer` |
| Browser headless | `true` | `harness/browser/session.py` |
| Browser viewport | `1280×800` | `harness/browser/session.py` |
| Exploration memory file | `exploration_memory.md` (repo root) | `harness/planner/bfs_explorer.py` |
| Report server port | `8765` (auto-increments if busy) | `harness/reporter/serve.py` |

---

## Running the harness

There is a single entry point:

```bash
uv run python run_harness.py [options]
```

### Execution modes overview

| Mode | Command | Phases run | Requires |
|---|---|---|---|
| **Full pipeline** | `uv run python run_harness.py` | 1 → 2 → 3 | Docker, Playwright, LLM (recommended) |
| **Planner only** | `--planner-only` | 1 → 2 | Docker, Playwright |
| **Skip planner** | `--skip-planner` | 2 → 3 | `trajectories.json` |
| **Goals only** | `--goals-only` | 2 | `trajectories.json` |
| **Run goals only** | `--run-goals` | 3 | `trajectories_goal.json` |
| **Report** | `--report` | 4 | `verifier_claims/` |
| **Rollout** | `--rollout N` (with full or `--run-goals`) | 3 × N | Clears executor artifacts first |

The flags `--report`, `--goals-only`, and `--run-goals` are **standalone entry modes** — each short-circuits the normal pipeline. All other flags compose with the default flow.

---

### 1. Full pipeline (default)

Runs BFS exploration → goal writing → parallel executors + verifiers.

```bash
# Local Qwen — no API key needed
uv run python run_harness.py

# Anthropic (better quality)
ANTHROPIC_API_KEY=sk-ant-... uv run python run_harness.py

# Tune depth, trajectory cap, and output location
uv run python run_harness.py --depth 4 --max-trajectories 15 --output my-run/
```

**Note:** The full pipeline does **not** auto-generate `report.html`. Run `--report` separately after executors finish (see mode 7).

---

### 2. Planner only (explore + write goals, no execution)

Runs Phase 1 (BFS) and Phase 2 (goal writer), prints discovered trajectories, then exits without running executors.

```bash
uv run python run_harness.py --planner-only
```

Produces:
- `trajectories.json`
- `screenshots/T-NNN/` (per-step PNGs)
- `trajectories_goal.json`
- `exploration_memory.md`

Example terminal output:

```
============================================================
  BFS exploration complete — 7 trajectory/trajectories found
============================================================

  [01] T-001  (3 step(s))
       create_account → create_new_memo → create_memo_with_tag
         1. create_account  →  button[type="submit"]
         2. create_new_memo  →  textarea[placeholder="Any thoughts..."]
         3. create_memo_with_tag  →  button[aria-label="Save"]

  trajectories.json written to output dir
```

Tune exploration:

```bash
uv run python run_harness.py --planner-only --depth 2 --output my-run/ --verbose-bfs
```

Deterministic exploration without LLM:

```bash
uv run python run_harness.py --no-llm --planner-only
```

---

### 3. Skip planner (reuse existing trajectories)

Loads `trajectories.json` from the output dir, regenerates goals, then runs executors + verifiers.

```bash
uv run python run_harness.py --skip-planner --output my-run/
```

Useful after `--planner-only` when you want to execute without re-exploring:

```bash
# Step 1: explore
uv run python run_harness.py --planner-only --output my-run/

# Step 2: execute against saved trajectories
uv run python run_harness.py --skip-planner --output my-run/
```

---

### 4. Goals only (regenerate goals from trajectories)

Reads existing `trajectories.json`, writes fresh `trajectories_goal.json`, and exits. Does not run BFS or executors.

```bash
uv run python run_harness.py --goals-only --output my-run/
```

Use when you want to regenerate plain-English goals (e.g. after changing the LLM provider or model) without re-exploring.

---

### 5. Run goals only (execute existing goals)

Reads `trajectories_goal.json`, runs executors + verifiers, and exits. Skips BFS and goal writing.

```bash
uv run python run_harness.py --run-goals --output my-run/
```

Use when goals are already written and you only need to re-execute.

---

### 6. Rollout (repeat executor passes)

Clears prior `executor_runs/` and `verifier_claims/`, then runs the full executor + verifier pass N times. Each pass creates new run IDs and claims files.

With existing goals:

```bash
uv run python run_harness.py --run-goals --rollout 3 --output my-run/
```

As part of the full pipeline:

```bash
uv run python run_harness.py --rollout 2
```

Rollout results are summarized in `executor_trajectories.json` with a `rollout` index per entry.

---

### 7. HTML report (standalone)

Loads all `verifier_claims/*/claims.json`, renders `report.html`, starts a local HTTP server (default port 8765), and opens the report in your browser. Blocks until Ctrl+C.

```bash
uv run python run_harness.py --report --output my-run/
```

---

### 8. No-LLM mode

Disables the LLM oracle entirely. BFS falls back to DOM heuristics for action identification. Goal writing uses a heuristic fallback. Executors cannot resolve plain-English instructions without an LLM.

```bash
uv run python run_harness.py --no-llm --planner-only   # exploration only
```

For meaningful executor runs, an LLM provider is required.

---

### Common multi-step workflows

**Explore → review → execute → report**

```bash
uv run python run_harness.py --planner-only --output my-run/
# Review my-run/trajectories.json and screenshots/
uv run python run_harness.py --skip-planner --output my-run/
uv run python run_harness.py --report --output my-run/
```

**Explore → execute in one shot**

```bash
uv run python run_harness.py --depth 3 --max-trajectories 10
uv run python run_harness.py --report
```

**Regenerate goals and re-execute**

```bash
ANTHROPIC_API_KEY=sk-ant-... uv run python run_harness.py --goals-only
uv run python run_harness.py --run-goals --rollout 2
uv run python run_harness.py --report
```

**Flaky-behavior investigation with rollouts**

```bash
uv run python run_harness.py --run-goals --rollout 5 --max-trajectories 5
```

---

## Output artifacts

All artifacts are written under `OUTPUT_DIR` (default `output/`).

| File / Directory | Produced by | Contents |
|---|---|---|
| `trajectories.json` | Phase 1 | BFS paths with action steps, selectors, screenshot paths |
| `screenshots/T-NNN/` | Phase 1 | `00_start.png`, `01_<action>.png`, … per trajectory |
| `trajectories_goal.json` | Phase 2 | Per-trajectory `goal`, `instructions`, `success_criteria` |
| `executor_runs/T-NNN_<run_id>/` | Phase 3 | `run.json`, `step_NN_before.png`, `step_NN_after.png` |
| `verifier_claims/T-NNN_<run_id>/claims.json` | Phase 3 | Findings with severity, evidence, reproduction steps |
| `executor_trajectories.json` | Phase 3 | Summary of all executor runs (includes rollout index) |
| `report.html` | `--report` | Human-readable findings report with screenshots |
| `exploration_memory.md` | BFS | Live log of explored nodes and actions (repo root) |

### `trajectories_goal.json` shape

Each entry contains:

```json
{
  "id": "T-001",
  "description": "create_account → create_new_memo → ...",
  "goal": "Verify that a user can ...",
  "instructions": ["On the 'Create your account' page, ...", "..."],
  "success_criteria": ["The user is successfully logged in ...", "..."]
}
```

### `executor_runs/.../run.json` shape

Records per-instruction execution: resolved UI steps, before/after screenshots, URLs, success/error per step, and overall `completed` / `final_state`.

### `verifier_claims/.../claims.json` shape

Contains `findings[]` with `severity`, `title`, `description`, `evidence`, `reproduction_steps`, and screenshot paths — consumed by the HTML report.

---

## Exit codes

CI-friendly exit codes:

| Code | Meaning |
|---|---|
| `0` | No critical/high findings (or successful planner/goals-only run) |
| `1` | At least one critical or high finding |
| `2` | Required input missing (`trajectories.json`, `trajectories_goal.json`, or `verifier_claims/`) |

---

## Parallelism

Up to `MAX_PARALLEL` (default 4) `GoalExecutor` instances run concurrently, each with its own Docker container and browser session. A single `VerifierAgent` consumer reads from a shared `asyncio.Queue` and routes step messages by `(test_case_id, run_id)`.

---

## Running tests

```bash
uv run pytest                    # all tests under harness/tests/
uv run pytest -v -k planner      # filter by name
```

Tests mock Docker, browser, and LLM — no real processes required.

---

## Architecture

```
run_harness.py
  │
  ├─ Phase 1: BFSExplorer → ActionIdentifier → trajectory_extractor
  │            LogicOracle (crash detection during exploration)
  │            → trajectories.json + screenshots/
  │
  ├─ Phase 2: goal_writer (LLM + heuristic fallback)
  │            → trajectories_goal.json
  │
  ├─ Phase 3: GoalExecutor × N  ──StepMessage queue──▶  VerifierAgent
  │            (LLM resolves instructions → step_runner)
  │            → executor_runs/ + verifier_claims/
  │
  └─ Phase 4: collector → render → serve  (--report only)
              → report.html
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `playwright` | Browser automation |
| `anthropic` | Claude API |
| `openai` | GPT API |
| `httpx` | Local LLM HTTP calls |
| `pillow` | Image handling |
| `pytest`, `pytest-asyncio`, `respx` | Testing |
