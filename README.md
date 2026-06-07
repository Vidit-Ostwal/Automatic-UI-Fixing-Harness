# Autonomous UI Test Harness

Black-box UI defect discovery for locally-hosted web apps.  
Drives a real browser (Playwright), explores the app via BFS, fires visual and logic oracles at every step, and produces a structured HTML + JSON report.

---

## How it works

```
Phase 1 — PLANNER
  One Docker instance → BFS exploration → state graph → trajectories.json

Phase 2 — EXECUTORS  (parallel)
  N Docker instances, one per trajectory → oracles fire at every step → findings

Phase 3 — REPORT
  Merge & deduplicate findings → report.html + report.json
```

---

## Setup

### 1. Load the Docker image (one-time)

```bash
docker load -i memos-buggy.image.tar.gz
```

### 2. Install Python dependencies

```bash
uv sync
uv run playwright install chromium
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Use Anthropic (claude-sonnet-4-6) as LLM oracle |
| `OPENAI_API_KEY` | — | Use OpenAI (gpt-4o) as LLM oracle |
| `LLM_PROVIDER` | auto | Force provider: `anthropic` \| `openai` \| `local` |
| `LOCAL_LLM_URL` | `http://20.150.215.227` | Override local Qwen endpoint base URL |
| `LOCAL_LLM_MODEL` | `Qwen/Qwen3.5-9B` | Override local model name |
| `DEPTH_N` | `3` | BFS exploration depth |
| `MAX_TRAJECTORIES` | `20` | Cap on executor runs |
| `MAX_PARALLEL` | `4` | Concurrent Docker instances |
| `OUTPUT_DIR` | `output/` | Where reports and screenshots are written |

**LLM provider priority** (first match wins):
1. Anthropic — `LLM_PROVIDER=anthropic` or `ANTHROPIC_API_KEY` set
2. OpenAI — `LLM_PROVIDER=openai` or `OPENAI_API_KEY` set
3. Local Qwen — `LLM_PROVIDER=local` or last-resort fallback (no API key needed)

---

## Running

### Full run (explore + test + report)

```bash
# With local Qwen model — no API key needed
uv run python run_harness.py

# With Anthropic (better quality)
ANTHROPIC_API_KEY=sk-ant-... uv run python run_harness.py

# Tune depth and parallelism
uv run python run_harness.py --depth 4 --max-trajectories 15
```

Output goes to `output/`:
- `report.html` — open in browser; findings with screenshots, severity badges, reproduction steps
- `report.json` — machine-readable findings
- `trajectories.json` — BFS graph (reusable with `--skip-planner`)
- `screenshots/` — individual PNG evidence files

**Exit codes** (CI-friendly):
- `0` — no critical/high findings
- `1` — at least one critical or high finding
- `2` — `--skip-planner` requested but `trajectories.json` not found

---

## Inspecting the BFS exploration only

To run **only Phase 1** and see what the BFS explorer discovers — without spinning up parallel executors or firing oracles — use `--planner-only`:

```bash
uv run python run_harness.py --planner-only
```

This will:
1. Start one Docker instance of the app
2. Launch Playwright and explore the UI via BFS (default depth 3)
3. Print every discovered trajectory to the terminal — each step with its action and selector
4. Write `output/trajectories.json` and exit

Example output:
```
============================================================
  BFS exploration complete — 7 trajectory/trajectories found
============================================================

  [01] T-000  (2 step(s))
       Navigate to home, create memo
         1. signup
         2. create_memo  →  button[aria-label="New memo"]

  [02] T-001  (3 step(s))
         1. signup
         2. create_memo  →  button[aria-label="New memo"]
         3. pin_memo     →  button[aria-label="Pin"]
  ...

  trajectories.json written to output dir
```

Tune BFS depth and output location:
```bash
uv run python run_harness.py --planner-only --depth 2 --output my-run/
```

Once you're happy with the trajectories, run the executors against them without re-exploring:
```bash
uv run python run_harness.py --skip-planner --output my-run/
```

---

## All flags

| Flag | Description |
|---|---|
| `--planner-only` | Run BFS only; print trajectories and exit |
| `--skip-planner` | Skip BFS; load existing `trajectories.json` from output dir |
| `--no-llm` | Disable LLM oracle (deterministic geometry/logic checks only) |
| `--depth N` | BFS depth override |
| `--max-trajectories N` | Executor cap override |
| `--output DIR` | Output directory override |

---

## Running tests

```bash
uv run pytest          # all 263 tests
uv run pytest -v -k planner   # filter by name
```
