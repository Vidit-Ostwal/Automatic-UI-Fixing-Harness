# Autonomous UI Test Harness — Plan

## The Problem
Given only a Docker image + compose file for an unknown app, autonomously find visual and logic bugs, and report them with reproduction steps, screenshots, and severity — without any app-specific test cases.

## Core Philosophy
- **Autonomy over enumeration**: no hardcoded assertions. Every check works on an app we've never seen.
- **Two-phase architecture**: Planner explores and maps the app, Executors run trajectories in parallel.
- **Deterministic first, LLM second**: DOM geometry and CRUD invariants fire before LLM is invoked.
- **Noise discipline**: every suppressed false positive is logged with a reason.

---

## Architecture

```
harness/
  run_harness.py              # entry point: planner → executors → merge report
  planner/
    bfs_explorer.py           # BFS traversal, state graph construction
    action_identifier.py      # DOM enumeration + LLM semantic grouping
    trajectory_extractor.py   # root-to-leaf paths → trajectories.json
    state_hasher.py           # URL + a11y tree → structural hash
  executor/
    runner.py                 # runs one trajectory against one Docker instance
    workflows/
      auth.py                 # sign-up spine
      memos.py                # create / edit / pin / archive / search / delete
  oracles/
    visual.py                 # geometric heuristics (overflow, overlap, 404)
    logic.py                  # CRUD invariants, state toggle checks
    diff.py                   # before/after state diff oracle
    llm.py                    # LLM agent: screenshot + a11y tree → verdict
  browser/
    session.py                # Playwright setup, capture_state(), network log
  reporter/
    collector.py              # accumulates Finding objects per executor
    merger.py                 # merges findings from N parallel executors
    render.py                 # report.json + report.html
  models.py                   # Finding dataclass, Severity enum
  docker_manager.py           # spin up / tear down Docker instances
```

---

## Two-Phase Execution Flow

```
python run_harness.py
       │
       ├─ PHASE 1: PLANNER (single Docker instance)
       │       │
       │       ├─ Spin up Docker, wait for HTTP 200
       │       ├─ BFS traversal to depth N (default: 3)
       │       │     At each node:
       │       │       1. capture_state() → screenshot + a11y tree
       │       │       2. DOM enumerates all interactive elements
       │       │       3. LLM groups into semantic action types
       │       │       4. Hash state → skip if already visited
       │       │       5. Click each action → record (state → action → new_state)
       │       │
       │       ├─ Output: state_graph.json
       │       ├─ Extract all root-to-leaf paths → trajectories.json
       │       └─ Tear down planner Docker
       │
       └─ PHASE 2: PARALLEL EXECUTORS
               │
               ├─ Spawn one Docker instance per trajectory (parallel)
               ├─ Each executor:
               │     ├─ Fresh Docker + fresh browser context
               │     ├─ Runs its trajectory step by step
               │     ├─ At every step: oracles fire (visual + logic + diff + LLM)
               │     └─ Writes findings to findings_T{id}.json
               │
               └─ PHASE 3: REPORT
                     ├─ Merge all findings_T*.json
                     ├─ Deduplicate overlapping findings
                     └─ render.py → report.json + report.html
```

---

## State Capture (Every Step)

Both captured atomically before any action:

```python
async def capture_state(page):
    screenshot = await page.screenshot(full_page=True)   # PNG bytes
    a11y_tree  = await page.accessibility.snapshot()     # JSON tree
    return screenshot, a11y_tree
```

---

## BFS Explorer Detail

### State Hashing (Cycle Detection)
```python
def state_hash(url, a11y_tree):
    structure = extract_structural_skeleton(a11y_tree)   # strip dynamic values
    return hashlib.md5(f"{url}:{structure}".encode()).hexdigest()
```
- Strips dynamic content (timestamps, memo text, counts)
- Keeps structural shape (what element types exist, their roles, nesting)
- If hash already in `visited` → skip, don't re-expand
- Prevents infinite loops: pin→unpin→pin→... collapses to same hash

### Action Identification (At Each BFS Node)
1. **DOM layer**: find all interactive elements — `button`, `a[href]`, `input`, `select`, `[role="button"]`
2. **LLM layer**: receives element list + screenshot → groups semantically identical actions
   ```
   DOM finds:  [pin_memo_1, pin_memo_2, pin_memo_3, edit_memo, archive_memo, search_input]
   LLM groups: ["pin_a_memo", "edit_a_memo", "archive_a_memo", "use_search"]
   BFS expands: 4 branches, not 6
   ```
3. One representative element per semantic type → click it → record transition

### BFS Queue Structure
```python
queue:   [(state_hash, url, path_of_actions_taken)]
visited: set of state_hashes
graph:   {state_hash: [(semantic_action, resulting_state_hash, element_selector)]}
```

### Trajectory Extraction
After BFS completes, extract all root-to-leaf paths from the graph.
Each path becomes one trajectory in `trajectories.json`:
```json
[
  {
    "id": "T-01",
    "path": ["signup", "create_memo", "pin_memo"],
    "states": ["hash_A", "hash_B", "hash_C"],
    "description": "LLM-generated summary of what this trajectory tests"
  }
]
```

---

## Executor Detail (Per Trajectory)

Each executor gets a trajectory and its own fresh Docker instance.

### At Every Step, Oracles Fire:

**1. Diff Oracle (before → after every action)**
```python
before = capture_state(page)
execute_action(action)
after  = capture_state(page)
diff_oracle.evaluate(action, before, after)
```
LLM receives before/after screenshot + a11y tree:
*"I performed action X. Is this change expected, unexpected, or indicative of a bug?"*
- No change when change expected → bug
- Catastrophic unexpected change (UI disappears) → bug
- Expected change (memo moves to pinned section) → ok

**2. Visual Oracle (deterministic geometry)**
| Check | Method |
|-------|--------|
| Text overflow | `scrollWidth > offsetWidth` on text containers |
| Element overlap | `getBoundingClientRect` intersection > 5px |
| Broken images | HTTP status on `<img src>` |
| Viewport clip | Element bottom/right > viewport dimensions |

If flag fires → retry 2x with 500ms gap → if persists → send to LLM for confirmation

**3. Logic Oracle (CRUD invariants)**
| Action | Invariant |
|--------|-----------|
| Create memo | Text appears in memo list |
| Edit memo | Updated text replaces old |
| Pin memo | Memo in pinned section + button state changed |
| Archive memo | Gone from main list + in archive |
| Delete memo | Gone from everywhere |
| Search | Returns matching memos |
| Toggle | After pin→unpin, memo back in normal list |

**4. LLM Oracle (visual sanity)**
- Screenshot + a11y tree → LLM: *"Ignore loading states and empty states. Flag only clear rendering or logic defects."*
- Returns: `{"verdict": "bug|ok|noise", "description": "...", "severity": "..."}`

**5. Multi-Viewport Check**
- Core trajectories replayed at 375px / 768px / 1280px
- Re-run geometry heuristics at each size
- Compare a11y tree: elements disappearing/reordering unexpectedly = flagged
- Before/after screenshots sent to LLM

---

## False Positive Suppression

1. **Wait discipline** — `wait_for_load_state("networkidle")` + `wait_for_selector` before every capture
2. **Retry confirmation** — geometry flags re-checked 2x with 500ms gap before reporting
3. **LLM prompt filter** — explicit instruction to ignore spinners, skeleton screens, intentional empty states
4. **All suppressed items logged** with reason in `suppressed_noise` in final report

---

## How We Navigate Different Situations

| Situation | Response |
|-----------|----------|
| Expected element not found | LLM agent loop: screenshot + a11y → find element or report missing |
| Action completes, state doesn't change | Logic oracle flags; retry once; report as High |
| Geometry heuristic fires | 2x retry; LLM confirmation; file if confirmed |
| Console error / network failure | Correlate with step; filter known noise; report with context |
| Multi-viewport layout breaks | Compare a11y tree + screenshot across sizes; LLM judges |
| BFS cycle detected | State hash match → skip; don't re-expand |
| Loading state mid-capture | networkidle wait + retry |
| N executors produce duplicate findings | Merger deduplicates by (url + action + bug_type) |

---

## Report Structure

### `report.json`
```json
{
  "run_id": "...",
  "timestamp": "...",
  "trajectories_explored": 12,
  "findings": [{
    "id": "BUG-001",
    "type": "visual | logic",
    "severity": "critical | high | medium | low",
    "title": "...",
    "trajectory_id": "T-03",
    "steps": ["1. Sign up", "2. Create memo", "3. Click pin"],
    "evidence": {
      "screenshot_before": "screenshots/bug_001_before.png",
      "screenshot_after": "screenshots/bug_001_after.png",
      "console_errors": [],
      "network_errors": []
    },
    "detected_by": "heuristic | llm | both",
    "reasoning": "..."
  }],
  "suppressed_noise": [
    {"description": "...", "reason": "loading spinner during networkidle wait"}
  ]
}
```

### `report.html`
- Single self-contained file, embedded screenshots
- Summary table: bugs by severity + type
- Each finding: title, severity badge, steps, before/after screenshots
- Suppressed noise section at bottom

---

## Severity Rubric
| Severity | Criteria |
|----------|----------|
| Critical | Crash, hang, data loss, blank screen |
| High | Core workflow broken; element missing; state never changes |
| Medium | Wrong label/state, feature partially broken, wrong data |
| Low | Visual glitch, cosmetic overlap, minor layout issue |

---

## What Makes This General (Works on Any App)
1. BFS exploration discovers the app's structure — no app knowledge encoded
2. State hashing works on any DOM structure
3. Semantic action grouping via LLM works on any UI
4. Geometric visual checks are app-agnostic
5. CRUD invariants apply to any data-driven app
6. Diff oracle reasons about any action/state pair
7. Only app-specific input: the Docker compose file and the URL

---

## Configuration
```
DEPTH_N=3              # BFS depth (default 3)
MAX_TRAJECTORIES=20    # cap on parallel executors
LLM_MODEL=claude-sonnet-4-6  # vision model
APP_URL=http://localhost:5230
VIEWPORTS=375,768,1280
```

---

## Known Limitations
- Deep paths (> depth N) not explored unless N increased
- Server state is shared during planner phase — branches see each other's created data (isolated in executor phase via separate Docker)
- LLM visual findings marked `detected_by: llm` for traceability — may include hallucinations
- Cannot test file upload, native OS dialogs, non-HTTP interactions
- Multi-user / concurrency bugs not covered
