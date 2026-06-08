"""
Goal writer — converts BFS trajectories into natural-language test goals.

For each trajectory, asks the LLM to produce:
  goal             — one-sentence high-level test objective
  instructions     — ordered plain-English steps for the executor/verifier
  success_criteria — verifiable conditions to check at the end of the run

Screenshots are loaded from disk (relative to output_dir) and sent to the LLM
so it can ground instructions in what the UI actually looks like at each state.

Heuristic fallback is used when no LLM is available or the LLM call fails,
so the output file is always written regardless.

Can be run via run_harness.py --goals-only to regenerate goals from an
existing output/trajectories.json without re-running BFS or executors.
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_screenshots(trajectory: dict, output_dir: Path | None) -> list[tuple[str, bytes]]:
    """
    Return (label, bytes) pairs for every screenshot in the trajectory,
    in display order.  Missing files are silently skipped.

    Labels describe which UI state each image shows, e.g.
      "State 0 — before 'create_account'"
      "State 1 — after 'create_account' / before 'create_new_memo'"
      "State 2 — after 'create_new_memo' (final state)"
    """
    if output_dir is None:
        return []

    steps = trajectory.get("steps", [])
    screenshot_paths: list[str] = trajectory.get("screenshots", [])
    if not screenshot_paths:
        return []

    # Build labels: first image is the starting state; each subsequent image is
    # the state reached after the corresponding step.
    labels: list[str] = []
    for i, path in enumerate(screenshot_paths):
        if i == 0:
            first_action = steps[0].get("action", "step 1") if steps else "step 1"
            labels.append(f"State 0 — initial page (before '{first_action}')")
        else:
            step = steps[i - 1] if i - 1 < len(steps) else {}
            action = step.get("action", f"step {i}")
            if i < len(screenshot_paths) - 1:
                next_action = steps[i].get("action", f"step {i+1}") if i < len(steps) else ""
                suffix = f" / before '{next_action}'" if next_action else ""
                labels.append(f"State {i} — after '{action}'{suffix}")
            else:
                labels.append(f"State {i} — after '{action}' (final state)")

    result: list[tuple[str, bytes]] = []
    for label, rel_path in zip(labels, screenshot_paths):
        full_path = output_dir / rel_path
        if full_path.exists():
            try:
                result.append((label, full_path.read_bytes()))
            except OSError:
                logger.debug("Goal writer: could not read %s", full_path)
        else:
            logger.debug("Goal writer: screenshot not found: %s", full_path)

    return result


async def write_trajectory_goals(
    trajectories: list[dict],
    llm_oracle,
    output_dir: Path | None = None,
) -> list[dict]:
    """
    Produce a goal dict for every trajectory.  Runs all LLM calls concurrently.
    Returns one entry per input trajectory in the same order.

    output_dir  — root of the output folder so screenshot paths in the
                  trajectory can be resolved to actual files on disk.
    """
    tasks = [_write_one(t, llm_oracle, output_dir) for t in trajectories]
    return await asyncio.gather(*tasks)


async def _write_one(
    trajectory: dict,
    llm_oracle,
    output_dir: Path | None,
) -> dict:
    traj_id     = trajectory.get("id", "T-???")
    description = trajectory.get("description", "")

    if llm_oracle is not None:
        screenshots = _load_screenshots(trajectory, output_dir)
        if not screenshots:
            logger.debug("Goal writer: %s — no screenshots found, LLM call text-only", traj_id)
        try:
            goal_data = await llm_oracle.write_trajectory_goal(trajectory, screenshots)
            if goal_data:
                logger.info(
                    "Goal writer: %s — LLM goal written (%d screenshot(s) sent)",
                    traj_id, len(screenshots),
                )
                return {
                    "id":          traj_id,
                    "description": description,
                    **goal_data,
                }
        except Exception as e:
            logger.warning("Goal writer: LLM failed for %s: %s", traj_id, e)

    logger.info("Goal writer: %s — using heuristic fallback", traj_id)
    return _heuristic_goal(trajectory)


def _heuristic_goal(trajectory: dict) -> dict:
    """Build a basic goal dict from trajectory metadata without LLM."""
    traj_id     = trajectory.get("id", "T-???")
    description = trajectory.get("description", "")
    steps       = trajectory.get("steps", [])

    instructions: list[str] = []
    for i, step in enumerate(steps, start=1):
        action    = step.get("action", "unknown")
        step_desc = step.get("description", "")
        if step_desc:
            instructions.append(f"{i}. {step_desc}")
        else:
            instructions.append(f"{i}. Perform action: {action}")

    last_action = steps[-1].get("action", "complete") if steps else "complete"
    success_criteria = [
        f"All {len(steps)} step(s) in the trajectory complete without error",
        f"The final action '{last_action}' produces a visible state change",
        "No crash or blank page is shown at any point",
    ]

    return {
        "id":               traj_id,
        "description":      description,
        "goal":             f"Execute the '{description}' workflow successfully end-to-end",
        "instructions":     instructions,
        "success_criteria": success_criteria,
    }
