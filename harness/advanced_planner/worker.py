"""
ParallelWorker — one asyncio task per Docker container.

Lifecycle per work item:
  1. Pre-dedup check: if (parent_hash, action_name) already claimed, skip.
  2. Spin up Docker + BrowserSession.
  3. Replay trajectory via direct step execution (navigate → execute each step).
  4. Verify resulting state hash matches expected_parent_hash (3 retries).
  5. Execute assigned action.
  6. Hash resulting state; post-dedup check against visited_hashes.
  7. Identify child actions at new state (one LLM call).
  8. Submit result to coordinator.
  9. Docker + browser tear down automatically via context managers.

Bootstrap (trajectory=[], action_to_execute=None):
  - Navigate to app root, capture root state, identify actions, report.
  - No replay, no hash verification, no action execution.
"""

import asyncio
import logging

from harness.browser import BrowserSession
from harness.docker import DockerInstance
from harness.oracles.logic import LogicOracle
from harness.planner.action_identifier import ActionIdentifier, InteractionStep, SemanticAction
from harness.planner.state_hasher import state_hash
from harness.utils.step_runner import execute_steps
from harness.advanced_planner.coordinator import WorkCoordinator, WorkItem, WorkerResult
from harness.advanced_planner.shared_state import SharedVisitedState

logger = logging.getLogger(__name__)

MAX_REPLAY_RETRIES = 3
MAX_ACTIONS_PER_NODE = 8


def _dict_to_semantic(action: dict) -> SemanticAction:
    return SemanticAction(
        name=action["name"],
        description=action.get("description", ""),
        steps=[
            InteractionStep(
                type=s["type"],
                selector=s["selector"],
                value=s.get("value", ""),
            )
            for s in action.get("steps", [])
        ],
        expected_outcome=action.get("expected_outcome", ""),
    )


def _semantic_to_dict(action: SemanticAction) -> dict:
    return {
        "name":             action.name,
        "description":      action.description,
        "steps":            [{"type": s.type, "selector": s.selector, "value": s.value}
                             for s in action.steps],
        "expected_outcome": action.expected_outcome,
    }


async def _execute_steps(session: BrowserSession, action: SemanticAction, llm_oracle) -> tuple[bool, str]:
    steps = [{"type": s.type, "selector": s.selector, "value": s.value or ""}
             for s in action.steps]
    success, reason, _ = await execute_steps(session, steps, llm_oracle)
    return success, reason


async def _replay(
    session: BrowserSession,
    trajectory: list[dict],
    base_url: str,
    llm_oracle,
) -> bool:
    """
    Navigate to base_url then execute each action in trajectory in order.
    Returns False if any step fails.
    """
    await session.navigate(base_url)
    for action_dict in trajectory:
        action = _dict_to_semantic(action_dict)
        success, reason = await _execute_steps(session, action, llm_oracle)
        if not success:
            logger.debug(
                "Worker: replay step '%s' failed: %s",
                action.name, reason.split("\n")[0][:80],
            )
            return False
    return True


async def run_worker(
    item: WorkItem,
    coordinator: WorkCoordinator,
    shared_state: SharedVisitedState,
    llm_oracle,
    max_depth: int,
) -> None:
    """
    Run one parallel worker. Always calls coordinator.submit_result() exactly once
    so the in_flight counter stays accurate regardless of outcome.
    """
    action_name = item.action_to_execute["name"] if item.action_to_execute else "bootstrap"

    # Pre-spinup dedup: check-and-claim (parent_hash, action_name) atomically.
    if item.action_to_execute and item.expected_parent_hash:
        if not shared_state.check_and_claim_action(
            item.expected_parent_hash,
            item.action_to_execute["name"],
        ):
            logger.debug(
                "Worker: skip already-claimed action=%s parent=%s",
                action_name, item.expected_parent_hash[:8],
            )
            await coordinator.submit_result(WorkerResult(
                parent_hash=item.parent_hash,
                action_executed=item.action_to_execute,
                child_hash=None,
                child_url=None,
                child_a11y_tree=None,
                child_screenshot=None,
                child_actions=[],
                child_trajectory=item.trajectory,
                depth=item.depth,
                action_failed=True,
            ))
            return

    logic_oracle = LogicOracle()
    identifier = ActionIdentifier(llm_client=llm_oracle)

    try:
        async with DockerInstance.start() as docker:
            async with BrowserSession.create() as session:
                # ── Bootstrap ──────────────────────────────────────────────
                if item.action_to_execute is None:
                    await session.navigate(docker.url)
                    root_state = await session.capture_state()
                    root_elements = await session.get_interactive_elements()
                    root_hash = state_hash(root_state.url, root_state.a11y_tree, root_elements)

                    snapshot = coordinator.get_exploration_snapshot()
                    actions = await identifier.identify(
                        root_elements, root_state.a11y_tree, root_state.screenshot,
                        explored_context=snapshot,
                    )
                    actions = actions[:MAX_ACTIONS_PER_NODE]

                    shared_state.mark_visited(root_hash)

                    logger.info(
                        "Worker: bootstrap — root=%s  %d action(s) identified",
                        root_hash[:8], len(actions),
                    )
                    await coordinator.submit_result(WorkerResult(
                        parent_hash=None,
                        action_executed=None,
                        child_hash=root_hash,
                        child_url=root_state.url,
                        child_a11y_tree=root_state.a11y_tree,
                        child_screenshot=root_state.screenshot,
                        child_actions=[_semantic_to_dict(a) for a in actions],
                        child_trajectory=[],
                        depth=0,
                    ))
                    return

                # ── Normal worker: replay → verify → execute ───────────────
                logger.info(
                    "Worker: action=%s  depth=%d  parent=%s",
                    action_name, item.depth,
                    (item.expected_parent_hash or "")[:8],
                )

                parent_verified = False
                for attempt in range(1, MAX_REPLAY_RETRIES + 1):
                    replay_ok = await _replay(session, item.trajectory, docker.url, llm_oracle)
                    if replay_ok:
                        state = await session.capture_state()
                        elements = await session.get_interactive_elements()
                        actual_hash = state_hash(state.url, state.a11y_tree, elements)
                        if actual_hash == item.expected_parent_hash:
                            parent_verified = True
                            break
                        logger.warning(
                            "Worker: hash mismatch attempt %d/%d — expected=%s got=%s",
                            attempt, MAX_REPLAY_RETRIES,
                            item.expected_parent_hash[:8], actual_hash[:8],
                        )
                    else:
                        logger.warning(
                            "Worker: replay steps failed attempt %d/%d for action=%s",
                            attempt, MAX_REPLAY_RETRIES, action_name,
                        )

                    if attempt < MAX_REPLAY_RETRIES:
                        # Re-navigate to app root for next attempt.
                        try:
                            await session.navigate(docker.url)
                        except Exception:
                            pass

                if not parent_verified:
                    logger.error(
                        "Worker: all %d replay attempts failed for action=%s — discarding",
                        MAX_REPLAY_RETRIES, action_name,
                    )
                    await coordinator.submit_result(WorkerResult(
                        parent_hash=item.parent_hash,
                        action_executed=item.action_to_execute,
                        child_hash=None,
                        child_url=None,
                        child_a11y_tree=None,
                        child_screenshot=None,
                        child_actions=[],
                        child_trajectory=item.trajectory,
                        depth=item.depth,
                        replay_failed=True,
                    ))
                    return

                # Execute the assigned action.
                action = _dict_to_semantic(item.action_to_execute)
                success, fail_reason = await _execute_steps(session, action, llm_oracle)

                if not success:
                    logger.warning(
                        "Worker: action '%s' failed: %s",
                        action_name, fail_reason.split("\n")[0][:80],
                    )
                    await coordinator.submit_result(WorkerResult(
                        parent_hash=item.parent_hash,
                        action_executed=item.action_to_execute,
                        child_hash=None,
                        child_url=None,
                        child_a11y_tree=None,
                        child_screenshot=None,
                        child_actions=[],
                        child_trajectory=item.trajectory,
                        depth=item.depth,
                        action_failed=True,
                    ))
                    return

                # Capture resulting state.
                new_state = await session.capture_state()
                new_elements = await session.get_interactive_elements()
                new_hash = state_hash(new_state.url, new_state.a11y_tree, new_elements)

                # Crash detection.
                finding = logic_oracle.check_no_crash(new_state, action_name)
                if finding:
                    finding.trajectory_id = f"ADV-{(item.expected_parent_hash or '')[:6]}"

                child_trajectory = item.trajectory + [item.action_to_execute]

                # Post-execution dedup: skip children if state already visited.
                if shared_state.is_visited(new_hash):
                    logger.info(
                        "Worker: action=%s → already-visited state=%s",
                        action_name, new_hash[:8],
                    )
                    await coordinator.submit_result(WorkerResult(
                        parent_hash=item.parent_hash,
                        action_executed=item.action_to_execute,
                        child_hash=new_hash,
                        child_url=new_state.url,
                        child_a11y_tree=new_state.a11y_tree,
                        child_screenshot=new_state.screenshot,
                        child_actions=[],
                        child_trajectory=child_trajectory,
                        depth=item.depth,
                        finding=finding,
                    ))
                    return

                shared_state.mark_visited(new_hash)

                # Identify child actions if there are more depth levels.
                child_actions = []
                if item.depth + 1 < max_depth:
                    snapshot = coordinator.get_exploration_snapshot()
                    child_actions_semantic = await identifier.identify(
                        new_elements, new_state.a11y_tree, new_state.screenshot,
                        explored_context=snapshot,
                    )
                    child_actions = [_semantic_to_dict(a)
                                     for a in child_actions_semantic[:MAX_ACTIONS_PER_NODE]]

                logger.info(
                    "Worker: action=%s → new_state=%s  children=%d",
                    action_name, new_hash[:8], len(child_actions),
                )

                await coordinator.submit_result(WorkerResult(
                    parent_hash=item.parent_hash,
                    action_executed=item.action_to_execute,
                    child_hash=new_hash,
                    child_url=new_state.url,
                    child_a11y_tree=new_state.a11y_tree,
                    child_screenshot=new_state.screenshot,
                    child_actions=child_actions,
                    child_trajectory=child_trajectory,
                    depth=item.depth,
                    finding=finding,
                ))

    except Exception as exc:
        logger.exception(
            "Worker: unhandled exception for action=%s: %s", action_name, exc
        )
        # Always submit a result so in_flight is decremented.
        await coordinator.submit_result(WorkerResult(
            parent_hash=item.parent_hash,
            action_executed=item.action_to_execute,
            child_hash=None,
            child_url=None,
            child_a11y_tree=None,
            child_screenshot=None,
            child_actions=[],
            child_trajectory=item.trajectory,
            depth=item.depth,
            action_failed=True,
        ))
