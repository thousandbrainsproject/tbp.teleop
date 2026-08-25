# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Read this step's goals, their sources, and jump outcomes from a Monty model.

Goals are proposed by sensor modules (salience locations, `sender_type "SM"`) and by
learning modules' goal generators (hypothesis-testing targets, `sender_type "GSG"`).
During `Monty._pass_goals` the attention system stamps every proposed goal's
`info["passed_attention_filter"]` before dropping the filtered ones, and the stamped
objects stay readable on their emitters, so the plotter can show both the goals that
fell within the active attention space and those that were filtered out. The motor
system's policy selector records the goal it enacted (if any) in `_selected_goals`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tbp.monty.frameworks.models.motor_policy_selectors import DistantPolicySelector

if TYPE_CHECKING:
    from tbp.monty.cmp import Goal
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty


def proposed_goals(model: Monty) -> list[Goal]:
    """Return every goal proposed by the model's SMs and LMs this step.

    Reads the same per-module `propose_goals` accessors `Monty._pass_goals` reads, so
    the returned objects are the ones the attention system stamped with
    `info["passed_attention_filter"]`. Includes both the goals that survived the
    attention filter and the ones that were dropped. On a motor-only step the modules
    keep their last proposals, so the returned goals are the most recent real step's.

    Args:
        model: The Monty model whose modules are read.

    Returns:
        The proposed goals, in LM-then-SM module order.
    """
    goals: list[Goal] = []
    for lm in model.learning_modules:
        goals.extend(lm.propose_goals())
    for sm in model.sensor_modules:
        goals.extend(sm.propose_goals())
    return goals


def passed_attention_filter(goal: Goal) -> bool:
    """Whether a goal fell within the active attention space.

    The flag is stamped by the attention system's goal filter during
    `Monty._pass_goals`. A goal with no flag (e.g. under the `NoopAttentionSystem`,
    which never filters) counts as passed.

    Args:
        goal: The proposed goal to inspect.

    Returns:
        False only when the attention filter explicitly dropped the goal.
    """
    return goal.info.get("passed_attention_filter") is not False


def enacted_goal(model: Monty) -> Goal | None:
    """Return the goal the motor system's policy selector enacted this step.

    Args:
        model: The Monty model whose motor system is read.

    Returns:
        The selected goal recorded in the policy selector's telemetry, or `None` when
        no step has run yet or this step was not goal-driven.
    """
    selector = model.motor_system._policy_selector
    selected = getattr(selector, "_selected_goals", [])
    return selected[-1] if selected else None


def enacted_policy_kind(model: Monty) -> str | None:
    """Classify the goal-driven policy that produced this step's actions.

    Only the `DistantPolicySelector` runs goal-driven policies. When the jump policy
    produced actions, the selector's `_is_jumping` telemetry separates a fresh jump
    (still in progress, awaiting the visibility check) from the undo of a failed one
    (the jump state was just cleared and the returned actions move the agent back).

    Args:
        model: The Monty model whose motor system is read.

    Returns:
        `"jump"` for a hypothesis-testing jump, `"move back"` for the undo of a
        failed jump, `"look"` for a sensor module's look-at goal, or `None` when this
        step was not goal-driven.
    """
    selector = model.motor_system._policy_selector
    if not isinstance(selector, DistantPolicySelector):
        return None
    selected = selector._selected_policies
    if not selected:
        return None
    policy = selected[-1]
    if policy is selector._jump_to_goal:
        return "jump" if selector._is_jumping else "move back"
    if policy is selector._look_at_goal:
        return "look"
    return None


def goal_status(model: Monty) -> str:
    """One-line description of this step's goal-driven action and its source.

    Args:
        model: The Monty model whose motor system is read.

    Returns:
        E.g. `"jump goal from learning_module_0 (GSG)"` or `"moving back from failed
        jump"`, or an empty string when this step was not goal-driven.
    """
    kind = enacted_policy_kind(model)
    if kind is None:
        return ""
    if kind == "move back":
        return "moving back from failed jump"
    goal = enacted_goal(model)
    if goal is None:
        return ""
    return f"{kind} goal from {goal.sender_id} ({goal.sender_type})"


def jump_button_text(model: Monty) -> str:
    """The interactive jump button's caption for this step's proposed goal action.

    Args:
        model: The Monty model whose motor system is read.

    Returns:
        The goal's source in parentheses after the verb (e.g.
        `"jump (learning_module_0)"`, `"look (view_finder)"`), or `"move back"` when
        the proposed actions undo a failed jump.
    """
    kind = enacted_policy_kind(model)
    if kind == "move back":
        return "move back"
    goal = enacted_goal(model)
    if goal is None:
        return "jump"
    verb = "look" if kind == "look" else "jump"
    return f"{verb} ({goal.sender_id})"


class JumpWatcher:
    """Detects when a hypothesis-testing jump failed and Monty moved back.

    A jump plays out over two steps: the step that emits the jump actions leaves the
    selector's `_is_jumping` telemetry set, and the next step either validates the new
    viewpoint (the jump policy returns no actions and another policy is selected) or
    finds no object visible and returns actions that move the agent back to its
    pre-jump pose (the jump policy is selected again with `_is_jumping` cleared).
    Observing that transition once per step yields a user-facing failure message,
    attributed to the goal that started the jump.
    """

    def __init__(self) -> None:
        """Initialize the watcher with no jump in flight."""
        self._was_jumping = False
        self._jump_sender: str | None = None

    def observe(self, model: Monty) -> str | None:
        """Record this step's jump state and report a failed jump.

        Must be called exactly once per step (repaints of the same frame must not
        re-observe), because failure detection compares against the previous step's
        jump state.

        Args:
            model: The Monty model whose motor system is read.

        Returns:
            The failure message when this step moved the agent back from a failed
            jump, else `None`.
        """
        selector = model.motor_system._policy_selector
        if not isinstance(selector, DistantPolicySelector):
            return None
        is_jumping = selector._is_jumping
        selected = selector._selected_policies
        undone = (
            self._was_jumping
            and not is_jumping
            and bool(selected)
            and selected[-1] is selector._jump_to_goal
        )
        message = None
        if undone:
            source = self._jump_sender or "unknown source"
            message = (
                f"Goal from {source} unsuccessful: no object visible at the goal "
                "location; moving back"
            )
        if is_jumping:
            goal = enacted_goal(model)
            self._jump_sender = goal.sender_id if goal is not None else None
        self._was_jumping = is_jumping
        return message
