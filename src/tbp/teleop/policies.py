# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

import quaternion as qt
from tbp.monty.frameworks.actions.actions import MoveTangentially
from tbp.monty.frameworks.models.motor_policies import (
    BasePolicy,
    InformedPolicy,
    InformedPolicyRandomWalk,
    SurfacePolicy,
    fixme_undo_last_action,
)
from tbp.monty.frameworks.models.motor_policy_selectors import (
    DistantPolicySelector,
    SinglePolicySelector,
)

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty
    from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
    from tbp.monty.math import VectorXYZ


# The exploration headings offered as button labels. Every policy offers these same
# four; what each one does depends on the policy.
HEADINGS = ("up", "down", "left", "right")

# The shared step-size multiplier window for the interactive slider. The slider widget
# never changes between policies; each policy declares its own absolute default step,
# and the executed step is that default scaled by the slider value (1.0 = the default).
STEP_SCALE_MIN, STEP_SCALE_MAX, STEP_SCALE_INIT = 0.1, 3.0, 1.0


class InteractivePolicy(Protocol):
    """What the interactive plotter needs to drive a policy by hand.

    An interactive policy adapts one of the autonomous motor policies into a small set
    of user-selectable headings (up, down, left, right). The user picks a heading; what
    it does depends on the wrapped policy: a sampler rotation for the distant agent, a
    tangential surface move for the surface agent. The wrapped policy's automatic
    per-step corrections still run and are never overridden. The surface policy, for
    instance, keeps orienting the sensor and moving forward to touch the object.

    Because the user's choice replaces the action the wrapped policy proposed, the
    policy is left believing it took an action the environment never executed. The
    chosen action is fed back through `feedback` so that whatever the policy derives
    from its own last action, e.g., when undoing an action, stays true. The `feedback`
    implementation is different per policy, and for some policies, e.g., surface policy,
    it is nothing at all, so each adapter documents what its own `feedback` does.
    """

    def awaits_choice(self, proposed: list[Action]) -> bool:
        """Whether this step is a user choice point rather than an automatic step.

        Args:
            proposed: The actions the wrapped policy computed for this step.

        Returns:
            True when the user should choose the action instead of running the
            policy's automatic correction.
        """
        ...

    def compute(
        self,
        ctx: RuntimeContext,
        name: str,
        state: MotorSystemState,
        scale: float = 1.0,
    ) -> list[Action]:
        """Compute the action for the chosen heading from the current motor state.

        Args:
            ctx: The runtime context supplying the random state.
            name: The selected heading.
            state: The current state of the motor system.
            scale: Multiplier on the policy's default step; 1.0 = the declared default.

        Returns:
            The actions to execute next.
        """
        ...

    def feedback(self, chosen: list[Action]) -> None:
        """Tell the wrapped policy which action is actually being executed.

        Called after the user's choice replaces the policy's own proposal, so that
        whatever the policy derives from its last action is derived from the action
        the environment actually received. Note that policies do not necessarily keep
        the last action itself, so an implementation may have to update a derived
        value instead, or may have nothing to update at all.

        Args:
            chosen: The actions returned by `compute` that will be executed.
        """
        ...


class SampledInteractivePolicy:
    """Interactive adapter for sampler-driven policies (distant, informed, base).

    Every step is a user choice. The user picks a heading (up, down, left, right),
    which maps to the corresponding sampler rotation (look up/down, turn left/right)
    so the configured action amounts are respected.
    """

    _HEADINGS: ClassVar[dict[str, str]] = {
        "up": "look_up",
        "down": "look_down",
        "left": "turn_left",
        "right": "turn_right",
    }

    # The interactive default rotation per heading, in degrees. Scaled by the slider
    # multiplier; declared here rather than read from the sampler.
    DEFAULT_ROTATION_DEGREES: ClassVar[float] = 5.0

    def __init__(self, policy: BasePolicy | InformedPolicyRandomWalk) -> None:
        """Initialize the adapter from a sampler-driven policy.

        Args:
            policy: The wrapped policy exposing an action sampler and agent id.
        """
        self._policy = policy
        self._sampler = policy.action_sampler
        self._agent_id = policy.agent_id

    def awaits_choice(self, proposed: list[Action]) -> bool:  # noqa: ARG002
        return True

    def compute(
        self,
        ctx: RuntimeContext,
        name: str,
        state: MotorSystemState,  # noqa: ARG002
        scale: float = 1.0,
    ) -> list[Action]:
        sample = getattr(self._sampler, f"sample_{self._HEADINGS[name]}")
        action = sample(self._agent_id, ctx.rng)
        action.rotation_degrees = self.DEFAULT_ROTATION_DEGREES * scale
        return [action]

    def feedback(self, chosen: list[Action]) -> None:
        """Point the wrapped policy's undo at the user's action.

        The informed policies never keep their last action around, so there is no
        `last_action` to assign to. What they keep instead is `_undo_action`: the
        already-inverted action they will take to step back on to the object the next
        time they find themselves off it. So the equivalent of recording the last
        action is to store its inverse, which is what `fixme_undo_last_action` builds.

        Args:
            chosen: The actions returned by `compute` that will be executed.
        """
        if not isinstance(self._policy, (InformedPolicy, InformedPolicyRandomWalk)):
            return
        self._policy._undo_action = fixme_undo_last_action(chosen[-1])


class SurfaceInteractivePolicy:
    """Interactive adapter for the surface policies.

    The user picks a tangential exploration heading in the agent's reference frame
    (up, down, left, or right); the policy's forward/orient corrections run
    automatically every step. The choice is offered only on the tangential step of
    the surface policy's action cycle.
    """

    _HEADINGS: ClassVar[dict[str, VectorXYZ]] = {
        "up": (0.0, 1.0, 0.0),
        "down": (0.0, -1.0, 0.0),
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
    }

    # The interactive default tangential step, in meters. Matches the surface sampler's
    # `translation_distance` default. Scaled by the slider multiplier; declared here
    # rather than read from the sampler.
    DEFAULT_DISTANCE: ClassVar[float] = 0.004

    def __init__(self, policy: SurfacePolicy) -> None:
        """Initialize the adapter from a surface policy.

        Args:
            policy: The wrapped surface policy.
        """
        self._policy = policy
        self._sampler = policy.action_sampler
        self._agent_id = policy.agent_id

    def awaits_choice(self, proposed: list[Action]) -> bool:
        return bool(proposed) and isinstance(proposed[0], MoveTangentially)

    def compute(
        self,
        ctx: RuntimeContext,
        name: str,
        state: MotorSystemState,
        scale: float = 1.0,
    ) -> list[Action]:
        agent_frame_heading = self._HEADINGS[name]
        action = self._sampler.sample_move_tangentially(self._agent_id, ctx.rng)
        action.direction = tuple(
            qt.rotate_vectors(state[self._agent_id].rotation, agent_frame_heading)
        )
        action.distance = self.DEFAULT_DISTANCE * scale
        return [action]

    def feedback(self, chosen: list[Action]) -> None:
        """Do nothing, because the surface policy's record is already correct.

        The surface policy does not recover from off-object by reversing its last action
        the way the informed policies do. When it falls off the object it runs its
        `touch_object` search to find its way back to the object, which reads the
        current state rather than any record of what actions were executed. So there
        is nothing here for the user's choice to correct.

        Args:
            chosen: The actions returned by `compute` that will be executed. Unused.
        """


def interactive_policy_for(model: Monty) -> InteractivePolicy:
    """Build the interactive adapter matching the model's motor policy.

    Args:
        model: The Monty model whose motor system exposes the policy.

    Returns:
        The interactive adapter for the policy type.

    Raises:
        ValueError: When the motor system exposes no policy, or no adapter exists
            for the policy type.
    """
    policy_selector = model.motor_system._policy_selector
    if isinstance(policy_selector, SinglePolicySelector):
        policy = policy_selector._policy
    if isinstance(policy_selector, DistantPolicySelector):
        # The goal-driven jump_to_goal / look_at_goal policies carry no action
        # sampler, so the default fall-back policy is the one to expose.
        policy = policy_selector._default
    if policy is None:
        raise ValueError(
            "Interactive plotter requires a policy, but the motor system's "
            "selector exposes no policy."
        )
    if isinstance(policy, SurfacePolicy):
        return SurfaceInteractivePolicy(policy)
    if isinstance(policy, (BasePolicy, InformedPolicyRandomWalk)):
        return SampledInteractivePolicy(policy)
    raise ValueError(f"Interactive plotter has no adapter for {type(policy).__name__}.")


def goal_driven(model: Monty) -> bool:
    """Whether the model's proposed actions this step enact a goal.

    Reads the policy selector's telemetry for the policy that produced this step's
    actions. With a `DistantPolicySelector` that is any policy other than the default
    fall-back: `JumpToGoal` for a learning module's goal (including the undo of a
    previous jump) or `LookAtGoal` for a sensor module's goal. A `SinglePolicySelector`
    hands its goals to a policy that ignores them, so nothing is ever goal-driven.

    Args:
        model: The Monty model whose motor system exposes the policy selector.

    Returns:
        True when the step's actions were chosen by a goal-driven policy.
    """
    policy_selector = model.motor_system._policy_selector
    if not isinstance(policy_selector, DistantPolicySelector):
        return False
    selected = policy_selector._selected_policies
    return bool(selected) and selected[-1] is not policy_selector._default
