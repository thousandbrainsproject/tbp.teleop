# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tbp.monty.frameworks.experiments.hooks import StepHook
from typing_extensions import Self

if TYPE_CHECKING:
    from tbp.monty.cmp import Message
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations

    from tbp.teleop.plotter import Plotter


class Default(StepHook):
    _initialized: bool = False
    _plotter: Plotter

    def __init__(self, plotter: Plotter) -> None:
        self._initialized = False
        self._plotter = plotter

    def __call__(
        self: Self,
        ctx: RuntimeContext,
        monty: Monty,
        supervised_lm_ids: list[str],
        step: int,
        observations: Observations,
        actions: list[Action],
        experiment: Any | None = None,
    ) -> list[Action]:
        """Draw the current state and, if interactive, return the user-chosen action.

        The user only overrides an action while the agent is on the object and the
        step is a user choice step (e.g. the surface policy's tangential step); the
        policy's automatic corrections and off-object self-positioning run unchanged.

        Args:
            ctx: The runtime context supplying the random state.
            monty: The Monty model whose sensor and learning modules are plotted.
            supervised_lm_ids: The list of supervised learning module IDs.
            step: The index of the current step within the episode.
            observations: The observations from the most recent step.
            actions: The actions returned by the model for the next step.
            experiment: The Monty experiment, if any, that is running this step.

        Returns:
            The actions to execute next, unchanged or overridden by the user.
        """
        if not self._initialized:
            self._plotter.initialize(monty, supervised_lm_ids)
            self._initialized = True

        self._plotter.update(observations, step)

        if not self._plotter.interactive or monty.is_done:
            return actions

        percept: Message = monty.sensor_module_outputs[0]
        on_object = percept.get_on_object()
        user_choice = self._plotter.awaits_choice(actions)
        if not on_object or not user_choice:
            return actions

        return self._plotter.override_action(ctx, actions)

    def close(self) -> None:
        pass
