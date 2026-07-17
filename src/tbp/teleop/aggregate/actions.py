# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Building a frame's `actions` from what the model computed for this step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tbp.teleop.aggregate.plain import plain
from tbp.teleop.frames import AgentID

if TYPE_CHECKING:
    from tbp.monty.frameworks.actions.actions import Action

    from tbp.teleop.frames import ActionFrame, ActionsFrame


class ActionAggregator(Protocol):
    """Builds a frame's `actions` from what the model computed for this step."""

    def __call__(self, actions: list[Action]) -> ActionsFrame:
        """Aggregate one step's actions.

        Args:
            actions: The actions the model computed for this step.

        Returns:
            The frame's `actions`.
        """
        ...


class NoOpActionAggregator(ActionAggregator):
    """An `ActionAggregator` that does nothing."""

    def __call__(self, actions: list[Action]) -> ActionsFrame:  # noqa: ARG002
        """Return an empty `ActionsFrame`.

        Args:
            actions: The actions the model computed for this step.

        Returns:
            An empty `ActionsFrame`.
        """
        return {}


class DefaultActionAggregator(ActionAggregator):
    """Describes each action the model proposes to take next."""

    def __call__(self, actions: list[Action]) -> ActionsFrame:
        """Aggregate one step's actions.

        Args:
            actions: The actions the model computed for this step.

        Returns:
            An `ActionsFrame` describing the proposed actions.
        """
        return {"actions": [self._action(action) for action in actions]}

    @staticmethod
    def _action(action: Action) -> ActionFrame:
        """Describe one action.

        Args:
            action: The action to describe.

        Returns:
            The action's name, agent, and its own parameters.
        """
        # Monty's own `ActionJSONEncoder` encodes an action as `dict(action)`, which
        # yields its name followed by its parameters, so this is its own idea of what
        # an action is made of rather than a guess at it.
        params = dict(action)
        action_name = params.pop("action", type(action).__name__)
        agent_id = params.pop("agent_id", "")
        return {
            "action_name": str(action_name),
            "agent_id": AgentID(str(agent_id)),
            "params": {key: plain(value) for key, value in params.items()},
        }
