# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The actions section of a frame: what the model will do next."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from tbp.teleop.frames.observations import AgentID


class ActionFrame(TypedDict):
    """One action the model proposes to take next.

    Monty's `Action.__iter__` yields its name followed by whatever parameters its own
    class happens to carry, which is what `ActionJSONEncoder` encodes and what this
    mirrors. The name and the agent are declared because every action has them, and are
    always present; the parameters are held apart in a bag because which ones exist is
    decided by the action's class rather than by anything here.

    Attributes:
        action_name: Monty's snake_case name for the action, e.g. `move_tangentially`.
        agent_id: The agent that would take it.
        params: The action's own parameters. Monty types these as plain data, a
            `VectorXYZ` or a `QuaternionWXYZ` being tuples of floats, but hands over a
            `quaternion.quaternion` for a rotation and an `ndarray` for a location. An
            aggregator is what reconciles the two, since a frame may hold only plain
            containers, arrays, and primitives.
    """

    action_name: str
    agent_id: AgentID
    params: dict[str, Any]


class ActionsFrame(TypedDict, total=False):
    """What the model will do next.

    Attributes:
        actions: The actions the model computed for this step.
    """

    actions: list[ActionFrame]
