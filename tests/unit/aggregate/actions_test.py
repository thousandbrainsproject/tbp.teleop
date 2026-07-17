# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import unittest
from typing import Any

import quaternion as qt

from tbp.teleop.aggregate.actions import DefaultActionAggregator, NoOpActionAggregator


class FakeAction:
    """Stands in for a Monty `Action`, which yields its name then its parameters."""

    def __init__(self, name: str, **params: Any) -> None:  # noqa: ANN401
        """Build an action of `name` carrying `params`."""
        self.name = name
        self.agent_id = "agent_id_0"
        self.params = params

    def __iter__(self) -> Any:  # noqa: ANN401, D105
        yield "action", self.name
        yield "agent_id", self.agent_id
        yield from self.params.items()


class NoOpActionAggregatorTest(unittest.TestCase):
    def test_aggregates_nothing(self) -> None:
        self.assertEqual(NoOpActionAggregator()([FakeAction("move_forward")]), {})


class DefaultActionAggregatorTest(unittest.TestCase):
    def aggregate(self, actions: list) -> list:
        return DefaultActionAggregator()(actions)["actions"]

    def test_describes_each_proposed_action(self) -> None:
        """Monty's own `dict(action)` decides what an action is made of."""
        actions = self.aggregate([FakeAction("move_tangentially", distance=0.004)])

        self.assertEqual(
            actions,
            [
                {
                    "action_name": "move_tangentially",
                    "agent_id": "agent_id_0",
                    "params": {"distance": 0.004},
                }
            ],
        )

    def test_a_rotation_is_reduced_to_the_floats_monty_says_it_is(self) -> None:
        """Monty types a rotation as four floats but hands over a quaternion.

        numpy-quaternion registers it as a numpy scalar type whose `item` returns the
        quaternion again, so it would reach the codec and fail there. An aggregator is
        what keeps a frame to plain data.
        """
        action = FakeAction(
            "set_sensor_rotation", rotation_quat=qt.quaternion(1, 0, 0, 0)
        )

        params = self.aggregate([action])[0]["params"]

        self.assertEqual(params["rotation_quat"], [1.0, 0.0, 0.0, 0.0])

    def test_describes_every_action_in_order(self) -> None:
        actions = self.aggregate([FakeAction("move_forward"), FakeAction("turn_left")])

        self.assertEqual(
            [action["action_name"] for action in actions],
            ["move_forward", "turn_left"],
        )

    def test_reports_no_actions_at_all(self) -> None:
        self.assertEqual(self.aggregate([]), [])


if __name__ == "__main__":
    unittest.main()
