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
from typing import TYPE_CHECKING, Any

import numpy as np

from tbp.teleop.commands import (
    CommandResult,
    QuitCommand,
    RunMode,
    SetRunModeCommand,
    StepCommand,
)
from tbp.teleop.control import Teleoperator
from tbp.teleop.frames import Frame, Pursue

if TYPE_CHECKING:
    from tbp.teleop.commands import Command


class FakeMotorSystem:
    """Stands in for the motor system, which records every call it is put through.

    Real ones append to their action sequence whenever they are called, which is what
    makes running one a second time cost something.
    """

    def __init__(self) -> None:
        """Start out having been called once, as a stepped model's would have been."""
        self._action_sequence: list[tuple] = [(["proposed"], "state")]
        self.goals: list = []

    @property
    def action_sequence(self) -> list[tuple]:
        return self._action_sequence

    def __call__(
        self,
        ctx: Any,  # noqa: ANN401, ARG002
        observations: Any,  # noqa: ANN401, ARG002
        state: Any,  # noqa: ANN401
        percept: Any,  # noqa: ANN401, ARG002
        goals: Any,  # noqa: ANN401
    ) -> list[str]:
        self.goals.append(goals)
        self._action_sequence.append((["pursued"], state))
        return ["pursued"]


class FakeMonty:
    """Stands in for the model: enough of it to be steered, and to notice being told."""

    def __init__(self) -> None:
        """Stand still, mid-episode."""
        self.is_done = False
        self.motor_system = FakeMotorSystem()
        self.sensor_module_outputs = ["percept"]
        self.timed_out = False

    def deal_with_time_out(self) -> None:
        self.timed_out = True


class FakeServer:
    """Stands in for the `CommandServer`, handing out a queued script of commands.

    Each `receive` returns the next command, or `None` once the script runs dry, which
    is how a poll that finds nothing is spelled. Every acknowledgement is recorded, so a
    test can check the run's reported state.
    """

    def __init__(self, commands: list[Command | None]) -> None:
        """Answer `receive` with each of `commands` in turn, then with `None`."""
        self._commands = list(commands)
        self.acknowledged: list[CommandResult] = []
        self.closed = False

    def receive(self, timeout: float | None = None) -> Command | None:  # noqa: ARG002
        return self._commands.pop(0) if self._commands else None

    def acknowledge(self, result: CommandResult) -> None:
        self.acknowledged.append(result)

    def close(self) -> None:
        self.closed = True


class TeleoperatorTest(unittest.TestCase):
    def frame(self) -> Frame:
        return Frame(episode=0, step=7)

    def advance(
        self,
        commands: list[Command | None],
        *,
        mode: str = RunMode.STEP,
        interval: float = 0.0,
    ) -> tuple[list, FakeServer, FakeMonty]:
        # Runs one step through a teleoperator scripted with `commands`.
        server = FakeServer(commands)
        monty = FakeMonty()
        teleoperator = Teleoperator(server, mode=mode, interval=interval)
        actions = teleoperator.advance(
            "ctx", monty, "observations", self.frame(), ["proposed"]
        )
        return actions, server, monty


class StepModeTest(TeleoperatorTest):
    def test_a_bare_step_advances_with_the_models_own_actions(self) -> None:
        """Step mode waits, then runs what the model computed when told to step."""
        actions, server, _ = self.advance([StepCommand()])

        self.assertEqual(actions, ["proposed"])
        self.assertEqual(server.acknowledged[0].run_mode, RunMode.STEP)
        self.assertEqual(server.acknowledged[0].step, 7)

    def test_a_step_runs_the_drivers_spelled_out_actions(self) -> None:
        command = StepCommand(
            actions=[
                {
                    "action_name": "turn_left",
                    "agent_id": "agent_id_0",
                    "params": {"rotation_degrees": 5.0},
                }
            ],
        )

        actions, _, _ = self.advance([command])

        self.assertEqual([type(a).__name__ for a in actions], ["TurnLeft"])
        self.assertEqual(actions[0].rotation_degrees, 5.0)

    def test_it_keeps_waiting_past_a_poll_that_finds_nothing(self) -> None:
        """A quiet poll is no instruction; the step blocks until a real one comes."""
        actions, server, _ = self.advance([None, None, StepCommand()])

        self.assertEqual(actions, ["proposed"])
        self.assertEqual(len(server.acknowledged), 1)

    def test_quitting_times_the_model_out_and_stops(self) -> None:
        server = FakeServer([QuitCommand()])
        monty = FakeMonty()
        teleoperator = Teleoperator(server, mode=RunMode.STEP)

        with self.assertRaises(StopIteration):
            teleoperator.advance("ctx", monty, "obs", self.frame(), ["proposed"])

        self.assertTrue(monty.timed_out)
        self.assertTrue(server.acknowledged[0].stopped)


class GoalTest(TeleoperatorTest):
    def test_a_goal_is_handed_to_the_motor_system_to_work_out(self) -> None:
        """A goal names a place; the motor system decides how to get there."""
        command = StepCommand(goals=[{"location": [0.1, 1.5, 0.2]}])

        actions, _, monty = self.advance([command])

        self.assertEqual(actions, ["pursued"])
        [goals] = monty.motor_system.goals
        np.testing.assert_allclose(goals[0].location, [0.1, 1.5, 0.2])

    def test_a_goal_says_what_should_be_done_about_it(self) -> None:
        """Monty reads a goal's sender to pick jumping there over looking at it."""
        for sender, expected in ((Pursue.JUMP_TO, "GSG"), (Pursue.LOOK_AT, "SM")):
            with self.subTest(sender=sender):
                command = StepCommand(
                    goals=[{"location": [0.0, 0.0, 0.0], "sender_type": sender}]
                )

                _, _, monty = self.advance([command])

                [goals] = monty.motor_system.goals
                self.assertEqual(goals[0].sender_type, expected)

    def test_a_goals_pose_is_restored_to_the_array_monty_wants(self) -> None:
        """The wire flattens a pose to lists; Monty's `Goal` reads `pose_vectors.shape`.

        So the pose is put back to an array before Monty sees it -- the difference
        between a jump-to goal working over the wire and crashing in the motor system.
        """
        command = StepCommand(
            goals=[
                {
                    "location": [0.0, 0.0, 0.0],
                    "sender_type": Pursue.JUMP_TO,
                    "morphological_features": {
                        # As a wire delivers it: plain nested lists, not an array.
                        "pose_vectors": [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0, 0, 0]],
                        "pose_fully_defined": None,
                        "on_object": 1,
                    },
                }
            ]
        )

        actions, _, monty = self.advance([command])

        self.assertEqual(actions, ["pursued"])
        [goals] = monty.motor_system.goals
        vectors = goals[0].morphological_features["pose_vectors"]
        self.assertIsInstance(vectors, np.ndarray)
        self.assertEqual(vectors.shape, (3, 3))

    def test_the_step_leaves_one_entry_in_the_action_sequence(self) -> None:
        """Running the motor system twice would else record a step that never ran."""
        command = StepCommand(goals=[{"location": [0.0, 0.0, 0.0]}])

        _, _, monty = self.advance([command])

        self.assertEqual(len(monty.motor_system.action_sequence), 1)
        self.assertEqual(monty.motor_system.action_sequence[-1][0], ["pursued"])


class SwitchModeTest(TeleoperatorTest):
    def test_switching_to_continuous_begins_running(self) -> None:
        """From step mode, switching to continuous advances this step at once."""
        command = SetRunModeCommand(run_mode=RunMode.CONTINUOUS)

        actions, server, _ = self.advance([command])

        self.assertEqual(actions, ["proposed"])
        self.assertEqual(server.acknowledged[0].run_mode, RunMode.CONTINUOUS)

    def test_continuous_advances_on_its_own_when_no_command_waits(self) -> None:
        """With nothing to obey and no interval, it runs the model's own actions."""
        actions, server, _ = self.advance([], mode=RunMode.CONTINUOUS)

        self.assertEqual(actions, ["proposed"])
        self.assertEqual(server.acknowledged, [])

    def test_continuous_obeys_a_command_that_arrives(self) -> None:
        """A step's actions can be injected while the experiment flows."""
        command = StepCommand(
            actions=[
                {
                    "action_name": "move_forward",
                    "agent_id": "agent_id_0",
                    "params": {"distance": 0.01},
                }
            ],
        )

        actions, _, _ = self.advance([command], mode=RunMode.CONTINUOUS)

        self.assertEqual([type(a).__name__ for a in actions], ["MoveForward"])

    def test_switching_back_to_step_stops_flowing_and_waits(self) -> None:
        """A switch to step mode drops into the wait, and the next command advances."""
        actions, server, _ = self.advance(
            [
                SetRunModeCommand(run_mode=RunMode.STEP),
                StepCommand(),
            ],
            mode=RunMode.CONTINUOUS,
        )

        self.assertEqual(actions, ["proposed"])
        self.assertEqual(
            [ack.run_mode for ack in server.acknowledged],
            [RunMode.STEP, RunMode.STEP],
        )


class ClosingTest(TeleoperatorTest):
    def test_closing_closes_the_server(self) -> None:
        server = FakeServer([])

        Teleoperator(server).close()

        self.assertTrue(server.closed)


if __name__ == "__main__":
    unittest.main()
