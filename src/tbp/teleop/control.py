# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Obeying a driver's commands, and turning them into the actions the model runs.

This is the half of teleoperation that cannot leave the process. It holds the mode --
whether the experiment steps on command or runs on its own -- and turns a command's
spelled-out actions or its goal into the actions Monty executes. A goal is a place,
which the motor system works out how to reach from the live motor state, and that state
cannot cross a wire, so the driver sends the place and this does the reaching.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from tbp.monty.cmp import Goal
from tbp.monty.frameworks.actions.actions import ActionJSONDecoder

from tbp.teleop.commands import CommandOperation, CommandResult, RunMode
from tbp.teleop.frames import Pursue

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations

    from tbp.teleop.commands import Command
    from tbp.teleop.frames import ActionFrame, Frame, GoalFrame
    from tbp.teleop.wire import CommandServer

# What a driver's goal calls itself. Monty reads a goal's sender to decide what to do
# about it, not to attribute it, so this only ever has to be recognizable in a log.
DRIVER = "teleop"

# How often the wait for a command wakes up, in seconds. Blocking in ZeroMQ's C code
# would leave the process deaf to Ctrl-C until a command arrived, so the wait is a loop
# of short polls instead: long enough to idle cheaply, short enough to stay responsive.
POLL = 0.25

# Monty's decoder is the exact inverse of the encoder an `ActionFrame` mirrors, so a
# driver's action is rebuilt by the same code that would read one back out of a log, and
# goes on being right as Monty gains actions. It also restores the tuples Monty's
# `VectorXYZ` and `QuaternionWXYZ` are declared as, which a wire would otherwise have
# flattened to lists.
_ACTIONS = ActionJSONDecoder()


def _pose(features: dict | None) -> dict | None:
    """Restore a goal's pose to the arrays Monty's `Goal` validates.

    A wire flattens everything to lists, but `Goal` insists `pose_vectors` be an array
    -- it reads its `.shape` -- so it is put back before Monty ever sees it. A goal
    without a pose (a look-at) is left alone.

    Args:
        features: The goal's morphological features, or `None`.

    Returns:
        The features with `pose_vectors` as an array, or `None`.
    """
    if not features or "pose_vectors" not in features:
        return features
    return {
        **features,
        "pose_vectors": np.asarray(features["pose_vectors"], dtype=float),
    }


class Teleoperator:
    """Runs the experiment at a driver's command, in one of two modes.

    In step mode the experiment takes one step per `STEP` command and waits between, so
    a driver decides every move. In continuous mode it advances on its own, pausing an
    optional interval between steps and checking for a command as it goes, so a driver
    can set it running and only step in to change course. A `SWITCH_MODE` command moves
    between the two; `QUIT` ends the run.

    A `STEP` may carry the driver's own actions, or a goal for the motor system to
    reach, or nothing -- in which case the model runs what it computed. So a driver can
    drive as tightly as spelling out each action, or as loosely as letting the model run
    itself a step at a time.
    """

    _server: CommandServer
    _mode: RunMode
    _interval: float

    def __init__(
        self,
        server: CommandServer,
        mode: RunMode = RunMode.STEP,
        interval: float = 0.0,
    ) -> None:
        """Initialize the teleoperator.

        Args:
            server: The control channel commands arrive on.
            mode: The mode to start in.
            interval: Seconds between steps in continuous mode, `0.0` for as fast as the
                experiment runs.
        """
        self._server = server
        # Coerced, since config hands over a plain string rather than the enum member.
        self._mode = RunMode(mode)
        self._interval = interval

    def advance(
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        frame: Frame,
        actions: list[Action],
    ) -> list[Action]:
        """Decide this step's actions, obeying commands according to the mode.

        The frame has already been published, so a driver has seen the step it is
        answering. In step mode this blocks for a command; in continuous mode it
        advances on its own, obeying any command that arrives in the meantime.

        Args:
            ctx: The runtime context, which the motor system samples from for a goal.
            monty: The Monty model being driven.
            observations: The observations from this step, which a goal needs.
            frame: The step's snapshot, whose `episode` and `step` the reply reports.
            actions: The actions the model computed for this step.

        Raises `StopIteration` when the driver quits, after timing the model out so the
        episode still logs cleanly; it propagates up from `_obey`, so it is named here
        rather than in a `Raises:` section.

        Returns:
            The actions to run: the driver's, or the model's own when a step is taken
            without the driver naming any.
        """
        if self._mode == RunMode.CONTINUOUS:
            return self._advance_continuous(ctx, monty, observations, frame, actions)
        return self._advance_step(ctx, monty, observations, frame, actions)

    def close(self) -> None:
        """Close the control channel."""
        self._server.close()

    def _advance_step(
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        frame: Frame,
        actions: list[Action],
    ) -> list[Action]:
        """Wait for an instruction, then advance.

        Blocks -- as a loop of short polls, so the process stays interruptible --
        until a command says to advance, quit, or start running continuously.

        Args:
            ctx: The runtime context.
            monty: The Monty model being driven.
            observations: The observations from this step.
            frame: The step's snapshot.
            actions: The actions the model computed for this step.

        Returns:
            The actions to run.
        """
        while True:
            command = self._server.receive(timeout=POLL)
            if command is None:
                continue
            advanced = self._obey(ctx, monty, observations, frame, actions, command)
            if advanced is not None:
                return advanced
            # A mode change: begin running now if it switched to continuous, otherwise
            # keep waiting for the next instruction.
            if self._mode == RunMode.CONTINUOUS:
                return actions

    def _advance_continuous(
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        frame: Frame,
        actions: list[Action],
    ) -> list[Action]:
        """Advance on the experiment's own clock, obeying any command that arrives.

        Waits up to the interval before advancing, but no command is left waiting that
        long: it is checked for every poll, and acted on at once.

        Args:
            ctx: The runtime context.
            monty: The Monty model being driven.
            observations: The observations from this step.
            frame: The step's snapshot.
            actions: The actions the model computed for this step.

        Returns:
            The actions to run.
        """
        deadline = time.monotonic() + self._interval
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # The interval is up; take any command already waiting, else advance.
                command = self._server.receive(timeout=0)
                if command is None:
                    return actions
            else:
                command = self._server.receive(timeout=min(remaining, POLL))
                if command is None:
                    continue
            advanced = self._obey(ctx, monty, observations, frame, actions, command)
            if advanced is not None:
                return advanced
            # A mode change: drop into step mode's wait, or reset the clock for the new
            # interval and keep running.
            if self._mode == RunMode.STEP:
                return self._advance_step(ctx, monty, observations, frame, actions)
            deadline = time.monotonic() + self._interval

    def _obey(  # noqa: PLR0913
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        frame: Frame,
        actions: list[Action],
        command: Command,
    ) -> list[Action] | None:
        """Carry out one command, and acknowledge it.

        Every command received is answered before the next, which the reply socket
        requires and a driver relies on.

        Args:
            ctx: The runtime context.
            monty: The Monty model being driven.
            observations: The observations from this step.
            frame: The step's snapshot.
            actions: The actions the model computed for this step.
            command: The command to carry out.

        Returns:
            The actions to advance with, or `None` when the command only changed the
            mode and the caller should decide what to do next.

        Raises:
            StopIteration: When the command is `QUIT`, after timing the model out.
        """
        if command.operation == CommandOperation.QUIT:
            monty.deal_with_time_out()
            self._server.acknowledge(self._status(frame, stopped=True))
            raise StopIteration
        if command.operation == CommandOperation.SET_RUN_MODE:
            if command.run_mode:
                self._mode = RunMode(command.run_mode)
            self._interval = command.interval
            self._server.acknowledge(self._status(frame))
            return None
        # A step, and anything unrecognized, advances rather than wedging the run.
        advanced = self._step_actions(ctx, monty, observations, command, actions)
        self._server.acknowledge(self._status(frame))
        return advanced

    def _step_actions(
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        command: Command,
        actions: list[Action],
    ) -> list[Action]:
        """Work out the actions a step command asks for.

        Args:
            ctx: The runtime context.
            monty: The Monty model being driven.
            observations: The observations from this step.
            command: The step command.
            actions: The actions the model computed for this step.

        Returns:
            The driver's spelled-out actions, the motor system's answer to the goals, or
            the model's own actions when the command named neither.
        """
        if command.actions:
            return [self._rebuild(action) for action in command.actions]
        if command.goals:
            return self._pursue(ctx, monty, observations, command.goals)
        return actions

    def _status(self, frame: Frame, *, stopped: bool = False) -> CommandResult:
        """Say where the run now stands, for a command's acknowledgement.

        Args:
            frame: The step's snapshot, for where the run is.
            stopped: Whether the run has just been quit.

        Returns:
            The acknowledgement.
        """
        return CommandResult(
            run_mode=self._mode,
            episode=frame.episode,
            step=frame.step,
            stopped=stopped,
        )

    def _pursue(
        self,
        ctx: RuntimeContext,
        monty: Monty,
        observations: Observations,
        goals: list[GoalFrame],
    ) -> list[Action]:
        """Hand the motor system the driver's goals, and run it again for its answer.

        A goal is not something a step hook can answer with, because by the time the
        hook is called the motor system has already run and produced the actions being
        overridden. So it is run a second time, with the driver's goals in place of the
        model's own.

        This is not free. The motor system appends to its action sequence on every call,
        so a driven step leaves two entries: the actions the model proposed, and the
        actions the goals produced. Only the second is ever executed, so the first is
        dropped here rather than left to misreport the run to everything that reads the
        sequence afterwards. Surface policies also append to their telemetry, which is
        left doubled; nothing reads it yet, and reaching further into the motor system's
        privates to tidy it would cost more than it is worth.

        The proper fix is a seam between Monty proposing goals and its motor system
        acting on them, where a driver's goals would simply join the model's own and the
        motor system would run once. That is a change to Monty, not to this.

        Args:
            ctx: The runtime context.
            monty: The Monty model being driven.
            observations: The observations from this step.
            goals: The places the driver wants reached.

        Returns:
            Whatever the motor system decided the goals are worth doing about, which may
            be nothing at all.
        """
        pursued = [self._goal(goal) for goal in goals]

        # The goals are passed straight in, so Monty's own `_goals` is left alone: it is
        # only read by the step it has already taken.
        actions = monty.motor_system(
            ctx,
            observations,
            monty.motor_system.action_sequence[-1][1],
            monty.sensor_module_outputs[0],
            pursued,
        )
        del monty.motor_system._action_sequence[-2]  # noqa: SLF001
        return actions

    @staticmethod
    def _goal(goal: GoalFrame) -> Goal:
        """Build Monty's own `Goal` from the driver's, as it wants it.

        A `GoalFrame` mirrors Monty's `Goal`, so its parts are handed straight over. The
        sender picks what the motor system does about the goal rather than attributing
        it: a GSG's goal is jumped to, an SM's is looked at. A jump needs a pose to
        face, which the driver supplies in `morphological_features`; a look-at, the
        default, needs only the place.

        Args:
            goal: The place the driver wants reached.

        Returns:
            Monty's own goal.
        """
        return Goal(
            location=np.asarray(goal["location"], dtype=float),
            # The wire flattens the pose to lists, but Monty's `Goal` wants its
            # `pose_vectors` as an array, so they are put back before it sees them.
            morphological_features=_pose(goal.get("morphological_features")),
            non_morphological_features=goal.get("non_morphological_features"),
            # Monty acts on the most confident goal it holds, so this competes with the
            # model's own goal state generators rather than silencing them.
            confidence=goal.get("confidence", 1.0),
            use_state=True,
            sender_id=DRIVER,
            sender_type=goal.get("sender_type", Pursue.LOOK_AT),
            goal_tolerances=None,
        )

    @staticmethod
    def _rebuild(action: ActionFrame) -> Action:
        """Turn one spelled-out action back into the action Monty runs.

        Args:
            action: The action as the driver spelled it.

        Returns:
            The rebuilt action.
        """
        # A frame holds an action's parameters apart from its name, because which ones
        # exist is the action class's business; Monty's decoder wants them all together,
        # the way its encoder writes them.
        return _ACTIONS.object_hook(
            {
                # Monty's key is `action`; the frame's is `action_name`. This is the
                # seam between the two, so both spellings appear once, here.
                "action": action["action_name"],
                "agent_id": action["agent_id"],
                **action["params"],
            }
        )
