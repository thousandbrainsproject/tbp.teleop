# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The control channel's vocabulary: what a driver sends, and what comes back.

Kept apart from the frame schema because the two channels want opposite things --
telemetry is a lossy fan-out of frames, control is a reliable exchange of commands. A
command is a request; a `CommandResult` is its small acknowledgement.

There is a class per operation rather than one command with a discriminator, so each
command carries exactly its own fields and nothing else. Every command still names its
`operation`, as a `ClassVar`, so the receiving side can dispatch on it and the codec can
tell which class to rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Union

if TYPE_CHECKING:
    from tbp.teleop.frames import ActionFrame, GoalFrame


class CommandOperation(str, Enum):
    """What a command carries out.

    A command names one, so the receiving side dispatches on it without inspecting the
    rest of the command, and the codec rebuilds the right command class from it. A `str`
    enum, so it crosses the wire as its plain value.
    """

    STEP = "STEP"  # take one step, optionally with the driver's own actions or goals
    SET_RUN_MODE = "SET_RUN_MODE"  # run in step or continuous mode from now on
    QUIT = "QUIT"  # end the run here


class RunMode(str, Enum):
    """How the experiment advances.

    In `STEP` mode it takes one step per `StepCommand` and waits between; in
    `CONTINUOUS` mode it advances on its own, checking for commands as it goes. A `str`
    enum, so it crosses the wire as its plain value.
    """

    STEP = "STEP"
    CONTINUOUS = "CONTINUOUS"


@dataclass(frozen=True)
class StepCommand:
    """Take one step.

    Carries the driver's own `actions` or `goals`; carrying neither means "advance with
    whatever the model computed". Only spelled-out `actions` are a literal instruction
    -- a goal is a place, which the model's motor system works out how to reach, because
    that needs the live motor state and cannot cross a wire.

    Attributes:
        actions: The actions to run, spelled out. Empty to use the model's own.
        goals: Places for the motor system to work out how to reach.
    """

    actions: list[ActionFrame] = field(default_factory=list)
    goals: list[GoalFrame] = field(default_factory=list)
    operation: ClassVar[CommandOperation] = CommandOperation.STEP


@dataclass(frozen=True)
class SetRunModeCommand:
    """Run in step or continuous mode from now on.

    Attributes:
        run_mode: The mode to run in.
        interval: For `RunMode.CONTINUOUS`: seconds to wait between steps, `0.0` for as
            fast as the experiment runs.
    """

    run_mode: RunMode
    interval: float = 0.0
    operation: ClassVar[CommandOperation] = CommandOperation.SET_RUN_MODE


@dataclass(frozen=True)
class QuitCommand:
    """End the run here."""

    operation: ClassVar[CommandOperation] = CommandOperation.QUIT


# The commands a driver may send, as one type for the code that carries any of them.
Command = Union[StepCommand, SetRunModeCommand, QuitCommand]

# What the codec rebuilds a decoded command into, keyed by the operation it carries.
COMMANDS: dict[str, type] = {
    command.operation.value: command
    for command in (StepCommand, SetRunModeCommand, QuitCommand)
}


@dataclass(frozen=True)
class CommandResult:
    """What the experiment reports back when it has obeyed a command.

    A small acknowledgement, not a frame: the frame the step produced travels on the
    telemetry channel, and this only tells the driver where the run now stands, so it
    knows the command landed and what to do next.

    Attributes:
        run_mode: The mode the experiment is in, after the command.
        episode: The episode it is on, or `None` before the first step.
        step: The step it is on within the episode, or `None` before the first step.
        stopped: Whether the run has ended, in answer to a `QuitCommand`.
    """

    run_mode: RunMode
    episode: int | None = None
    step: int | None = None
    stopped: bool = False
