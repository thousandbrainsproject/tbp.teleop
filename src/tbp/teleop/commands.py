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

This module is a single working `Command` with an `operation` discriminator. Splitting
it into a class per operation is the next step, not this one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tbp.teleop.frames import ActionFrame, GoalFrame


class CommandOperation(str, Enum):
    """What a command carries out.

    A command names one, so the receiving side dispatches on it without inspecting the
    rest of the command. A `str` enum, so it crosses the wire as its plain value.
    """

    STEP = "STEP"  # take one step, optionally with the driver's own actions or goal
    SET_RUN_MODE = "SET_RUN_MODE"  # run in step or continuous mode from now on
    QUIT = "QUIT"  # end the run here


class RunMode(str, Enum):
    """How the experiment advances.

    In `STEP` mode it takes one step per `CommandOperation.STEP` command and waits
    between; in `CONTINUOUS` mode it advances on its own, checking for commands as it
    goes. A `str` enum, so it crosses the wire as its plain value.
    """

    STEP = "STEP"
    CONTINUOUS = "CONTINUOUS"


@dataclass(frozen=True)
class Command:
    """An instruction a driver sends the experiment, on the control channel.

    A command is a request, structured so the receiving side can act on it by its
    `operation` alone: `STEP` to advance, `SET_RUN_MODE` to change how the experiment
    runs, `QUIT` to end it. The other fields are read only by the operation that needs
    them, and ignored otherwise.

    A `STEP` may carry the driver's own `actions` or a `goal`; carrying neither means
    "advance with whatever the model computed". Only spelled-out `actions` are a literal
    instruction -- a `goal` is a place, which the model's motor system works out how to
    reach, because that needs the live motor state and cannot cross a wire.

    Attributes:
        operation: What to do.
        run_mode: For `SET_RUN_MODE`: the mode to run in. `None` otherwise.
        interval: For a switch to `RunMode.CONTINUOUS`: seconds to wait between steps,
            `0.0` for as fast as the experiment runs.
        actions: For `STEP`: the actions to run, spelled out. `None` to use the model's.
        goal: For `STEP`: a place for the motor system to work out how to reach.
    """

    operation: CommandOperation
    run_mode: RunMode | None = None
    interval: float = 0.0
    actions: list[ActionFrame] | None = None
    goal: GoalFrame | None = None


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
        stopped: Whether the run has ended, in answer to `CommandOperation.QUIT`.
    """

    run_mode: RunMode
    episode: int | None = None
    step: int | None = None
    stopped: bool = False
