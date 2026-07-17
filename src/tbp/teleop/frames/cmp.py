# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The parts of Monty's cortical messaging protocol a driver may set: goals."""

from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict


class Pursue(str, Enum):
    """What a goal asks the motor system to do about it.

    Monty reads a goal's `sender_type` to decide how to act on it, not to attribute it,
    so a driver claims one of these senders to ask for its behaviour: a goal from a GSG
    is jumped to, a goal from an SM is turned towards and looked at.

    A `str` enum, so it crosses the wire as its plain value and is what Monty's motor
    system already expects for a goal's `sender_type`.
    """

    JUMP_TO = "GSG"
    LOOK_AT = "SM"


class GoalFrame(TypedDict, total=False):
    """A place a driver wants the agent to get to, or to look at.

    A goal is the least specific thing a driver can send, and the most work for the
    model: it names a place, and Monty's motor system works out the actions that get
    there. An action says do this; a heading says go that way; a goal says be there.

    Mirrors the parts of Monty's `Goal` a driver has any business setting. The rest --
    the morphological features, the tolerances -- is what a real goal state generator
    reasons about, and a frame does not pretend to.

    Attributes:
        location: the location to move to in global/body-centric coordinates, or
              `None` if the location is not specified as part of the goal.
              For example, this may be a point on an object's surface or a location
              nearby from which a sensor would have a good view of the target point.
        morphological_features: dictionary of morphological features or `None`.
              For example, it may include pose vectors, whether the pose is fully
              defined, etc.
        non_morphological_features: a dictionary containing non-morphological
              features at the target location or `None`.
        confidence: a float between 0 and 1 representing the confidence in the goal.
        use_state: a boolean indicating whether the goal should be used.
        sender_id: the ID of the sender of the goal (e.g., `"LM_0"`).
        sender_type: Use `GSG` to use JumpTo and `SM` to use LookAt. Note that
            morphological features are required for a jump. Defaults to `SM`.
        goal_tolerances: Dictionary of tolerances that GSGs use when determining
                whether the current state of the LM matches the driving goal
                or `None`. As such, a GSG can send a goal with more or less
                strict tolerances if certain elements of the message (e.g. the location
                of a mug vs its orientation) are more or less important.
        info: Optional metadata for logging purposes.

    """

    location: list[float]
    morphological_features: dict[str, Any] | None
    non_morphological_features: dict[str, Any] | None
    confidence: float
    use_state: bool
    sender_id: str
    sender_type: str
    goal_tolerances: dict[str, Any] | None
    info: dict[str, Any] | None
