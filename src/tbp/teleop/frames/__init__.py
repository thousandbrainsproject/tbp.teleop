# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The per-step snapshot of an experiment, and who receives one.

The schema, split one module per frame section. This package imports numpy only for
annotations and nothing of Monty, so a consumer can read a frame where Monty and habitat
are not installed -- what lets a recording outlive the run that made it, and lets
something out there drive an experiment it holds no part of.
"""

from __future__ import annotations

from tbp.teleop.frames.actions import ActionFrame, ActionsFrame
from tbp.teleop.frames.cmp import GoalFrame, Pursue
from tbp.teleop.frames.consumer import (
    Broadcast,
    FrameConsumer,
    NoOpFrameConsumer,
)
from tbp.teleop.frames.experiment import ExperimentFrame
from tbp.teleop.frames.frame import Frame
from tbp.teleop.frames.observations import (
    AgentID,
    AgentObservationsFrame,
    ObservationsFrame,
    SensorID,
    SensorObservationsFrame,
)

__all__ = [
    "ActionFrame",
    "ActionsFrame",
    "AgentID",
    "AgentObservationsFrame",
    "Broadcast",
    "ExperimentFrame",
    "Frame",
    "FrameConsumer",
    "GoalFrame",
    "NoOpFrameConsumer",
    "ObservationsFrame",
    "Pursue",
    "SensorID",
    "SensorObservationsFrame",
]
