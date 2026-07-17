# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The per-step snapshot of an experiment: a `Frame`.

The whole schema is this package. It imports numpy only for annotations and nothing of
Monty, so a consumer can read a frame in a process where Monty and habitat are not
installed, which is what lets a recording outlive the run that made it, and what would
let something out there drive an experiment it holds no part of.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tbp.teleop.frames.actions import ActionsFrame
    from tbp.teleop.frames.experiment import ExperimentFrame
    from tbp.teleop.frames.observations import ObservationsFrame


@dataclass(frozen=True)
class Frame:
    """One step of an experiment, in a form that could be put on a wire.

    A frame holds only plain containers, numpy arrays, and primitives. It is meant to
    grow a field at a time as consumers need more of the model's state.

    Every section is optional, and defaults to `None` when no aggregator was configured
    to fill it. A consumer that needs nothing from a section does not pay to gather it,
    which matters most for the sections that are expensive to snapshot every step.

    Frozen to mark it as a snapshot, but the arrays it holds are shared with the
    observations it was built from rather than copied, so treat its contents as
    read-only.

    Attributes:
        episode: The index of the current episode within the run.
        step: The index of the current step within the episode.
        observations: The observations from the most recent step, keyed by agent id and
            then by sensor id.
        actions: What the model will do next, and what a driver could do instead.
        experiment: Where the run has got to, as the experiment counts it.
    """

    episode: int | None = None
    step: int | None = None
    observations: ObservationsFrame | None = None
    actions: ActionsFrame | None = None
    experiment: ExperimentFrame | None = None
