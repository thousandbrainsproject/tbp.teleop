# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Building a frame's `experiment` from the experiment running the step."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from tbp.teleop.aggregate.plain import plain

if TYPE_CHECKING:
    from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment

    from tbp.teleop.frames import ExperimentFrame


class ExperimentAggregator(Protocol):
    """Builds a frame's `experiment` from the experiment running the step."""

    def __call__(self, experiment: MontyExperiment) -> ExperimentFrame:
        """Aggregate the experiment's own account of where the run has got to.

        Args:
            experiment: The experiment running this step, or `None` when it did not
                offer itself.

        Returns:
            The frame's `experiment`.
        """
        ...


class NoOpExperimentAggregator(ExperimentAggregator):
    """An `ExperimentAggregator` that does nothing."""

    def __call__(self, experiment: MontyExperiment) -> ExperimentFrame:  # noqa: ARG002
        """Return an empty `ExperimentFrame`.

        Args:
            experiment: The experiment running this step.

        Returns:
            An empty `ExperimentFrame`.
        """
        return {}


class DefaultExperimentAggregator(ExperimentAggregator):
    """Reports what the experiment tells its own loggers.

    `MontyExperiment.logger_args` is the experiment's own answer to "where am I": its
    step, episode and epoch counters, the seed this episode started from, and the object
    the episode is about. Taking it whole means a frame counts a run the same way
    Monty's logs do, rather than offering a second opinion that could disagree.

    Note that only `pretraining_experiments` passes its experiment to a step hook today;
    `monty_experiment` and `object_recognition_experiments` do not, and a hook has no
    other way to reach one. Against those, this reports nothing rather than raising,
    since a run is still perfectly watchable without its counters.
    """

    def __call__(self, experiment: MontyExperiment) -> ExperimentFrame:
        """Aggregate the experiment's own account of where the run has got to.

        Args:
            experiment: The experiment running this step, or `None` when it did not
                offer itself.

        Returns:
            What `logger_args` reports, as plain data, or nothing when there is no
            experiment to ask.
        """
        params = getattr(experiment, "logger_args", {})
        if "target" in params:
            primary_target = params.pop("target")
            params["primary_target"] = primary_target
        return {key: plain(value) for key, value in params.items()}
