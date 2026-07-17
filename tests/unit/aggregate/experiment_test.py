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

import numpy as np
import quaternion as qt
from tbp.monty.geometry import Rotation

from tbp.teleop.aggregate.experiment import (
    DefaultExperimentAggregator,
    NoOpExperimentAggregator,
)

SEED = 1170100843


class FakeExperiment:
    """Stands in for a `MontyExperiment`, which reports itself through `logger_args`."""

    def __init__(self, **args: Any) -> None:  # noqa: ANN401
        """Report `args` as this experiment's own account of where the run is."""
        self.logger_args = args


class NoOpExperimentAggregatorTest(unittest.TestCase):
    def test_aggregates_nothing(self) -> None:
        self.assertEqual(NoOpExperimentAggregator()(FakeExperiment(train_epochs=2)), {})


class DefaultExperimentAggregatorTest(unittest.TestCase):
    def test_reports_the_counters_the_experiment_keeps(self) -> None:
        """Taken whole, so a frame counts a run the way Monty's own logs do."""
        experiment = FakeExperiment(
            total_train_steps=120,
            train_episodes=4,
            train_epochs=1,
            total_eval_steps=21,
            eval_episodes=1,
            eval_epochs=0,
            episode_seed=SEED,
        )

        aggregated = DefaultExperimentAggregator()(experiment)

        self.assertEqual(aggregated["total_train_steps"], 120)
        self.assertEqual(aggregated["eval_episodes"], 1)
        self.assertEqual(aggregated["episode_seed"], SEED)

    def test_reduces_a_target_to_plain_data(self) -> None:
        """A target carries a rotation, which Monty hands over as a quaternion."""
        experiment = FakeExperiment(
            target={
                "object": "mug",
                "quat_rotation": qt.quaternion(1, 0, 0, 0),
                "position": np.array([0.0, 1.5, 0.0]),
            }
        )

        target = DefaultExperimentAggregator()(experiment)["primary_target"]

        self.assertEqual(target["object"], "mug")
        self.assertEqual(target["quat_rotation"], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(target["position"], [0.0, 1.5, 0.0])

    def test_reduces_a_rotation_the_same_way_however_it_arrived(self) -> None:
        """Monty carries a rotation as its own `Rotation` as well as a quaternion.

        `Rotation.as_quat` is a method, not a property, and gives back scalar-first
        `wxyz`, so both spellings land on the same four floats.
        """
        experiment = FakeExperiment(
            target={"rotation": Rotation.from_quat([1.0, 0.0, 0.0, 0.0])}
        )

        target = DefaultExperimentAggregator()(experiment)["primary_target"]

        self.assertEqual(target["rotation"], [1.0, 0.0, 0.0, 0.0])

    def test_an_experiment_that_never_offered_itself_reports_nothing(self) -> None:
        """Only `pretraining_experiments` passes its experiment to a step hook.

        A hook has no other way to reach one, so against every other experiment this
        gets `None`. A run is still watchable without its counters, so that is reported
        rather than raised.
        """
        self.assertEqual(DefaultExperimentAggregator()(None), {})

    def test_an_experiment_with_nothing_to_say_reports_nothing(self) -> None:
        self.assertEqual(DefaultExperimentAggregator()(FakeExperiment()), {})


if __name__ == "__main__":
    unittest.main()
