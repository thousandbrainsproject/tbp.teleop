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

from tbp.teleop.aggregate import (
    Aggregator,
    DefaultObservationAggregator,
    NoOpObservationAggregator,
    ObservationAggregator,
    SmartObservationAggregator,
)

if TYPE_CHECKING:
    from tbp.teleop.frame import Frame

STEP = 7

# Four pixels of a 2x2 frame: xyz in world coordinates, then the semantic id. The two
# on-object pixels sit on the frame's diagonal.
SEMANTIC_3D = np.array(
    [
        [1.0, 2.0, 3.0, 0.0],
        [4.0, 5.0, 6.0, 1.0],
        [7.0, 8.0, 9.0, 1.0],
        [10.0, 11.0, 12.0, 0.0],
    ]
)


class FakeObservations(dict):
    """Stands in for Monty's `Observations`, which is a `dict` subclass."""


class FakeAgentObservations(dict):
    """Stands in for Monty's `AgentObservations`, which is a `dict` subclass."""


class Collector:
    """A `FrameConsumer` that keeps every frame it is handed."""

    def __init__(self) -> None:
        """Start out having collected nothing."""
        self.frames: list[Frame] = []

    def __call__(self, frame: Frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


class SpyAggregator(Aggregator):
    """An `Aggregator` that counts how many times its run was initialized."""

    def __init__(self, consumer: Any, observations: Any) -> None:  # noqa: ANN401
        """Start out having initialized nothing."""
        super().__init__(consumer, observations)
        self.runs_initialized = 0

    def initialize_run(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().initialize_run(*args, **kwargs)
        self.runs_initialized += 1


class AggregationTest(unittest.TestCase):
    """Drives the hook without a model, an experiment, or a simulator.

    Nothing is read from `ctx`, `monty`, or `experiment` yet, so a step can be
    aggregated by passing `None` for them.
    """

    def rgba(self, value: int = 0) -> np.ndarray:
        return np.full((2, 2, 4), value, dtype=np.uint8)

    # Defaults to the rgba and depth that `SmartObservationAggregator` requires.
    def sensor_of(self, **modalities: Any) -> dict:  # noqa: ANN401
        return {"rgba": self.rgba(), "depth": np.zeros((2, 2)), **modalities}

    # Nests one sensor's observation the way Monty nests it, under one agent.
    def observations_of(self, sensor: dict) -> FakeObservations:
        return FakeObservations(
            {"agent_id_0": FakeAgentObservations({"view_finder": sensor})}
        )

    # Runs one step through the hook, returning the frame the consumer was handed.
    def aggregate(
        self,
        observations: FakeObservations,
        aggregator: ObservationAggregator,
        step: int = 0,
    ) -> Frame:
        collector = Collector()
        Aggregator(collector, aggregator)(None, None, [], step, observations, [])
        return collector.frames[0]

    def sensor_frame(self, frame: Frame) -> dict:
        return frame.observations["agent_id_0"]["view_finder"]


class AggregatorTest(AggregationTest):
    def test_passes_the_models_actions_through_untouched(self) -> None:
        """The hook observes; it must never steer the experiment."""
        actions = ["an action"]
        hook = Aggregator(Collector(), DefaultObservationAggregator())

        returned = hook(None, None, [], 0, FakeObservations(), actions)

        self.assertIs(returned, actions)

    def test_frame_carries_the_step(self) -> None:
        frame = self.aggregate(
            FakeObservations(), DefaultObservationAggregator(), step=STEP
        )

        self.assertEqual(frame.step, STEP)

    def test_counts_an_episode_when_the_step_goes_backwards(self) -> None:
        """A step count that restarts is the only signal of an episode boundary."""
        collector = Collector()
        hook = Aggregator(collector, DefaultObservationAggregator())

        for step in (0, 1, 2, 0, 1):
            hook(None, object(), [], step, FakeObservations(), [])

        self.assertEqual([frame.episode for frame in collector.frames], [0, 0, 0, 1, 1])

    def test_initializes_the_run_on_the_first_step(self) -> None:
        """Monty has no init call of its own, so the first step has to do it."""
        hook = SpyAggregator(Collector(), DefaultObservationAggregator())

        hook(None, object(), [], 0, FakeObservations(), [])

        self.assertEqual(hook.runs_initialized, 1)

    def test_initializes_the_run_only_once(self) -> None:
        """The run's schema is built once, not rebuilt every step."""
        hook = SpyAggregator(Collector(), DefaultObservationAggregator())

        for step in range(3):
            hook(None, object(), [], step, FakeObservations(), [])

        self.assertEqual(hook.runs_initialized, 1)


class DefaultObservationAggregatorTest(AggregationTest):
    def test_rebuilds_observations_as_plain_dicts(self) -> None:
        """No Monty class survives into a frame, at any level of the nesting."""
        frame = self.aggregate(
            self.observations_of(self.sensor_of()), DefaultObservationAggregator()
        )

        self.assertIs(type(frame.observations), dict)
        self.assertIs(type(frame.observations["agent_id_0"]), dict)
        self.assertIs(type(self.sensor_frame(frame)), dict)

    def test_shares_arrays_rather_than_copying_them(self) -> None:
        """Rebuilding the containers must not copy the arrays they hold."""
        image = self.rgba()

        frame = self.aggregate(
            self.observations_of(self.sensor_of(rgba=image)),
            DefaultObservationAggregator(),
        )

        self.assertIs(self.sensor_frame(frame)["rgba"], image)

    def test_keeps_every_modality(self) -> None:
        """Whatever a sensor reports is carried through untouched."""
        observations = self.observations_of(self.sensor_of(semantic_3d=SEMANTIC_3D))

        frame = self.aggregate(observations, DefaultObservationAggregator())

        self.assertEqual(
            set(self.sensor_frame(frame)), {"rgba", "depth", "semantic_3d"}
        )


class SmartObservationAggregatorTest(AggregationTest):
    def test_narrows_to_rgba_and_depth(self) -> None:
        """A sensor reporting nothing derivable carries only the two base modalities."""
        frame = self.aggregate(
            self.observations_of(self.sensor_of()), SmartObservationAggregator()
        )

        self.assertEqual(set(self.sensor_frame(frame)), {"rgba", "depth"})

    def test_unpacks_locations_and_semantic(self) -> None:
        """Habitat's flattened per-pixel rows reshape back to the sensor's frame."""
        observations = self.observations_of(self.sensor_of(semantic_3d=SEMANTIC_3D))

        sensor = self.sensor_frame(
            self.aggregate(observations, SmartObservationAggregator())
        )

        self.assertEqual(
            set(sensor), {"rgba", "depth", "locations_rel_world", "semantic"}
        )
        self.assertEqual(sensor["locations_rel_world"].shape, (2, 2, 3))
        self.assertEqual(sensor["semantic"].tolist(), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(
            sensor["locations_rel_world"][0][0], [1.0, 2.0, 3.0]
        )
        np.testing.assert_array_equal(
            sensor["locations_rel_world"][1][1], [10.0, 11.0, 12.0]
        )

    def test_narrows_the_derived_dtypes(self) -> None:
        """Habitat reports all of `semantic_3d` as float64; neither half needs 8 bytes.

        These two modalities are 40% of a frame's bytes, and a frame is what gets
        written to disk and sent over a wire, so the widths are part of the contract
        rather than an incidental detail of how habitat happened to report them.
        """
        observations = self.observations_of(self.sensor_of(semantic_3d=SEMANTIC_3D))

        sensor = self.sensor_frame(
            self.aggregate(observations, SmartObservationAggregator())
        )

        self.assertEqual(sensor["locations_rel_world"].dtype, np.float32)
        self.assertEqual(sensor["semantic"].dtype, np.int32)


class NoOpObservationAggregatorTest(AggregationTest):
    def test_aggregates_nothing(self) -> None:
        """A section with no aggregator configured costs nothing to gather."""
        frame = self.aggregate(
            self.observations_of(self.sensor_of()), NoOpObservationAggregator()
        )

        self.assertEqual(frame.observations, {})


if __name__ == "__main__":
    unittest.main()
