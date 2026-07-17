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

from tbp.teleop.aggregate.observations import (
    DefaultObservationAggregator,
    NoOpObservationAggregator,
    SmartObservationAggregator,
)

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


class ObservationsTest(unittest.TestCase):
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

    def sensor_frame(self, observations: dict) -> dict:
        return observations["agent_id_0"]["view_finder"]


class NoOpObservationAggregatorTest(ObservationsTest):
    def test_aggregates_nothing(self) -> None:
        """A section with no aggregator configured costs nothing to gather."""
        aggregated = NoOpObservationAggregator()(self.observations_of(self.sensor_of()))

        self.assertEqual(aggregated, {})


class DefaultObservationAggregatorTest(ObservationsTest):
    def test_rebuilds_observations_as_plain_dicts(self) -> None:
        """No Monty class survives into a frame, at any level of the nesting."""
        aggregated = DefaultObservationAggregator()(
            self.observations_of(self.sensor_of())
        )

        self.assertIs(type(aggregated), dict)
        self.assertIs(type(aggregated["agent_id_0"]), dict)
        self.assertIs(type(self.sensor_frame(aggregated)), dict)

    def test_shares_arrays_rather_than_copying_them(self) -> None:
        """Rebuilding the containers must not copy the arrays they hold."""
        image = self.rgba()

        aggregated = DefaultObservationAggregator()(
            self.observations_of(self.sensor_of(rgba=image))
        )

        self.assertIs(self.sensor_frame(aggregated)["rgba"], image)

    def test_keeps_every_modality(self) -> None:
        """Whatever a sensor reports is carried through untouched."""
        observations = self.observations_of(self.sensor_of(semantic_3d=SEMANTIC_3D))

        aggregated = DefaultObservationAggregator()(observations)

        self.assertEqual(
            set(self.sensor_frame(aggregated)), {"rgba", "depth", "semantic_3d"}
        )


class SmartObservationAggregatorTest(ObservationsTest):
    def test_narrows_to_rgba_and_depth(self) -> None:
        """A sensor reporting nothing derivable carries only the two base modalities."""
        aggregated = SmartObservationAggregator()(
            self.observations_of(self.sensor_of())
        )

        self.assertEqual(set(self.sensor_frame(aggregated)), {"rgba", "depth"})

    def test_unpacks_locations_and_semantic(self) -> None:
        """Habitat's flattened per-pixel rows reshape back to the sensor's frame."""
        observations = self.observations_of(self.sensor_of(semantic_3d=SEMANTIC_3D))

        sensor = self.sensor_frame(SmartObservationAggregator()(observations))

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

        sensor = self.sensor_frame(SmartObservationAggregator()(observations))

        self.assertEqual(sensor["locations_rel_world"].dtype, np.float32)
        self.assertEqual(sensor["semantic"].dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
