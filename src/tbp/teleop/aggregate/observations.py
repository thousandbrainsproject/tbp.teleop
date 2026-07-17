# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Building a frame's `observations` from Monty's."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

from tbp.teleop.frames import AgentID, SensorID

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        Observations,
        SensorObservation,
    )

    from tbp.teleop.frames import ObservationsFrame, SensorObservationsFrame


class ObservationAggregator(Protocol):
    """Builds a frame's `observations` from Monty's.

    One sub-aggregator per section of a `Frame`, so how a section is gathered can be
    swapped in config without the hook, the frame, or any consumer knowing about it.
    """

    def __call__(self, observations: Observations) -> ObservationsFrame:
        """Aggregate one step's observations.

        Args:
            observations: The observations from the most recent step.

        Returns:
            The frame's `observations`.
        """
        ...


class NoOpObservationAggregator(ObservationAggregator):
    """An `ObservationAggregator` that does nothing."""

    def __call__(self, observations: Observations) -> ObservationsFrame:  # noqa: ARG002
        """Return an empty `ObservationsFrame`.

        Args:
            observations: The observations from the most recent step.

        Returns:
            An empty `ObservationsFrame`.
        """
        return {}


class DefaultObservationAggregator(ObservationAggregator):
    """Keeps every modality each sensor reports, rebuilt as plain dicts.

    Monty's `Observations` and `AgentObservations` are `Dict` subclasses, so the mapping
    is rebuilt rather than passed through: what comes out holds no Monty class and can
    be read where Monty is not installed. Monty's `AgentID` and `SensorID` are
    `NewType`s over `str` and the frame's are its own, so the keys are restated as the
    frame's rather than carried across.

    The arrays themselves are shared rather than copied, since Monty builds fresh
    observations for each step.
    """

    def __call__(self, observations: Observations) -> ObservationsFrame:
        """Aggregate one step's observations.

        Args:
            observations: The observations from the most recent step.

        Returns:
            Every sensor's every modality, as plain nested dicts of numpy arrays.
        """
        # `SensorFrame` declares every key `SensorObservation` does, so a pass-through
        # satisfies it by construction; there is nothing to validate at runtime, where
        # a `TypedDict` is a plain dict anyway.
        return {
            AgentID(str(agent_id)): {
                SensorID(str(sensor_id)): dict(sensor_observation)
                for sensor_id, sensor_observation in agent_observations.items()
            }
            for agent_id, agent_observations in observations.items()
        }


class SmartObservationAggregator(ObservationAggregator):
    """Massages data into a more consumer-friendly shape than Monty provides.

    Provides a limited set of `SensorFrame` modalities:
      - rgba
      - depth
      - locations_rel_world (if possible)
      - semantic (if possible)

    Where `DefaultObservationAggregator` hands habitat's values on as they came, this
    one narrows them to what a consumer can use and to the widths a frame should pay
    for, so it is the aggregator whose output the `SensorFrame` dtypes describe.
    """

    def __call__(self, observations: Observations) -> ObservationsFrame:
        """Aggregate one step's observations.

        Args:
            observations: The observations from the most recent step.

        Returns:
            Each sensor's kept modalities, plus anything derived from `semantic_3d`.
        """
        return {
            AgentID(str(agent_id)): {
                SensorID(str(sensor_id)): self._sensor(sensor_observation)
                for sensor_id, sensor_observation in agent_observations.items()
            }
            for agent_id, agent_observations in observations.items()
        }

    def _sensor(self, sensor_observation: SensorObservation) -> SensorObservationsFrame:
        """Aggregate one sensor's observation.

        Args:
            sensor_observation: The sensor's observation this step.

        Returns:
            Limited and modified sensor data.
        """
        rgba = sensor_observation["rgba"]
        depth = sensor_observation["depth"]
        frame_shape = depth.shape

        sensor_frame: SensorObservationsFrame = {
            "rgba": np.asarray(rgba),
            "depth": np.asarray(depth),
        }

        if "semantic_3d" in sensor_observation:
            semantic_3d = sensor_observation["semantic_3d"]
            # Habitat reports all four columns as float64, which is more than either
            # half of them means. A location is a position in metres, and across a
            # scene's few-metre extent float32 still resolves to a fraction of a
            # micron: far below anything a sensor could tell apart. A semantic id is a
            # small integer. Together these two are 40% of a frame's bytes, and a frame
            # is what goes to a disk or down a wire.
            locations_rel_world = semantic_3d[:, 0:3].reshape((*frame_shape, 3))
            sensor_frame["locations_rel_world"] = locations_rel_world.astype(np.float32)
            sensor_frame["semantic"] = (
                semantic_3d[:, 3].reshape(frame_shape).astype(np.int32)
            )

        return sensor_frame
