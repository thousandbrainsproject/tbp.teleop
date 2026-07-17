# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""A step hook that snapshots each step into a `Frame`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from tbp.monty.frameworks.experiments.hooks import StepHook
from typing_extensions import Self

from tbp.teleop.frame import AgentID, Frame, NoOpFrameConsumer, SensorID

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        Monty,
        Observations,
        SensorObservation,
    )

    from tbp.teleop.frame import FrameConsumer, ObservationsFrame, SensorFrame


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

    def _sensor(self, sensor_observation: SensorObservation) -> SensorFrame:
        """Aggregate one sensor's observation.

        Args:
            sensor_observation: The sensor's observation this step.

        Returns:
            Limited and modified sensor data.
        """
        rgba = sensor_observation["rgba"]
        depth = sensor_observation["depth"]
        frame_shape = depth.shape

        sensor_frame: SensorFrame = {
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


class Aggregator(StepHook):
    """Step hook that snapshots each step into a `Frame` and renders nothing.

    This is intended to become the one place that reads Monty's objects on a consumer's
    behalf. Gathering those reads here, rather than leaving them spread across code that
    reaches into the live model, means a consumer only ever sees a `Frame` and could
    therefore run somewhere the model does not exist at all.

    The hook observes; it never steers. It passes the model's actions through untouched,
    so it cannot change the course of an experiment it is watching. Overriding actions
    is a separate concern that needs a return path from the consumer, and belongs in a
    separate hook.

    Each section of a `Frame` is gathered by its own sub-aggregator, so what a section
    contains, and how it is massaged out of the model, is a config choice rather than
    something baked into this hook. Adding a new section is deliberate: a field on
    `Frame` and a sub-aggregator to fill it, which is what keeps the frame a declared
    schema rather than whatever the model happened to be holding.

    Note that Monty's `step_hook` config is a single slot, so this cannot currently run
    alongside another hook without a chaining hook that does not exist yet.
    """

    _consumer: FrameConsumer
    _observations: ObservationAggregator

    _run_initialized: bool
    _episode: int
    _last_step: int

    def __init__(
        self,
        consumer: FrameConsumer | None = None,
        observations: ObservationAggregator | None = None,
    ) -> None:
        """Initialize the hook.

        Args:
            consumer: Receives each step's snapshot.
            observations: Builds each frame's `observations`.
        """
        self._consumer = consumer if consumer else NoOpFrameConsumer()
        self._observations = (
            observations if observations else NoOpObservationAggregator()
        )

        self._run_initialized = False
        self._episode = 0
        self._last_step = 0

    def initialize_run(
        self: Self,
        ctx: RuntimeContext,  # noqa: ARG002
        experiment: Any,  # noqa: ARG002, ANN401
        monty: Monty,
        observations: Observations,  # noqa: ARG002
        actions: list[Action],  # noqa: ARG002
    ) -> None:
        """Inspect the model and its data to construct an "input schema".

        Runs on the first step rather than before it, because Monty's `StepHook` has no
        initialization call of its own. Taking the whole step's context, rather than
        just the model, is what makes a schema knowable: which sensors report which
        modalities is a property of the environment's observations, not of `monty`.

        Args:
            ctx: The runtime context.
            experiment: The Monty experiment, if any, that is running this step.
            monty: The Monty model being observed.
            observations: The observations from the most recent step.
            actions: The actions returned by the model for the next step.
        """
        self._run_initialized = True

    # The argument list and the `Any` are Monty's `StepHook` signature, not ours.
    def __call__(  # noqa: PLR0913
        self: Self,
        ctx: RuntimeContext,
        monty: Monty,
        supervised_lm_ids: list[str],
        step: int,
        observations: Observations,
        actions: list[Action],
        experiment: Any | None = None,  # noqa: ANN401
    ) -> list[Action]:
        """Snapshot the step, hand it to the consumer, and leave the actions alone.

        Args:
            ctx: The runtime context.
            monty: The Monty model being observed.
            supervised_lm_ids: The list of supervised learning module IDs.
            step: The index of the current step within the episode.
            observations: The observations from the most recent step.
            actions: The actions returned by the model for the next step.
            experiment: The Monty experiment, if any, that is running this step.

        Returns:
            The actions the model output, unchanged.
        """
        if not self._run_initialized:
            self.initialize_run(
                ctx=ctx,
                experiment=experiment,
                monty=monty,
                observations=observations,
                actions=actions,
            )

        # Monty restarts the step count each episode, so a step that goes backwards is
        # the only signal that an episode boundary was crossed.
        if step < self._last_step:
            self._episode += 1

        frame = Frame(
            episode=self._episode,
            step=step,
            observations=self._observations(observations),
        )
        self._consumer(frame)

        self._last_step = step

        return actions

    def close(self) -> None:
        """Close the step hook, and the consumer it feeds.

        Monty closes the hook at the end of a run; a consumer that owns a file or a
        socket would otherwise never be told the run had ended.
        """
        self._consumer.close()
