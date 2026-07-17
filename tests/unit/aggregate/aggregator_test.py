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

from tbp.teleop.aggregate import Aggregator
from tbp.teleop.aggregate.observations import DefaultObservationAggregator

if TYPE_CHECKING:
    from tbp.teleop.frames import Frame

STEP = 7


class FakeObservations(dict):
    """Stands in for Monty's `Observations`, which is a `dict` subclass."""


class Collector:
    """A `FrameConsumer` that keeps every frame it is handed."""

    def __init__(self) -> None:
        """Start out having collected nothing."""
        self.frames: list[Frame] = []
        self.closed = False

    def __call__(self, frame: Frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class SpyAggregator(Aggregator):
    """An `Aggregator` that counts how many times its run was initialized."""

    def __init__(self, consumer: Any, observations: Any = None) -> None:  # noqa: ANN401
        """Start out having initialized nothing."""
        super().__init__(consumer, observations)
        self.runs_initialized = 0

    def initialize_run(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().initialize_run(*args, **kwargs)
        self.runs_initialized += 1


class AggregatorTest(unittest.TestCase):
    """Drives the hook without a model, an experiment, or a simulator.

    The hook itself reads nothing from `ctx`, `monty`, or `experiment`; it hands them to
    the aggregators, and a `NoOp` one wants nothing. So a step can be put through it by
    passing `None` for all three.
    """

    def step(self, hook: Aggregator, step: int = 0) -> None:
        hook(None, None, [], step, FakeObservations(), [])

    def test_passes_the_models_actions_through_untouched(self) -> None:
        """The hook observes; it must never steer the experiment."""
        actions = ["an action"]
        hook = Aggregator(Collector())

        returned = hook(None, None, [], 0, FakeObservations(), actions)

        self.assertIs(returned, actions)

    def test_frame_carries_the_step(self) -> None:
        collector = Collector()

        self.step(Aggregator(collector), step=STEP)

        self.assertEqual(collector.frames[0].step, STEP)

    def test_an_unconfigured_section_is_gathered_by_nobody(self) -> None:
        """A consumer that needs nothing from a section does not pay to gather it."""
        collector = Collector()

        self.step(Aggregator(collector))

        frame = collector.frames[0]
        self.assertEqual(frame.observations, {})
        self.assertEqual(frame.actions, {})
        self.assertEqual(frame.experiment, {})

    def test_a_configured_section_is_filled_by_its_aggregator(self) -> None:
        collector = Collector()
        hook = Aggregator(collector, observations=DefaultObservationAggregator())

        hook(None, None, [], 0, FakeObservations({"agent_id_0": {}}), [])

        self.assertEqual(collector.frames[0].observations, {"agent_id_0": {}})

    def test_counts_an_episode_when_the_step_goes_backwards(self) -> None:
        """A step count that restarts is the only signal of an episode boundary."""
        collector = Collector()
        hook = Aggregator(collector)

        for step in (0, 1, 2, 0, 1):
            self.step(hook, step)

        self.assertEqual([frame.episode for frame in collector.frames], [0, 0, 0, 1, 1])

    def test_initializes_the_run_on_the_first_step(self) -> None:
        """Monty has no init call of its own, so the first step has to do it."""
        hook = SpyAggregator(Collector())

        self.step(hook)

        self.assertEqual(hook.runs_initialized, 1)

    def test_initializes_the_run_only_once(self) -> None:
        """The run's schema is built once, not rebuilt every step."""
        hook = SpyAggregator(Collector())

        for step in range(3):
            self.step(hook, step)

        self.assertEqual(hook.runs_initialized, 1)

    def test_closing_the_hook_closes_the_consumer(self) -> None:
        """Monty closes the hook; a consumer holding a file or socket needs telling."""
        collector = Collector()

        Aggregator(collector).close()

        self.assertTrue(collector.closed)


if __name__ == "__main__":
    unittest.main()
