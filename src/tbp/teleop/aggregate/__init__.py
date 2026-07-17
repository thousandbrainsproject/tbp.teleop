# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""A step hook that snapshots each step into a `Frame`.

The hook lives here and each section's aggregator in a module of its own, so a section
can grow without the others having to be read past. The hook knows only that it has one
aggregator per section of a frame; what any of them gathers is that module's business.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tbp.monty.frameworks.experiments.hooks import StepHook
from typing_extensions import Self

from tbp.teleop.aggregate.actions import ActionAggregator, NoOpActionAggregator
from tbp.teleop.aggregate.experiment import (
    ExperimentAggregator,
    NoOpExperimentAggregator,
)
from tbp.teleop.aggregate.observations import (
    NoOpObservationAggregator,
    ObservationAggregator,
)
from tbp.teleop.frames import Frame, NoOpFrameConsumer

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty, Observations

    from tbp.teleop.control import Teleoperator
    from tbp.teleop.frames import FrameConsumer

__all__ = ["Aggregator"]


class Aggregator(StepHook):
    """Step hook that snapshots each step into a `Frame` and renders nothing.

    This is intended to become the one place that reads Monty's objects on a consumer's
    behalf. Gathering those reads here, rather than leaving them spread across code that
    reaches into the live model, means a consumer only ever sees a `Frame` and could
    therefore run somewhere the model does not exist at all.

    The hook observes unless it is given a teleoperator, and then it also steers. The
    two are kept apart on purpose: consumers see every frame and answer nothing, so
    there may be any number of them and dropping a frame costs a repaint; a teleoperator
    is asked, exactly once per choice point, and the model waits for it. Without one the
    hook passes the model's actions through untouched and cannot change the course of an
    experiment it is watching.

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
    _actions: ActionAggregator
    _experiment: ExperimentAggregator
    _teleoperator: Teleoperator | None

    _run_initialized: bool
    _episode: int
    _last_step: int

    def __init__(
        self,
        consumer: FrameConsumer | None = None,
        observations: ObservationAggregator | None = None,
        actions: ActionAggregator | None = None,
        experiment: ExperimentAggregator | None = None,
        teleoperator: Teleoperator | None = None,
    ) -> None:
        """Initialize the hook.

        Args:
            consumer: Receives each step's snapshot.
            observations: Builds each frame's `observations`.
            actions: Builds each frame's `actions`.
            experiment: Builds each frame's `experiment`.
            teleoperator: Lets a driver choose the action at the steps that allow it.
                Without one the hook only watches.
        """
        self._consumer = consumer if consumer else NoOpFrameConsumer()
        self._observations = (
            observations if observations else NoOpObservationAggregator()
        )
        self._actions = actions if actions else NoOpActionAggregator()
        self._experiment = experiment if experiment else NoOpExperimentAggregator()
        self._teleoperator = teleoperator

        self._run_initialized = False
        self._episode = 0
        self._last_step = 0

    def initialize_run(
        self: Self,
        ctx: RuntimeContext,  # noqa: ARG002
        experiment: Any,  # noqa: ARG002, ANN401
        monty: Monty,  # noqa: ARG002
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
        supervised_lm_ids: list[str],  # noqa: ARG002
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
            The actions to run: the model's own, or a driver's when this hook has a
            teleoperator and the driver answered.
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
            actions=self._actions(actions),
            experiment=self._experiment(experiment),
        )
        self._consumer(frame)

        self._last_step = step

        # The frame is published before the teleoperator waits, so a driver has seen the
        # step it is deciding about. In step mode Monty then blocks here for a command;
        # in continuous mode it advances on its own, which is what makes publishing
        # reliable enough in step mode without a lossless channel: Monty makes no
        # frames while it waits, so there is nothing for the telemetry queue to drop.
        if self._teleoperator is None:
            return actions
        return self._teleoperator.advance(ctx, monty, observations, frame, actions)

    def close(self) -> None:
        """Close the step hook, and whatever it feeds.

        Monty closes the hook at the end of a run; a consumer or a driver that owns a
        file or a socket would otherwise never be told the run had ended.
        """
        self._consumer.close()
        if self._teleoperator is not None:
            self._teleoperator.close()
