# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Who receives each step's frame."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tbp.teleop.frames.frame import Frame


class FrameConsumer(Protocol):
    """Receives each step's snapshot.

    The seam a consumer eventually plugs into: a plotter, a per-step logger, or the
    encoder that puts a frame on a wire for another process to render.
    """

    def __call__(self, frame: Frame) -> None:
        """Handle one step's snapshot.

        Args:
            frame: The step's snapshot.
        """
        ...

    def close(self) -> None:
        """Release whatever the consumer holds open.

        Called once, when the run ends. A consumer that owns a file, a socket, or a
        process needs this: without it a recording loses its buffered tail and a
        connection is never shut down.
        """
        ...


class NoOpFrameConsumer(FrameConsumer):
    """A `FrameConsumer` that does nothing.

    Lives here beside the protocol rather than with the real consumers so that a default
    one costs no dependency: an aggregator that was given no consumer should not drag in
    a plotting library to say so.
    """

    def __call__(self, frame: Frame) -> None:
        """Do nothing with the frame.

        Args:
            frame: The step snapshot.
        """

    def close(self) -> None:
        """Hold nothing, release nothing."""


class Broadcast(FrameConsumer):
    """A `FrameConsumer` that hands each frame to several others.

    A frame has more than one home: it is recorded for keeps and published for looking
    at, at the same time. This is how a run does both -- one hook, one aggregator, and a
    consumer that fans out to a writer, a publisher, and whatever else wants to watch.

    A frame is shared, not copied, so every consumer sees the same arrays; none may
    write to them, which is already a frame's contract.
    """

    def __init__(self, consumers: Iterable[FrameConsumer]) -> None:
        """Fan out to `consumers`, in the order given.

        Args:
            consumers: The consumers to hand each frame to.
        """
        self._consumers = list(consumers)

    def __call__(self, frame: Frame) -> None:
        """Hand the frame to every consumer.

        Args:
            frame: The step snapshot.
        """
        for consumer in self._consumers:
            consumer(frame)

    def close(self) -> None:
        """Close every consumer, even if one of them raises on the way out.

        Re-raises the first error a consumer raised while closing, once the rest have
        each been given their chance to close too, so one leaky consumer does not leave
        the others open.
        """
        errors = []
        for consumer in self._consumers:
            try:
                consumer.close()
            except Exception as error:  # noqa: BLE001, PERF203
                errors.append(error)
        if errors:
            raise errors[0]
