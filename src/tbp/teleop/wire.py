# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Publishing frames to a socket, and subscribing to them.

Like `recording`, this reaches only for `codec` and `frame`, so a subscriber needs
numpy, msgpack, and pyzmq, but neither Monty nor habitat. That is the whole point: the
far end is free to be an interpreter Monty could not run on.

A recording and a subscription are the same frames with different guarantees. A
recording loses nothing and is read afterwards; a subscription is live and lossy. An
experiment should never stall because nobody is watching it, so publishing drops what a
viewer cannot keep up with rather than applying back pressure to the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import zmq

from tbp.teleop.codec import decode, encode

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tbp.teleop.frame import Frame

# The publisher binds and subscribers connect: the experiment is the long-lived end,
# and there may be no viewer, one, or several.
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"

# How many frames may queue for a slow subscriber before the oldest are dropped. A
# frame is a couple of hundred kilobytes, so ZeroMQ's default of a thousand would let
# hundreds of megabytes pile up behind a viewer that stalled. A live view wants the
# newest frames, not a backlog.
SEND_QUEUE = 10


class FramePublisher:
    """A `FrameConsumer` that publishes every frame it is handed.

    Publishes rather than sends: an experiment must never wait on whoever is watching
    it. ZeroMQ drops frames a subscriber cannot keep up with, and drops every frame when
    nobody is listening at all, which is the right trade for a live view and the reason
    `FrameWriter` exists for when nothing may be lost.

    A subscriber that connects late misses whatever came before it, since a publisher
    has no memory of what it sent. Habitat takes long enough to start that bringing a
    viewer up first is usually enough; a recording is the answer when it is not.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT) -> None:
        """Bind the publishing socket.

        Args:
            endpoint: The ZeroMQ endpoint to publish on.
        """
        self.endpoint = endpoint
        self.frames = 0
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.set_hwm(SEND_QUEUE)
        # Do not let closing the socket wait on frames nobody has collected: with no
        # subscriber there is nothing to wait for, and an experiment should not hang at
        # the end of a run because of its telemetry.
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)

    def __call__(self, frame: Frame) -> None:
        """Publish one frame.

        Args:
            frame: The step snapshot.
        """
        self._socket.send(encode(frame))
        self.frames += 1

    def close(self) -> None:
        """Close the socket."""
        self._socket.close()
        self._context.term()

    def __getstate__(self) -> dict:
        """Leave the socket out of the pickle Monty takes of its config.

        Monty saves its state at the end of every epoch (`post_epoch` ->
        `save_state_dir` -> `torch.save(self.config, ...)`), the hook lives in that
        config, and no setting turns it off. A socket cannot be pickled, so a publisher
        that kept one would take the run down with it.

        Returns:
            The publisher's state, without its socket.
        """
        return {**self.__dict__, "_context": None, "_socket": None}


def subscribe(
    endpoint: str = DEFAULT_ENDPOINT, timeout: float | None = None
) -> Iterator[Frame]:
    """Yield frames as they are published.

    Args:
        endpoint: The ZeroMQ endpoint to subscribe to.
        timeout: Seconds of silence to wait before giving up, or `None` to wait for
            ever. A publisher says nothing when a run ends, so a caller that must
            return needs to decide how long quiet means over.

    Yields:
        Each frame, as it arrives.
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.subscribe(b"")
    if timeout is not None:
        socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
    socket.connect(endpoint)
    try:
        while True:
            try:
                payload = socket.recv()
            except zmq.Again:
                return
            yield decode(payload)
    finally:
        socket.close()
        context.term()
