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

from tbp.teleop.codec import (
    decode,
    decode_command,
    decode_result,
    encode,
    encode_command,
    encode_result,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tbp.teleop.commands import Command, CommandResult
    from tbp.teleop.frames import Frame

# The experiment binds and the far ends connect: it is the long-lived side, and there
# may be no viewer, one, or several. Telemetry and control each get their own endpoint,
# because the two channels want opposite things of ZeroMQ -- telemetry drops and fans
# out, control is reliable and answered.
DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"
DEFAULT_CONTROL_ENDPOINT = "tcp://127.0.0.1:5556"

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


class CommandServer:
    """The experiment's end of the control channel: receive a command, answer it.

    Binds a reply socket the driver connects to, and speaks only when spoken to, which
    is all the control channel ever needs: the driver always asks, and the experiment
    answers where it stands. A command received must be answered with `acknowledge`
    before the next is received; a receive that times out is answered by nobody, so no
    reply is owed.

    How long `receive` waits is the caller's to decide, and it is the whole of what
    separates the modes: block for a command in step mode, poll for one in continuous.
    """

    def __init__(self, endpoint: str = DEFAULT_CONTROL_ENDPOINT) -> None:
        """Bind the control socket.

        Args:
            endpoint: The ZeroMQ endpoint to receive commands on.
        """
        self.endpoint = endpoint
        self.received = 0
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)

    def receive(self, timeout: float | None = None) -> Command | None:
        """Wait for a command, up to `timeout`.

        Args:
            timeout: Seconds to wait, `0` to poll without waiting, or `None` to wait
                until a command arrives.

        Returns:
            The command, or `None` when the wait ran out first.
        """
        # Set per-receive, since the wait is what the mode varies: -1 blocks, 0 polls.
        self._socket.setsockopt(
            zmq.RCVTIMEO, -1 if timeout is None else int(timeout * 1000)
        )
        try:
            data = self._socket.recv()
        except zmq.Again:
            return None
        self.received += 1
        return decode_command(data)

    def acknowledge(self, result: CommandResult) -> None:
        """Answer the command just received.

        Args:
            result: Where the run now stands.
        """
        self._socket.send(encode_result(result))

    def close(self) -> None:
        """Close the socket."""
        self._socket.close()
        self._context.term()

    def __getstate__(self) -> dict:
        """Leave the socket out of the pickle Monty takes of its config.

        Monty saves its state at the end of every epoch, the hook lives in that config,
        and no setting turns it off. A socket cannot be pickled.

        Returns:
            The server's state, without its socket.
        """
        return {**self.__dict__, "_context": None, "_socket": None}


class CommandClient:
    """A driver's end of the control channel: send a command, get an acknowledgement.

    Connects to the experiment's `CommandServer` and speaks first, always. Each command
    is answered before the next may be sent, so the driver knows every command landed
    and where the run stands after it.

    The wait for an answer is finite: an experiment that has ended answers nothing, and
    a driver must not hang for ever on a run that is gone. A timed-out send returns
    `None`, and the socket is left able to send again rather than wedged.
    """

    def __init__(
        self, endpoint: str = DEFAULT_CONTROL_ENDPOINT, timeout: float | None = None
    ) -> None:
        """Connect the control socket.

        Args:
            endpoint: The ZeroMQ endpoint to send commands to.
            timeout: Seconds to wait for an acknowledgement, or `None` to wait for ever.
        """
        self.endpoint = endpoint
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        # A plain REQ socket is strictly lockstep, so a send that times out leaves it
        # unable to send again; relaxing that lets the next command go, and correlating
        # it lets a late answer to an abandoned command be spotted as stale rather than
        # taken for the answer to this one.
        self._socket.setsockopt(zmq.REQ_RELAXED, 1)
        self._socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        if timeout is not None:
            self._socket.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        self._socket.connect(endpoint)

    def send(self, command: Command) -> CommandResult | None:
        """Send a command and wait for its acknowledgement.

        Args:
            command: The command to send.

        Returns:
            Where the run stands after the command, or `None` when the experiment did
            not answer in time.
        """
        self._socket.send(encode_command(command))
        try:
            reply = self._socket.recv()
        except zmq.Again:
            return None
        return decode_result(reply)

    def close(self) -> None:
        """Close the socket."""
        self._socket.close()
        self._context.term()


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
