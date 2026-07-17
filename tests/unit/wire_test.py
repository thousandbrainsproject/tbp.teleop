# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tbp.teleop.commands import (
    Command,
    CommandOperation,
    CommandResult,
    RunMode,
)
from tbp.teleop.frames import Frame
from tbp.teleop.wire import (
    CommandClient,
    CommandServer,
    FramePublisher,
    subscribe,
)

STEP_INDEX = 7
RGBA = np.arange(2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 4)

# Generous: it only bounds how long a wedged test may hang, not how long one takes.
CONNECT_TIMEOUT = 10.0


class WireTest(unittest.TestCase):
    def setUp(self) -> None:
        # A socket per test, so tests that run at the same time cannot collide on an
        # address the way a fixed port would.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.endpoint = f"ipc://{Path(tmp.name) / 'frames.sock'}"

    def frame_at(self, step: int) -> Frame:
        return Frame(
            episode=0,
            step=step,
            observations={"agent_id_0": {"view_finder": {"rgba": RGBA + step}}},
        )


class SubscriptionTest(WireTest):
    def test_publishes_frames_a_subscriber_receives(self) -> None:
        """A frame crosses the socket whole, arrays and all."""
        publisher = FramePublisher(self.endpoint)
        self.addCleanup(publisher.close)

        # A publisher drops everything until a subscriber has finished connecting, so
        # the frame is republished until one lands rather than sent once and hoped for.
        stop = threading.Event()

        def publish_until_received() -> None:
            while not stop.is_set():
                publisher(self.frame_at(STEP_INDEX))
                time.sleep(0.02)

        thread = threading.Thread(target=publish_until_received, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(stop.set)

        frame = next(subscribe(self.endpoint, timeout=CONNECT_TIMEOUT))

        self.assertEqual(frame.step, STEP_INDEX)
        self.assertEqual(frame.episode, 0)
        np.testing.assert_array_equal(
            frame.observations["agent_id_0"]["view_finder"]["rgba"], RGBA + STEP_INDEX
        )

    def test_a_subscription_gives_up_after_silence(self) -> None:
        """A publisher says nothing when a run ends, so quiet is all there is."""
        frames = list(subscribe(self.endpoint, timeout=0.1))

        self.assertEqual(frames, [])


class PublisherTest(WireTest):
    def test_publishing_to_nobody_does_not_block(self) -> None:
        """An experiment must not stall because nothing is watching it."""
        publisher = FramePublisher(self.endpoint)
        self.addCleanup(publisher.close)

        for step in range(50):
            publisher(self.frame_at(step))

        self.assertEqual(publisher.frames, 50)

    def test_survives_being_pickled(self) -> None:
        """Monty pickles its config every epoch, and the step hook lives inside it.

        `post_epoch` -> `save_state_dir` -> `torch.save(self.config, ...)` runs whether
        or not anyone wants the state dicts, and no setting disables it. A socket cannot
        be pickled, so a publisher holding one would take the run down with it.
        """
        publisher = FramePublisher(self.endpoint)
        self.addCleanup(publisher.close)
        publisher(self.frame_at(0))

        restored = pickle.loads(pickle.dumps(publisher))  # noqa: S301

        self.assertEqual(restored.endpoint, self.endpoint)
        self.assertEqual(restored.frames, 1)

        # The pickle must not disturb the socket the live publisher still holds.
        publisher(self.frame_at(1))
        self.assertEqual(publisher.frames, 2)


class CommandChannelTest(WireTest):
    def serve_one(self, result: CommandResult) -> dict:
        # Runs a server in a thread that answers one command with `result`, and
        # returns a dict the thread fills with the command it received, so a test can
        # check both ends of the exchange.
        server = CommandServer(self.endpoint)
        self.addCleanup(server.close)
        received: dict = {}

        def serve() -> None:
            command = server.receive(timeout=CONNECT_TIMEOUT)
            received["command"] = command
            if command is not None:
                server.acknowledge(result)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        return received

    def test_a_command_is_received_and_acknowledged(self) -> None:
        """The driver's command reaches the experiment, and its answer comes back."""
        received = self.serve_one(
            CommandResult(run_mode=RunMode.STEP, episode=0, step=3)
        )
        client = CommandClient(self.endpoint, timeout=CONNECT_TIMEOUT)
        self.addCleanup(client.close)

        result = client.send(
            Command(
                operation=CommandOperation.SET_RUN_MODE,
                run_mode=RunMode.STEP,
                interval=0.5,
            )
        )

        self.assertEqual(
            result, CommandResult(run_mode=RunMode.STEP, episode=0, step=3)
        )
        self.assertEqual(received["command"].operation, CommandOperation.SET_RUN_MODE)
        self.assertEqual(received["command"].interval, 0.5)

    def test_a_server_polls_without_blocking(self) -> None:
        """With no command waiting, a zero timeout returns at once, not never."""
        server = CommandServer(self.endpoint)
        self.addCleanup(server.close)

        self.assertIsNone(server.receive(timeout=0))

    def test_a_client_gives_up_when_the_experiment_is_gone(self) -> None:
        """A command sent to nobody returns `None`, rather than hanging the driver."""
        client = CommandClient(self.endpoint, timeout=0.1)
        self.addCleanup(client.close)

        self.assertIsNone(client.send(Command(operation=CommandOperation.STEP)))

    def test_a_server_survives_being_pickled(self) -> None:
        """Monty pickles its config every epoch, and a socket cannot be pickled."""
        server = CommandServer(self.endpoint)
        self.addCleanup(server.close)

        restored = pickle.loads(pickle.dumps(server))  # noqa: S301

        self.assertEqual(restored.endpoint, self.endpoint)


class IsolationTest(unittest.TestCase):
    def assert_imports_cleanly(self, module: str, name: str) -> None:
        program = (
            "import sys\n"
            f"from {module} import {name}\n"
            "banned = ('tbp.monty', 'torch')\n"
            "leaked = [m for m in sys.modules if m.startswith(banned)]\n"
            "assert not leaked, leaked\n"
        )
        subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603

    def test_subscribing_needs_no_monty(self) -> None:
        """The far end of the wire is an interpreter Monty could not run on."""
        self.assert_imports_cleanly("tbp.teleop.wire", "subscribe")

    def test_commanding_needs_no_monty(self) -> None:
        """A driver commands the experiment without holding any part of it."""
        self.assert_imports_cleanly("tbp.teleop.wire", "CommandClient")


if __name__ == "__main__":
    unittest.main()
