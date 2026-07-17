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

from tbp.teleop.frame import Frame
from tbp.teleop.wire import FramePublisher, subscribe

STEP = 7
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
                publisher(self.frame_at(STEP))
                time.sleep(0.02)

        thread = threading.Thread(target=publish_until_received, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(stop.set)

        frame = next(subscribe(self.endpoint, timeout=CONNECT_TIMEOUT))

        self.assertEqual(frame.step, STEP)
        self.assertEqual(frame.episode, 0)
        np.testing.assert_array_equal(
            frame.observations["agent_id_0"]["view_finder"]["rgba"], RGBA + STEP
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


class IsolationTest(unittest.TestCase):
    def test_subscribing_needs_no_monty(self) -> None:
        """The far end of the wire is an interpreter Monty could not run on."""
        program = (
            "import sys\n"
            "from tbp.teleop.wire import subscribe\n"
            "banned = ('tbp.monty', 'torch')\n"
            "leaked = [m for m in sys.modules if m.startswith(banned)]\n"
            "assert not leaked, leaked\n"
        )

        subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603


if __name__ == "__main__":
    unittest.main()
