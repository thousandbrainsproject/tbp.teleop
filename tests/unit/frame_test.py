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

from tbp.teleop.frames import Broadcast, Frame


class Recorder:
    """A `FrameConsumer` that keeps what it was handed and whether it was closed."""

    def __init__(self, *, raise_on_close: bool = False) -> None:
        """Record nothing yet; optionally fail when closed."""
        self.frames: list[Frame] = []
        self.closed = False
        self._raise_on_close = raise_on_close

    def __call__(self, frame: Frame) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True
        if self._raise_on_close:
            error_message = "consumer refused to close"
            raise RuntimeError(error_message)


class BroadcastTest(unittest.TestCase):
    def test_hands_each_frame_to_every_consumer(self) -> None:
        """A frame has more than one home: recorded and published at once."""
        first, second = Recorder(), Recorder()
        frame = Frame(step=1)

        Broadcast([first, second])(frame)

        self.assertEqual(first.frames, [frame])
        self.assertEqual(second.frames, [frame])

    def test_shares_the_frame_rather_than_copying_it(self) -> None:
        """Every consumer sees the same frame, since none may write to it."""
        first, second = Recorder(), Recorder()
        frame = Frame(step=1)

        Broadcast([first, second])(frame)

        self.assertIs(first.frames[0], second.frames[0])

    def test_closes_every_consumer(self) -> None:
        first, second = Recorder(), Recorder()

        Broadcast([first, second]).close()

        self.assertTrue(first.closed)
        self.assertTrue(second.closed)

    def test_closes_the_rest_even_when_one_raises(self) -> None:
        """One leaky consumer must not leave the others open."""
        leaky, second = Recorder(raise_on_close=True), Recorder()

        with self.assertRaises(RuntimeError):
            Broadcast([leaky, second]).close()

        self.assertTrue(second.closed)


if __name__ == "__main__":
    unittest.main()
