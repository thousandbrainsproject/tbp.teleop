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
import unittest
from pathlib import Path

import numpy as np

from tbp.teleop.frames import Frame
from tbp.teleop.recording import FrameWriter, TruncatedRecordingError, read_frames

FRAME_COUNT = 3
RGBA = np.arange(2 * 2 * 4, dtype=np.uint8).reshape(2, 2, 4)


class RecordingTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "run.frames"

    # Arrays differ per step, so a reader's ordering is checkable.
    def frame_at(self, step: int) -> Frame:
        return Frame(
            episode=0,
            step=step,
            observations={"agent_id_0": {"view_finder": {"rgba": RGBA + step}}},
        )

    # Writes and closes a recording of `steps` frames.
    def record(self, steps: int = FRAME_COUNT) -> FrameWriter:
        writer = FrameWriter(str(self.path))
        for step in range(steps):
            writer(self.frame_at(step))
        writer.close()
        return writer


class ReadWriteTest(RecordingTest):
    def test_round_trips_a_recording(self) -> None:
        """Every frame comes back, in order, with its arrays intact."""
        self.record()

        frames = list(read_frames(self.path))

        self.assertEqual([frame.step for frame in frames], list(range(FRAME_COUNT)))
        for step, frame in enumerate(frames):
            with self.subTest(step=step):
                np.testing.assert_array_equal(
                    frame.observations["agent_id_0"]["view_finder"]["rgba"], RGBA + step
                )

    def test_reads_an_empty_recording(self) -> None:
        """A run that recorded nothing is not an error."""
        FrameWriter(str(self.path)).close()

        self.assertEqual(list(read_frames(self.path)), [])

    def test_reads_frames_lazily(self) -> None:
        """A recording is walked a frame at a time, not loaded whole."""
        self.record()

        first = next(iter(read_frames(self.path)))

        self.assertEqual(first.step, 0)

    def test_closing_is_what_commits_the_recording(self) -> None:
        """The stream is buffered, so an unclosed recording is not readable."""
        writer = FrameWriter(str(self.path))
        writer(self.frame_at(0))

        self.assertEqual(self.path.stat().st_size, 0)

        writer.close()

        self.assertGreater(self.path.stat().st_size, 0)


class TruncationTest(RecordingTest):
    def test_reports_a_recording_that_ends_mid_frame(self) -> None:
        """A killed run leaves a partial frame, which must not pass as a clean end."""
        self.record()
        self.path.write_bytes(self.path.read_bytes()[:-4])

        frames = read_frames(self.path)

        # Every whole frame is still handed over before the partial one is reported.
        self.assertEqual([next(frames).step, next(frames).step], [0, 1])
        with self.assertRaises(TruncatedRecordingError):
            next(frames)

    def test_reports_a_recording_that_ends_in_a_length_prefix(self) -> None:
        """A run killed between the prefix and the payload is truncated too."""
        self.record(steps=1)
        self.path.write_bytes(self.path.read_bytes()[:2])

        with self.assertRaises(TruncatedRecordingError):
            list(read_frames(self.path))


class SavePathTest(RecordingTest):
    def test_a_relative_path_lands_under_local(self) -> None:
        """Hydra chdirs mid-run, so a relative recording must not follow it."""
        writer = FrameWriter("nested/run.frames")
        writer.close()
        self.addCleanup(writer.save_path.parent.rmdir)
        self.addCleanup(writer.save_path.unlink)

        self.assertTrue(writer.save_path.is_absolute())
        self.assertEqual(writer.save_path.parent.name, "nested")
        self.assertIn("local", writer.save_path.parts)

    def test_an_absolute_path_is_left_alone(self) -> None:
        """A caller that names a file gets that file."""
        writer = FrameWriter(str(self.path))
        writer.close()

        self.assertEqual(writer.save_path, self.path)


class PicklingTest(RecordingTest):
    def test_survives_being_pickled_mid_recording(self) -> None:
        """Monty pickles its config every epoch, and the step hook lives inside it.

        `post_epoch` -> `save_state_dir` -> `torch.save(self.config, ...)` runs whether
        or not anyone wants the state dicts, and no setting disables it. An open file
        cannot be pickled, so a writer holding one would take the run down with it.
        """
        writer = FrameWriter(str(self.path))
        writer(self.frame_at(0))

        restored = pickle.loads(pickle.dumps(writer))  # noqa: S301

        self.assertEqual(restored.save_path, self.path)
        self.assertEqual(restored.frames, 1)

        # The pickle must not disturb the recording the live writer still holds.
        writer(self.frame_at(1))
        writer.close()
        self.assertEqual([frame.step for frame in read_frames(self.path)], [0, 1])


class IsolationTest(unittest.TestCase):
    """Pins the property the whole design exists for.

    A recording outlives the process that made it, so a replay must run where Monty and
    habitat are not installed. Each of these is one careless import away from being
    quietly untrue.
    """

    def assert_imports_cleanly(self, program: str) -> None:
        subprocess.run([sys.executable, "-c", program], check=True)  # noqa: S603

    def test_reading_a_recording_needs_neither_monty_nor_matplotlib(self) -> None:
        self.assert_imports_cleanly(
            "import sys\n"
            "from tbp.teleop.recording import read_frames\n"
            "banned = ('tbp.monty', 'matplotlib')\n"
            "leaked = [m for m in sys.modules if m.startswith(banned)]\n"
            "assert not leaked, leaked\n"
        )

    def test_drawing_a_recording_needs_no_monty(self) -> None:
        """A consumer takes frames, and a frame does not care where it came from."""
        self.assert_imports_cleanly(
            "import sys\n"
            "from tbp.teleop.consumers import ObservationSaver\n"
            "from tbp.teleop.recording import read_frames\n"
            "leaked = [m for m in sys.modules if m.startswith('tbp.monty')]\n"
            "assert not leaked, leaked\n"
        )


if __name__ == "__main__":
    unittest.main()
