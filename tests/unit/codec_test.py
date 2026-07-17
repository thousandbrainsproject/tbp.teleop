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

import msgpack
import numpy as np

from tbp.teleop.codec import VERSION, UnsupportedVersionError, decode, encode
from tbp.teleop.frame import Frame

EPISODE = 2
STEP = 7

# The modalities and dtypes a real frame carries, at the size habitat reports them.
RGBA = np.arange(64 * 64 * 4, dtype=np.uint8).reshape(64, 64, 4)
DEPTH = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
LOCATIONS = np.linspace(0.0, 1.0, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
SEMANTIC = np.tile(np.array([0, 1], dtype=np.int32), 64 * 32).reshape(64, 64)


class CodecTest(unittest.TestCase):
    # A frame carrying one agent's one sensor.
    def frame_of(self, **modalities: np.ndarray) -> Frame:
        return Frame(
            episode=EPISODE,
            step=STEP,
            observations={"agent_id_0": {"view_finder": dict(modalities)}},
        )

    def sensor_of(self, frame: Frame) -> dict:
        return frame.observations["agent_id_0"]["view_finder"]


class RoundTripTest(CodecTest):
    def test_round_trips_the_scalar_sections(self) -> None:
        frame = decode(encode(Frame(episode=EPISODE, step=STEP)))

        self.assertEqual(frame.episode, EPISODE)
        self.assertEqual(frame.step, STEP)
        self.assertIsNone(frame.observations)

    def test_round_trips_an_empty_frame(self) -> None:
        """Every section is optional, so a frame carrying nothing still encodes."""
        self.assertEqual(decode(encode(Frame())), Frame())

    def test_round_trips_every_modality_of_a_real_frame(self) -> None:
        """Values, dtypes, and shapes survive, for each dtype a frame really uses."""
        original = self.frame_of(
            rgba=RGBA, depth=DEPTH, locations=LOCATIONS, semantic=SEMANTIC
        )

        sensor = self.sensor_of(decode(encode(original)))

        for name, expected in (
            ("rgba", RGBA),
            ("depth", DEPTH),
            ("locations", LOCATIONS),
            ("semantic", SEMANTIC),
        ):
            with self.subTest(modality=name):
                np.testing.assert_array_equal(sensor[name], expected)
                self.assertEqual(sensor[name].dtype, expected.dtype)
                self.assertEqual(sensor[name].shape, expected.shape)

    def test_round_trips_the_nesting_and_the_episode(self) -> None:
        """The tree of agents and sensors comes back with plain string keys."""
        frame = decode(encode(self.frame_of(rgba=RGBA)))

        self.assertEqual(frame.episode, EPISODE)
        self.assertEqual(list(frame.observations), ["agent_id_0"])
        self.assertEqual(list(frame.observations["agent_id_0"]), ["view_finder"])

    def test_preserves_endianness(self) -> None:
        """A dtype's byte order travels with it, so a frame can cross machines."""
        big_endian = np.array([1.5, 2.5], dtype=">f8")

        decoded = self.sensor_of(decode(encode(self.frame_of(depth=big_endian))))

        np.testing.assert_array_equal(decoded["depth"], big_endian)
        self.assertEqual(decoded["depth"].dtype.byteorder, big_endian.dtype.byteorder)

    def test_round_trips_a_non_contiguous_view(self) -> None:
        """A view travels as its own contents, not as the buffer it looks into."""
        view = np.arange(12, dtype=np.int32).reshape(3, 4).T

        decoded = self.sensor_of(decode(encode(self.frame_of(semantic=view))))

        np.testing.assert_array_equal(decoded["semantic"], view)
        self.assertEqual(decoded["semantic"].shape, (4, 3))

    def test_round_trips_an_empty_array(self) -> None:
        """A sensor reporting a zero-length array is not a special case."""
        empty = np.empty((0, 3), dtype=np.float32)

        decoded = self.sensor_of(decode(encode(self.frame_of(locations=empty))))

        self.assertEqual(decoded["locations"].shape, (0, 3))
        self.assertEqual(decoded["locations"].dtype, np.float32)

    def test_numpy_scalars_become_python_scalars(self) -> None:
        """A section holding a numpy scalar encodes rather than raising."""
        frame = decode(encode(Frame(episode=EPISODE, step=np.int64(STEP))))

        self.assertEqual(frame.step, STEP)
        self.assertIsInstance(frame.step, int)


class EncodingCostTest(CodecTest):
    def test_encoding_costs_what_the_arrays_weigh(self) -> None:
        """Arrays travel as bytes, not text; `tolist()` would roughly triple this."""
        frame = self.frame_of(
            rgba=RGBA, depth=DEPTH, locations=LOCATIONS, semantic=SEMANTIC
        )
        raw_bytes = RGBA.nbytes + DEPTH.nbytes + LOCATIONS.nbytes + SEMANTIC.nbytes

        encoded = len(encode(frame))

        self.assertLess(encoded, raw_bytes * 1.01)


class VersionTest(CodecTest):
    def test_refuses_a_frame_from_a_different_codec_version(self) -> None:
        """A reader that cannot know the format must not guess at it."""
        data = msgpack.packb({"version": VERSION + 1, "frame": {}}, use_bin_type=True)

        with self.assertRaises(UnsupportedVersionError):
            decode(data)


class UnsupportedTypeTest(CodecTest):
    def test_refuses_to_encode_a_type_a_frame_cannot_carry(self) -> None:
        """A frame holds plain data; anything else is a mistake worth hearing about."""
        frame = Frame(observations={"agent_id_0": {"view_finder": {"rgba": object()}}})

        with self.assertRaises(TypeError):
            encode(frame)


if __name__ == "__main__":
    unittest.main()
