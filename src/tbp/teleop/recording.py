# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Recording frames to a file, and reading them back.

This module reaches only for `codec` and `frame`, so reading a recording needs numpy
and msgpack and nothing else. That is the point of it: a recorded run can be replayed
in a process where Monty and habitat are not installed at all, which is not something
Monty's own episode-granularity logs can offer.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

from tbp.teleop.codec import decode, encode

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import BinaryIO

    from tbp.teleop.frame import Frame

# The repo's gitignored scratch directory. Resolved from this file rather than the
# working directory because Hydra chdirs into its own output directory for a run.
LOCAL_DIR = Path(__file__).resolve().parents[3] / "local"

# A frame is written as its byte length followed by its payload. A writer can stream a
# recording without knowing how many frames it will hold, and a reader can walk one
# frame at a time without loading the whole file. The same framing is what a socket
# would need, so the wire and the file differ only in what the bytes are handed to.
_LENGTH = struct.Struct(">I")


class TruncatedRecordingError(ValueError):
    """Raised when a recording ends part way through a frame."""


def write_frame(stream: BinaryIO, frame: Frame) -> int:
    """Write one length-prefixed frame.

    Args:
        stream: The binary stream to write to.
        frame: The snapshot to write.

    Returns:
        The number of bytes written, including the length prefix.
    """
    payload = encode(frame)
    stream.write(_LENGTH.pack(len(payload)))
    stream.write(payload)
    return _LENGTH.size + len(payload)


def read_frames(path: Path | str) -> Iterator[Frame]:
    """Replay a recording, one frame at a time.

    Frames are yielded lazily, so a recording is never held in memory whole.

    A run that was killed leaves a partial frame behind, which raises rather than being
    passed off as the end of the recording. Every whole frame before it has already
    been yielded, so a caller that expects to read a killed run can catch the error and
    keep what it got.

    Args:
        path: The recording to read.

    Yields:
        Each frame, in the order it was written.

    Raises:
        TruncatedRecordingError: When the recording ends mid-frame.
    """
    with Path(path).open("rb") as stream:
        while True:
            header = stream.read(_LENGTH.size)
            if not header:
                return
            if len(header) < _LENGTH.size:
                msg = f"Recording ends in a {len(header)}-byte length prefix."
                raise TruncatedRecordingError(msg)

            (length,) = _LENGTH.unpack(header)
            payload = stream.read(length)
            if len(payload) < length:
                msg = (
                    f"Recording promises a {length}-byte frame but holds only "
                    f"{len(payload)} bytes of it."
                )
                raise TruncatedRecordingError(msg)

            yield decode(payload)


class FrameWriter:
    """A `FrameConsumer` that records every frame it is handed to a file.

    The stream is buffered rather than flushed per frame, so `close` is what commits
    the tail of a recording. A run that dies without it leaves a truncated last frame,
    which `read_frames` reports rather than hiding.

    A writer must survive being pickled, because Monty saves its state at the end of
    every epoch (`post_epoch` -> `save_state_dir` -> `torch.save(self.config, ...)`),
    the hook lives in that config, and there is no setting that turns the saving off.
    An open file is not picklable, so the stream is left out; the same will be true of
    a socket, whenever a consumer holds one.
    """

    def __init__(self, save_path: str) -> None:
        """Open the recording, replacing any previous run's.

        Args:
            save_path: The file to record to. A relative path lands under the repo's
                `local/`, since Hydra chdirs into its own output directory for a run.
        """
        path = Path(save_path)
        if not path.is_absolute():
            path = LOCAL_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.save_path = path
        self.frames = 0
        self.bytes_written = 0
        self._stream = path.open("wb")

    def __call__(self, frame: Frame) -> None:
        """Record one frame.

        Args:
            frame: The step snapshot.
        """
        self.bytes_written += write_frame(self._stream, frame)
        self.frames += 1

    def close(self) -> None:
        """Commit the recording."""
        self._stream.close()

    def __getstate__(self) -> dict:
        """Leave the open recording out of the pickle Monty takes of its config.

        A recording belongs to the run that opened it, so a restored writer reports
        where it wrote and how much, but cannot record: it is a record of a writer, not
        a working one.

        Returns:
            The writer's state, without its stream.
        """
        return {**self.__dict__, "_stream": None}
