# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Turning a `Frame` into bytes, and back."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import msgpack
import numpy as np

from tbp.teleop.frame import Frame

# Bumped whenever the encoded shape changes. A reader refuses a version it does not
# know rather than silently misreading it.
VERSION = 1

# The msgpack extension code carrying a numpy array. Arbitrary, but fixed: it is part
# of the format, so changing it is a version bump.
NDARRAY_EXT_CODE = 1


class UnsupportedVersionError(ValueError):
    """Raised when decoding bytes written by an incompatible codec version."""


def encode(frame: Frame) -> bytes:
    """Encode a frame for a socket or a file.

    Arrays travel as their raw bytes alongside a dtype and shape rather than as text,
    so a frame costs roughly what its arrays weigh. Nothing is pickled, so the reader
    may be a different interpreter running a different numpy than the writer, which is
    the whole point of a frame carrying no Monty class.

    Args:
        frame: The snapshot to encode.

    Returns:
        The encoded frame.
    """
    envelope = {
        "version": VERSION,
        # Read the fields shallowly: `dataclasses.asdict` deep-copies every value it
        # walks, which would copy every array on its way out.
        "frame": {field.name: getattr(frame, field.name) for field in fields(frame)},
    }
    return msgpack.packb(envelope, default=_pack_unsupported, use_bin_type=True)


def decode(data: bytes) -> Frame:
    """Decode a frame encoded by `encode`.

    The arrays are views onto `data` rather than copies, so they come back read-only,
    which matches a frame's contract of being a snapshot. A consumer that needs to
    write to one should copy it.

    Args:
        data: The encoded frame.

    Returns:
        The decoded frame.

    Raises:
        UnsupportedVersionError: When `data` was written by a different codec version.
    """
    envelope = msgpack.unpackb(data, ext_hook=_unpack_ext, raw=False)
    version = envelope.get("version")
    if version != VERSION:
        msg = f"Cannot decode a version {version} frame; this codec writes {VERSION}."
        raise UnsupportedVersionError(msg)
    return Frame(**envelope["frame"])


def _pack_unsupported(obj: Any) -> Any:  # noqa: ANN401
    """Encode a value msgpack has no native type for.

    Args:
        obj: The value to encode.

    Returns:
        A numpy array as an extension type, or a numpy scalar as its Python equivalent.

    Raises:
        TypeError: When `obj` is of a type a frame is not allowed to carry.
    """
    if isinstance(obj, np.ndarray):
        # `tobytes` is C-ordered whatever the array's own layout is, so a transposed or
        # sliced view travels as its own contents rather than its base's.
        payload = msgpack.packb(
            [obj.dtype.str, list(obj.shape), obj.tobytes()], use_bin_type=True
        )
        return msgpack.ExtType(NDARRAY_EXT_CODE, payload)
    if isinstance(obj, np.generic):
        return obj.item()
    msg = f"A frame cannot carry {type(obj).__name__}."
    raise TypeError(msg)


def _unpack_ext(code: int, data: bytes) -> Any:  # noqa: ANN401
    """Decode an extension type.

    Args:
        code: The extension code.
        data: The extension payload.

    Returns:
        The numpy array the payload carries, or the extension itself when unrecognized.
    """
    if code == NDARRAY_EXT_CODE:
        dtype_str, shape, buffer = msgpack.unpackb(data, raw=False)
        # `dtype.str` carries endianness ("<f8"), not just a width, so a frame written
        # on one machine decodes to the same numbers on another.
        return np.frombuffer(buffer, dtype=np.dtype(dtype_str)).reshape(shape)
    return msgpack.ExtType(code, data)
