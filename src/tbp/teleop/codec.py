# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Turning a `Frame`, a `Command`, or a `CommandResult` into bytes, and back."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import msgpack
import numpy as np

from tbp.teleop.commands import Command, CommandResult
from tbp.teleop.frames import Frame

# Bumped whenever the encoded shape changes. A reader refuses a version it does not
# know rather than silently misreading it.
VERSION = 1

# The msgpack extension code carrying a numpy array. Arbitrary, but fixed: it is part
# of the format, so changing it is a version bump.
NDARRAY_EXT_CODE = 1


class UnsupportedVersionError(ValueError):
    """Raised when decoding bytes written by an incompatible codec version."""


class UnexpectedMessageError(ValueError):
    """Raised when decoding a message of a different kind than the one asked for."""


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
    return _encode("frame", frame)


def decode(data: bytes) -> Frame:
    """Decode a frame encoded by `encode`.

    The arrays are views onto `data` rather than copies, so they come back read-only,
    which matches a frame's contract of being a snapshot. A consumer that needs to
    write to one should copy it.

    Raises `UnsupportedVersionError` when `data` came from a different codec version,
    and `UnexpectedMessageError` when it is not a frame at all. Both come out of
    `_decode`, so they are named here rather than in a `Raises:` section.

    Args:
        data: The encoded frame.

    Returns:
        The decoded frame.
    """
    return Frame(**_decode("frame", data))


def encode_command(command: Command) -> bytes:
    """Encode a driver's command, for the control channel.

    Args:
        command: The command to encode.

    Returns:
        The encoded command.
    """
    return _encode("command", command)


def decode_command(data: bytes) -> Command:
    """Decode a command encoded by `encode_command`.

    Raises `UnsupportedVersionError` when `data` came from a different codec version,
    and `UnexpectedMessageError` when it is not a command at all. Both come out of
    `_decode`, so they are named here rather than in a `Raises:` section.

    Args:
        data: The encoded command.

    Returns:
        The decoded command.
    """
    return Command(**_decode("command", data))


def encode_result(result: CommandResult) -> bytes:
    """Encode the experiment's acknowledgement of a command.

    Args:
        result: The result to encode.

    Returns:
        The encoded result.
    """
    return _encode("result", result)


def decode_result(data: bytes) -> CommandResult:
    """Decode a result encoded by `encode_result`.

    Raises `UnsupportedVersionError` when `data` came from a different codec version,
    and `UnexpectedMessageError` when it is not a result at all. Both come out of
    `_decode`, so they are named here rather than in a `Raises:` section.

    Args:
        data: The encoded result.

    Returns:
        The decoded result.
    """
    return CommandResult(**_decode("result", data))


def _encode(kind: str, message: Any) -> bytes:  # noqa: ANN401
    """Encode a dataclass message under its kind.

    The kind names the message inside the envelope, so a reader can tell a frame from a
    choice, and refuse one when it asked for the other.

    Args:
        kind: What sort of message this is.
        message: The dataclass to encode.

    Returns:
        The encoded message.
    """
    envelope = {
        "version": VERSION,
        # Read the fields shallowly: `dataclasses.asdict` deep-copies every value it
        # walks, which would copy every array on its way out.
        kind: {field.name: getattr(message, field.name) for field in fields(message)},
    }
    return msgpack.packb(envelope, default=_pack_unsupported, use_bin_type=True)


def _decode(kind: str, data: bytes) -> dict:
    """Decode a message, insisting it is of the expected kind.

    Args:
        kind: The sort of message the caller is expecting.
        data: The encoded message.

    Returns:
        The message's fields.

    Raises:
        UnsupportedVersionError: When `data` was written by a different codec version.
        UnexpectedMessageError: When `data` is of some other kind.
    """
    envelope = msgpack.unpackb(data, ext_hook=_unpack_ext, raw=False)
    version = envelope.get("version")
    if version != VERSION:
        msg = f"Cannot decode a version {version} message; this codec writes {VERSION}."
        raise UnsupportedVersionError(msg)
    if kind not in envelope:
        carried = sorted(set(envelope) - {"version"})
        msg = f"Expected a {kind}, but the message carries {carried}."
        raise UnexpectedMessageError(msg)
    return envelope[kind]


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
        # Every built-in numpy scalar unwraps to a Python primitive here, but a library
        # may register a scalar type of its own whose `item` hands back the same object
        # (numpy-quaternion does). Such a value has to be normalized before it reaches a
        # frame, so it is refused below rather than passed on to fail less clearly.
        unwrapped = obj.item()
        if isinstance(unwrapped, (bool, int, float, complex, str, bytes)):
            return unwrapped
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
