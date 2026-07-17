# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Reducing Monty's values to what a frame is allowed to hold."""

from __future__ import annotations

from typing import Any, Mapping

import quaternion as qt
from tbp.monty.geometry import Rotation


def plain(value: Any) -> Any:  # noqa: ANN401
    """Reduce a value to plain data, walking into whatever holds it.

    A frame may carry only plain containers, numpy arrays, and primitives, and enforcing
    that is an aggregator's job rather than the codec's: the codec cannot know what a
    given value was supposed to mean, and refusing it late gives a worse error than
    converting it early.

    Args:
        value: The value as Monty reports it.

    Returns:
        The value, as plain data.
    """
    if isinstance(value, qt.quaternion):
        # Monty types a rotation as `QuaternionWXYZ`, a tuple of four floats, but hands
        # over a `quaternion.quaternion`. numpy-quaternion registers it as a numpy
        # scalar type whose `item` returns the quaternion again, so it would slip past
        # any check for a numpy scalar and reach the codec, which has never heard of it.
        # Monty's own encoder converts it the same way; this makes the value match the
        # type Monty already declares for it.
        return qt.as_float_array(value).tolist()

    if isinstance(value, Rotation):
        # `as_quat` is a method rather than a property, and gives back scalar-first
        # `wxyz` as an array. Listed, so that a rotation looks the same on the wire
        # whichever of the two ways Monty happened to hand it over.
        return value.as_quat().tolist()

    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        # A tuple would survive the wire as a list anyway, so it becomes one here rather
        # than changing shape somewhere less obvious.
        return [plain(item) for item in value]

    return value
