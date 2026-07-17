# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""The observations section of a frame, and the ids that key it.

Imports numpy only for its annotations, so this can be read where Monty is not
installed, which is the point of a frame carrying no Monty class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, NewType, TypedDict

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

# `AgentID` and `SensorID` mirror Monty's own, but are declared here rather than
# imported: a frame that carried a Monty type could only be read where Monty is
# installed. A `NewType` is erased at runtime, so these name what a key is without
# costing a byte on a wire or a dependency at the far end.
AgentID = NewType("AgentID", str)
SensorID = NewType("SensorID", str)


class SensorObservationsFrame(TypedDict, total=False):
    """What one sensor reports for one step.

    Mirrors Monty's `SensorObservation`, and is `total=False` for the same reason: which
    modalities a sensor reports is a property of the environment's configuration rather
    than of this code. A `semantic_3d` only exists because a config asked for it, so a
    consumer must expect any of these to be absent.

    The dtypes here are the contract, not a note on what habitat happened to hand over.
    Two of them are deliberately narrower than the float64 habitat reports, because
    together they are 40% of a frame's bytes and a frame is what goes to a disk or down
    a wire: a location is a position in metres, where float32 still resolves to a
    fraction of a micron across a scene, and a semantic id is a small integer.

    `DefaultObservationAggregator` passes habitat's values through untouched, so a frame
    it built can carry the wider dtypes Monty declares; the narrowing is
    `SmartObservationAggregator`'s.

    Attributes:
        rgba: The sensor's colour image, shape `(H, W, 4)`.
        depth: Distance per pixel in metres, shape `(H, W)`.
        semantic: Monty's `SemanticID` per pixel, `0` off object, shape `(H, W)`.
        semantic_3d: Habitat's flattened per-pixel `xyz` and semantic id, one row per
            pixel, shape `(H * W, 4)`.
        locations_rel_world: `semantic_3d`'s `xyz` half, unflattened back to the
            sensor's own frame so a consumer can treat it as an image, shape
            `(H, W, 3)`.
        sensor_frame_data: As Monty reports it.
        cam_to_world: As Monty reports it.
        pixel_loc: As Monty reports it.
        raw: As Monty reports it.
    """

    rgba: npt.NDArray[np.uint8]
    depth: npt.NDArray[np.float32]
    semantic: npt.NDArray[np.int32]
    semantic_3d: npt.NDArray[np.float64]
    locations_rel_world: npt.NDArray[np.float32]
    sensor_frame_data: npt.NDArray[np.int_]
    cam_to_world: npt.NDArray[np.float64]
    pixel_loc: npt.NDArray[np.float64]
    raw: npt.NDArray[np.uint8]


# Aliases rather than the `Dict` subclasses Monty uses for `AgentObservations` and
# `Observations`. A frame comes back off a wire as the plain dicts msgpack decodes to,
# so a subclass here would be a promise the codec cannot keep, and teaching the codec to
# rebuild them would couple it to this schema for no gain. Monty's containers never
# leave the process, so it is free to make them types.
AgentObservationsFrame = Dict[SensorID, SensorObservationsFrame]
ObservationsFrame = Dict[AgentID, AgentObservationsFrame]
