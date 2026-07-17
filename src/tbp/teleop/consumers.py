# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Consumers that turn frames into something to look at.

These live apart from `aggregate` so that nothing on this side imports Monty. A
consumer takes frames, and frames are equally happy arriving from a live step hook or
from a recording, so the same consumer can draw a run that finished last week in a
process where Monty is not installed.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from tbp.teleop.frame import FrameConsumer
from tbp.teleop.recording import LOCAL_DIR

if TYPE_CHECKING:
    from tbp.teleop.frame import Frame


class ObservationSaver(FrameConsumer):
    """A `FrameConsumer` that saves each step's observations to a figure.

    A debugging aid for the aggregator: it draws every sensor's rgba and depth straight
    out of the frame, so what lands on disk is what the aggregator actually captured
    rather than what the model happens to be holding.

    Draws through `Figure` rather than `pyplot` so that saving needs no GUI backend and
    keeps no global figure state.
    """

    MODALITIES_TO_PLOT = ("rgba", "depth", "semantic")

    def __init__(self, save_path: str) -> None:
        """Initialize the consumer, clearing out any previous run's figures.

        Args:
            save_path: Directory to save figures to, under the repo's `local/`.
        """
        path = LOCAL_DIR / save_path
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        self.save_path = path
        self._index = 0

    def __call__(self, frame: Frame) -> None:
        """Save one figure of every sensor's rgba and depth for this step.

        Args:
            frame: The step snapshot.
        """
        sensors = [
            (agent_id, sensor_id, observation)
            for agent_id, agent_observations in (frame.observations or {}).items()
            for sensor_id, observation in agent_observations.items()
            if any(modality in observation for modality in self.MODALITIES_TO_PLOT)
        ]
        if not sensors:
            return

        fig = Figure(figsize=(4 * len(sensors), 4 * len(self.MODALITIES_TO_PLOT)))
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(self.MODALITIES_TO_PLOT), len(sensors), squeeze=False)
        for column, (agent_id, sensor_id, observation) in enumerate(sensors):
            for row, modality in enumerate(self.MODALITIES_TO_PLOT):
                ax = axes[row][column]
                ax.set_axis_off()
                ax.set_title(f"{agent_id} / {sensor_id}\n{modality}", fontsize=8)
                if modality not in observation:
                    ax.text(
                        0.5,
                        0.5,
                        f"no {modality}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                if modality == "rgba":
                    ax.imshow(np.squeeze(observation[modality]))

                elif modality == "depth":
                    depth = np.squeeze(observation[modality])
                    on_object = (
                        np.squeeze(observation["semantic"]) != 0
                        if "semantic" in observation
                        else np.zeros(depth.shape, dtype=bool)
                    )
                    # An all-off-object sensor selects no depths at all, and `nanmin`
                    # has no identity to fall back on, so scale to the whole frame.
                    if on_object.any():
                        min_on_object_depth = np.nanmin(depth[on_object])
                        max_on_object_depth = np.nanmax(depth[on_object])
                        depth_range = max_on_object_depth - min_on_object_depth
                        vmin = min_on_object_depth - 0.1 * depth_range
                        vmax = max_on_object_depth + 0.1 * depth_range
                    else:
                        vmin, vmax = np.nanmin(depth), np.nanmax(depth)

                    ax.imshow(depth, cmap="gray_r", vmin=vmin, vmax=vmax)

                elif modality == "semantic" and "semantic" in observation:
                    semantic_mask = np.squeeze(observation[modality])
                    ax.imshow(semantic_mask, cmap="gray", vmin=0, vmax=1)

        fig.suptitle(f"Step {frame.step}")
        # `step` restarts every episode, so the file name leads with a running count to
        # keep each step of a multi-episode run rather than overwriting it.
        fig.savefig(self.save_path / f"{self._index:05d}_step_{frame.step:04d}.png")
        self._index += 1

    def close(self) -> None:
        """Hold nothing open; each figure is written as it is drawn."""
