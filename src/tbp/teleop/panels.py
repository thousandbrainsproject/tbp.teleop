# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

from tbp.teleop.goals import enacted_goal, passed_attention_filter, proposed_goals
from tbp.teleop.helpers import (
    ChannelView,
    EvidenceHistory,
    FeatureInset,
    GoalTrace,
    SpreadTrace,
    corner_rect,
    draw_2d_segments,
    draw_buffer_series,
    is_3d,
    planar_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from tbp.monty.cmp import Goal
    from tbp.monty.frameworks.agents import AgentID
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        AgentObservations,
        Monty,
        Observations,
        SensorModule,
    )
    from tbp.monty.frameworks.models.goal_generation import SpreadRecord
    from tbp.monty.frameworks.models.object_model import GraphObjectModel

# Figure-fraction rectangle of the spread replay slider along the figure's bottom
# edge, below the interactive buttons and the speed slider.
SPREAD_SLIDER_BOTTOM, SPREAD_SLIDER_HEIGHT = 0.012, 0.022


@dataclass(frozen=True)
class _NodeHighlights:
    """Nodes of the drawn MLH graph to color over its point cloud.

    Attributes:
        replay: The nodes a replayed spread has reached so far, in the order they
            were reached, or `None` when no spread is being replayed.
        replay_total: The number of nodes the replayed spread reached in total.
        replay_label: The legend label of the replayed spread.
        weights: The inhibition weight of every node, or `None` when not shown. Only
            drawn when no spread is being replayed.
    """

    replay: npt.NDArray[np.int_] | None
    replay_total: int
    replay_label: str | None
    weights: npt.NDArray[np.float64] | None

    def draw(self, ax: Axes, pos: npt.NDArray[np.float64], *, three_d: bool) -> bool:
        """Color the highlighted nodes on the MLH axis.

        Args:
            ax: The MLH axis.
            pos: The graph node positions, shape `(N, 3)`.
            three_d: Whether `ax` is the 3D view, which plots `(y, x, z)`, rather
                than the 2D view, which plots `(x, y)`.

        Returns:
            Whether anything with a legend label was drawn.
        """

        def coords(points: npt.NDArray[np.float64]) -> tuple:
            points = np.atleast_2d(points)
            if three_d:
                return points[:, 1], points[:, 0], points[:, 2]
            return points[:, 0], points[:, 1]

        extra = {"depthshade": False} if three_d else {}
        if self.replay is not None:
            if len(self.replay) == 0:
                return False
            ax.scatter(
                *coords(pos[self.replay]),
                c=np.arange(len(self.replay)),
                cmap="viridis",
                vmin=0,
                vmax=max(self.replay_total - 1, 1),
                s=12,
                label=f"{self.replay_label} ({len(self.replay)}/{self.replay_total})",
                zorder=4,
                **extra,
            )
            ax.scatter(
                *coords(pos[self.replay[-1]]),
                facecolors="none",
                edgecolors="magenta",
                s=90,
                linewidths=1.5,
                zorder=5,
                **extra,
            )
            return True
        if self.weights is None:
            return False
        inhibited = np.nonzero(self.weights > 0)[0]
        if len(inhibited) == 0:
            return False
        ax.scatter(
            *coords(pos[inhibited]),
            c=self.weights[inhibited],
            cmap="Reds",
            # Start the colormap below 0 so nearly decayed nodes stay visible.
            vmin=-0.5,
            vmax=1.0,
            s=10,
            label=f"inhibited ({len(inhibited)})",
            zorder=4,
            **extra,
        )
        return True


class SimulatorPanel:
    """The Simulator section: a view finder above the RGB patch the sensor sees.

    Owns its two axes within the figure's left column and redraws them each frame from
    the driving sensor module's observations. Which sensor module drives the section is
    resolved by the plotter (it follows the selected channel, or the displayed LM's
    first sensor-module channel) and passed in per draw.
    """

    def __init__(self, fig: Figure, spec) -> None:
        """Lay out the view-finder and RGB-patch axes in the Simulator column.

        Args:
            fig: The figure to draw on.
            spec: The Simulator column's gridspec subplot spec.
        """
        self.fig = fig
        sim_grid = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=spec, height_ratios=[2, 1], hspace=0.3
        )
        self.ax_view = fig.add_subplot(sim_grid[0, 0])
        self.ax_rgb = fig.add_subplot(sim_grid[1, 0])

    def draw(
        self,
        observations: Observations,
        sm: SensorModule | None,
        sm_id: str | None,
    ) -> None:
        """Draw the view finder and RGB patch for the driving sensor module.

        Args:
            observations: The observations from the most recent step.
            sm: The sensor module driving the section, or `None` when the displayed LM
                has no sensor-module channel yet.
            sm_id: The id of that sensor module, or `None`.
        """
        agent_id = (
            self._resolve_obs_agent_id(observations, sm_id)
            if sm_id is not None
            else None
        )
        agent_obs: AgentObservations = (
            observations.get(agent_id, {}) if agent_id is not None else {}
        )
        self._draw_view_finder(agent_obs, sm, sm_id)
        self._draw_rgb_patch(agent_obs, sm_id)

    @staticmethod
    def _resolve_obs_agent_id(
        observations: Observations, sensor_module_id: str
    ) -> AgentID | None:
        """Find the agent id whose observations contain the given sensor module.

        Args:
            observations: The observations from the most recent step.
            sensor_module_id: The sensor module id to search for.

        Returns:
            The matching agent id, or `None` if no agent carries the sensor module.
        """
        for agent_id, agent_observations in observations.items():
            if sensor_module_id in agent_observations:
                return agent_id
        return None

    def _draw_view_finder(
        self,
        agent_obs: AgentObservations,
        sm: SensorModule | None,
        sm_id: str | None,
    ) -> None:
        """Draw the view finder, outlining the patch when a pixel location is known.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
            sm: The sensor module driving the Simulator (for its raw-pixel telemetry).
            sm_id: The id of that sensor module, used to read its patch observation.
        """
        ax = self.ax_view
        ax.cla()
        ax.set_title("View finder")
        ax.set_axis_off()
        if SensorID("view_finder") not in agent_obs:
            return
        rgba = np.asarray(agent_obs[SensorID("view_finder")]["rgba"])
        raw = sm._snapshot_telemetry.raw_observations if sm is not None else []
        patch_obs = agent_obs.get(sm_id) if sm_id is not None else None
        has_outline = (
            len(raw) > 0
            and "pixel_loc" in raw[-1]
            and patch_obs is not None
            and "depth" in patch_obs
        )
        if has_outline:
            patch_size = np.asarray(patch_obs["depth"]).shape[0]
            rgba = add_patch_outline_to_view_finder(
                rgba, np.array(raw[-1]["pixel_loc"]), patch_size
            )
        ax.imshow(rgba, zorder=-99)
        if not has_outline:
            shape = rgba.shape
            ax.add_patch(
                plt.Rectangle(
                    (shape[1] * 4.5 // 10, shape[0] * 4.5 // 10),
                    shape[1] / 10,
                    shape[0] / 10,
                    fc="none",
                    ec="white",
                )
            )

    def _draw_rgb_patch(self, agent_obs: AgentObservations, sm_id: str | None) -> None:
        """Draw what the sensor module sees, preferring RGB over depth.

        Args:
            agent_obs: The observing agent's per-sensor observations this step.
            sm_id: The id of the sensor module driving the Simulator.
        """
        ax = self.ax_rgb
        ax.cla()
        ax.set_axis_off()
        patch = agent_obs.get(sm_id) if sm_id is not None else None
        if patch is not None and "rgba" in patch:
            ax.imshow(np.asarray(patch["rgba"]))


class MontyPanel:
    """The Monty column: the selected channel's main graph and its feature inset.

    Owns the Monty column's single physical region, the axis-layout state machine, and
    the "Input Feature" corner inset stacked over it. During an exploratory step it
    draws the selected channel's buffered points (a flat edge cloud for a 2D sensor
    module, else a 3D cube with three head-on projections); during a matching step it
    draws the displayed LM's most likely hypothesis graph and location marker; otherwise
    a centered placeholder. Axes are rebuilt only when the layout or projection changes
    (a matplotlib axis cannot switch between 2D and 3D in place), to avoid flicker. The
    selection and per-channel features come from a `ChannelView`.

    During a matching step the drawn graph is always the selected channel's model of
    the MLH object, whether or not that channel has received input yet (the MLH
    location is shared by all of an object's channel models); when the MLH object has
    no model of the selected channel, a placeholder says so. A badge in the corner
    states whether the selected channel has received input this episode and on the
    latest step (with the object ID, for a learning-module channel).

    When the displayed LM's goal generator inhibits child objects (a
    `ChildObjectsGoalGenerator`), an "Inhibition" button in the corner of the MLH
    view toggles coloring the drawn graph's nodes by their inhibition weight. While
    it is on, and the LM spread inhibition through the drawn graph on the latest
    step, a temporary slider along the bottom of the figure replays that spread, from
    the first node it reached to the last.
    """

    def __init__(
        self,
        fig: Figure,
        spec,
        channel_view: ChannelView,
        goal_trace: GoalTrace,
        spread_trace: SpreadTrace,
        *,
        show_inhibition: bool,
        on_toggle_inhibition: Callable[[bool], None],
    ) -> None:
        """Bind the Monty column to its figure region and selection.

        Lays out the initial single 2D axis, matching the layout the placeholder and
        inference 2D views start from, and the (initially hidden) inhibition button.

        Args:
            fig: The figure to draw on.
            spec: The Monty column's gridspec subplot spec.
            channel_view: The selected LM/channel and per-channel feature accessors.
            goal_trace: The per-LM, per-graph trace of past model-frame goal targets.
            spread_trace: The inhibition spreads of the latest step.
            show_inhibition: Whether inhibition weights start out shown.
            on_toggle_inhibition: Called with the new state when the inhibition
                button is clicked; expected to repaint the figure.
        """
        self.fig = fig
        self.spec = spec
        self.channel_view = channel_view
        self.goal_trace = goal_trace
        self.spread_trace = spread_trace
        self.show_inhibition = show_inhibition
        self._on_toggle_inhibition = on_toggle_inhibition
        self._ax = fig.add_subplot(spec)
        self._mode: str | None = "single"
        self._projection: str | None = None
        self._proj_axes: list[Axes] = []
        self._inset: FeatureInset | None = None
        self._inhibition_button = self._build_inhibition_button()
        self._spread_slider: Slider | None = None
        self._slider_record: SpreadRecord | None = None
        monty = spec.get_position(fig)
        self._status_badge = fig.text(
            monty.x1,
            monty.y0 + 0.005,
            "",
            ha="right",
            va="bottom",
            fontsize=8,
            zorder=10,
            visible=False,
        )

    def draw_input_status(self) -> None:
        """Show what input the displayed LM has received on the selected channel.

        A badge in the bottom-right corner of the Monty column states whether the
        channel has had input this episode (and how long ago) and what it received
        on the latest step (the object ID, for a learning-module channel). It is
        green when the channel had input on the latest step, amber when it only
        had input earlier, and grey when it has had none this episode.
        """
        channel = self.channel_view.channel
        if channel is None:
            self._status_badge.set_visible(False)
            return
        status = self.channel_view.input_status(channel)
        if status.current_input is not None:
            episode = "episode: input received"
            color = "#c8f0c0"
        elif status.received_this_episode:
            episode = "episode: input received"
            if status.steps_since_input is not None:
                episode += f" ({status.steps_since_input} steps ago)"
            color = "#ffe2a8"
        else:
            episode = "episode: no input yet"
            color = "0.9"
        latest = f"this step: {status.current_input or 'no input'}"
        self._status_badge.set_text(f"{episode}\n{latest}")
        self._status_badge.set_bbox(
            {"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "0.5"}
        )
        self._status_badge.set_visible(True)

    def draw_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message in the Monty panel.

        Args:
            message: The text to display.
        """
        self._hide_inhibition_controls()
        ax = self._ensure_projection(None)
        ax.cla()
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )

    def draw_selected_channel(self, channel: str, pts: npt.NDArray[np.float64]) -> None:
        """Draw the selected channel's buffered points (exploratory step).

        A 2D sensor-module channel is drawn as a flat edge cloud mirroring the inference
        2D view; every other channel is drawn as a 3D cube with three head-on
        projections, since a partially built 3D graph may look planar by chance.

        Args:
            channel: The selected input channel.
            pts: The channel's `(M, 3)` non-padded location points.
        """
        self._hide_inhibition_controls()
        if isinstance(self.channel_view.resolve_sm_channel(channel), TwoDSensorModule):
            self._draw_selected_channel_2d(channel, pts)
            return
        main_ax, proj_axes = self._ensure_projected()
        groups = self.channel_view.channel_groups(channel, pts)
        draw_buffer_series(
            main_ax,
            proj_axes,
            groups,
            title="",
            title_fontsize=None,
        )

    def draw_mlh(self) -> None:
        """Render the most likely hypothesis graph and its location marker.

        Uses the selected channel's stored graph when that channel is part of the MLH
        graph, otherwise the graph's first sensor-module channel. Falls back to a "No
        MLH" placeholder when there is no current hypothesis or the graph cannot be
        retrieved. Planar graphs are drawn as edge-oriented segments; all other graphs
        as a 3D point cloud. When the displayed LM's goal generator holds a current
        hypothesis-testing goal targeting the drawn graph, the goal's model-frame
        target location is marked as a green dot labeled with the LM id; the LM's
        earlier goals on the same graph are drawn beneath it as faint green dots.
        Goals whose target was selected from a particular input channel's graph are
        only drawn on that channel's graph. On the step a goal is proposed, the
        hypothesis it was compared against (the second most likely object, or the MLH
        object's second most likely pose) is superimposed in light blue, posed
        relative to the MLH.

        With inhibition shown, the drawn graph's inhibited nodes are colored by their
        inhibition weight or, while a spread through the drawn graph is being
        replayed, the nodes the spread reached so far are colored by the order in
        which it reached them.
        """
        lm = self.channel_view.lm
        graph = None
        channel = None
        message = "No MLH"
        mlh = lm._get_current_mlh()
        if mlh and mlh.get("graph_id") not in (None, "no_observations_yet"):
            graph_id = mlh["graph_id"]
            if graph_id in lm.graph_memory.get_memory_ids():
                channel = self._mlh_channel(graph_id)
                if channel is not None:
                    graph = lm.graph_memory.get_graph(graph_id, channel)
                else:
                    message = (
                        f"MLH ({graph_id}) has no model of {self.channel_view.channel}"
                    )

        if (
            graph is None
            or getattr(graph, "pos", None) is None
            or len(np.asarray(graph.pos)) == 0
        ):
            self.draw_placeholder(message)
            self._inhibition_button.ax.set_visible(self._inhibition_gsg() is not None)
            return
        pos = np.asarray(graph.pos)

        self._inhibition_button.ax.set_visible(self._inhibition_gsg() is not None)
        record = self._spread_to_replay(mlh["graph_id"], channel)
        self._sync_spread_slider(record)
        highlights = _NodeHighlights(
            replay=self._replayed_nodes(record),
            replay_total=len(record.node_order) if record is not None else 0,
            replay_label=self._replay_label(record),
            weights=self._inhibition_weights(mlh["graph_id"], channel, graph),
        )

        color = self._mlh_marker_color(mlh, lm.object_evidence_threshold)
        if is_3d(pos):
            ax = self._ensure_projection("3d")
            self._show_mlh_3d(ax, mlh, channel, pos, color, highlights)
        else:
            ax = self._ensure_projection(None)
            self._show_mlh_2d(ax, mlh, channel, graph, pos, color, highlights)

    def _goal_in_model_frame(
        self, graph_id: str, channel: str
    ) -> tuple[npt.NDArray[np.float64], str] | None:
        """The displayed LM's current goal target on the drawn MLH graph.

        An LM's goal generator snapshots the model-frame target location (and the
        graph and input channel it was selected from) in the goal's `info` when the
        goal is created, so the target can be marked directly on the stored graph.
        The marker is only meaningful while the drawn MLH graph is the one the goal
        was proposed on, and its channel is the one the target was selected from
        (when recorded).

        Args:
            graph_id: The id of the MLH graph being drawn.
            channel: The input channel whose graph is being drawn.

        Returns:
            The `(location, sender_id)` of the goal target in the graph's model
            frame, or `None` when the LM has no goal generator, no current goal, or
            the goal targets a different graph or channel.
        """
        gsg = getattr(self.channel_view.lm, "gsg", None)
        goal = getattr(gsg, "output_goal", None)
        if goal is None:
            return None
        location = goal.info.get("model_frame_target_loc")
        if location is None or goal.info.get("model_frame_graph_id") != graph_id:
            return None
        if goal.info.get("model_frame_input_channel") not in (None, channel):
            return None
        return np.asarray(location, dtype=float), str(goal.sender_id)

    def _past_goals(self, graph_id: str, channel: str) -> npt.NDArray[np.float64]:
        """The displayed LM's earlier goal targets on the drawn MLH graph.

        Args:
            graph_id: The id of the MLH graph being drawn.
            channel: The input channel whose graph is being drawn.

        Returns:
            The `(K, 3)` model-frame goal targets the displayed LM selected from this
            channel's graph this episode, including the current goal (drawn over it).
        """
        return self.goal_trace.locations(
            self.channel_view.lm.learning_module_id, graph_id, channel
        )

    # ------------------------- Inhibition -------------------------

    def _build_inhibition_button(self) -> Button:
        """Add the (initially hidden) inhibition toggle to the MLH view's corner.

        Returns:
            The toggle button.
        """
        monty = self.spec.get_position(self.fig)
        ax = self.fig.add_axes([monty.x0, monty.y0, 0.09, 0.03])
        # Keep the button above the MLH axis, which is re-added whenever its
        # projection changes.
        ax.set_zorder(10)
        button = Button(ax, self._inhibition_caption())
        button.label.set_fontsize(8)
        button.on_clicked(self._on_inhibition_clicked)
        ax.set_visible(False)
        return button

    def _inhibition_caption(self) -> str:
        return f"Inhibition: {'on' if self.show_inhibition else 'off'}"

    def _on_inhibition_clicked(self, _event: object) -> None:
        """Toggle showing inhibition weights.

        Args:
            _event: The matplotlib button event (unused).
        """
        self.show_inhibition = not self.show_inhibition
        self._inhibition_button.label.set_text(self._inhibition_caption())
        self._on_toggle_inhibition(self.show_inhibition)

    def _hide_inhibition_controls(self) -> None:
        self._inhibition_button.ax.set_visible(False)
        self._sync_spread_slider(None)

    def _inhibition_gsg(self) -> object | None:
        """The displayed LM's goal generator, if it inhibits child objects.

        Returns:
            The goal generator, or `None` when it has no inhibition weights.
        """
        gsg = getattr(self.channel_view.lm, "gsg", None)
        return gsg if hasattr(gsg, "get_inhibition_weights") else None

    def _inhibition_weights(
        self, graph_id: str, channel: str, graph: GraphObjectModel
    ) -> npt.NDArray[np.float64] | None:
        """The inhibition weight of each node of the drawn graph, when shown.

        Args:
            graph_id: The id of the MLH graph being drawn.
            channel: The input channel whose graph is being drawn.
            graph: The drawn graph.

        Returns:
            One weight in [0, 1] per node, or `None` when inhibition is not shown,
            or the drawn graph does not store object IDs (and so is never
            inhibited).
        """
        gsg = self._inhibition_gsg()
        if not self.show_inhibition or gsg is None:
            return None
        if "object_id" not in (graph.feature_mapping or {}):
            return None
        return np.asarray(gsg.get_inhibition_weights(graph_id, channel), dtype=float)

    def _spread_to_replay(self, graph_id: str, channel: str) -> SpreadRecord | None:
        """The latest step's spread through the drawn graph, when inhibition is shown.

        Args:
            graph_id: The id of the MLH graph being drawn.
            channel: The input channel whose graph is being drawn.

        Returns:
            The spread, or `None` when inhibition is not shown or the displayed LM
            did not spread through the drawn graph on the latest step.
        """
        if not self.show_inhibition:
            return None
        return self.spread_trace.find(
            self.channel_view.lm.learning_module_id, graph_id, channel
        )

    def _sync_spread_slider(self, record: SpreadRecord | None) -> None:
        """Show the replay slider for a spread, or remove it when there is none.

        The slider is only rebuilt when the spread changes, so its position is kept
        while the same spread is redrawn (e.g. on a selector click).

        Args:
            record: The spread to replay, or `None` to remove the slider.
        """
        if record is self._slider_record:
            return
        if self._spread_slider is not None:
            self._spread_slider.ax.remove()
            self._spread_slider = None
        self._slider_record = record
        if record is None:
            return
        num_nodes = len(record.node_order)
        monty = self.spec.get_position(self.fig)
        width = monty.x1 - monty.x0
        ax = self.fig.add_axes(
            [
                monty.x0 + 0.1 * width,
                SPREAD_SLIDER_BOTTOM,
                0.8 * width,
                SPREAD_SLIDER_HEIGHT,
            ]
        )
        self._spread_slider = Slider(
            ax,
            "Spread",
            0,
            num_nodes,
            valinit=num_nodes,
            valstep=1,
            valfmt=f"%d / {num_nodes}",
        )
        self._spread_slider.on_changed(self._on_spread_slider_changed)

    def _on_spread_slider_changed(self, _value: float) -> None:
        """Redraw the MLH view at the new replay position.

        Args:
            _value: The slider value (read back from the slider when drawing).
        """
        self.draw_mlh()
        self.fig.canvas.draw_idle()

    def _replayed_nodes(
        self, record: SpreadRecord | None
    ) -> npt.NDArray[np.int_] | None:
        """The nodes the replayed spread has reached at the slider's position.

        Args:
            record: The spread being replayed, or `None`.

        Returns:
            The reached nodes in the order they were reached, or `None` when no
            spread is being replayed.
        """
        if record is None or self._spread_slider is None:
            return None
        return np.asarray(record.node_order[: int(self._spread_slider.val)], dtype=int)

    def _replay_label(self, record: SpreadRecord | None) -> str | None:
        if record is None:
            return None
        names = self.channel_view.object_id_names(record.input_channel)
        name = names.get(int(record.object_id), f"object {int(record.object_id)}")
        return f"spread of {name}"

    def _second_hypothesis_overlay(
        self, graph_id: str
    ) -> tuple[npt.NDArray[np.float64], str] | None:
        """The compared hypothesis's graph, posed in the drawn MLH graph's frame.

        Only available on the step the displayed LM proposed a goal on the drawn
        graph. The second hypothesis's graph is mapped from its learned frame to the
        environment and then into the MLH graph's learned frame, i.e. the inverse of
        `EvidenceGoalGenerator._transform_to_second_mlh_rf`, so it overlays the MLH
        graph as the two hypotheses predict the objects would overlap in the world.

        Args:
            graph_id: The id of the MLH graph being drawn.

        Returns:
            The `(N, 3)` transformed node positions and a legend label, or `None`
            when no goal was proposed on this graph this step.
        """
        lm = self.channel_view.lm
        comparison = self.goal_trace.comparison_by_lm.get(lm.learning_module_id)
        if comparison is None or comparison[0]["graph_id"] != graph_id:
            return None
        top, second = comparison
        second_id = second["graph_id"]
        channel = self._mlh_channel(second_id)
        if channel is None:
            return None
        graph = lm.graph_memory.get_graph(second_id, channel)
        if getattr(graph, "pos", None) is None or len(graph.pos) == 0:
            return None
        # MLH rotations map displacements into the model, so `.inv()` takes model
        # coordinates to the environment.
        env_pts = (
            second["rotation"]
            .inv()
            .apply(np.asarray(graph.pos, dtype=float) - second["location"])
        )
        pts = top["rotation"].apply(env_pts) + top["location"]
        label = "2nd pose" if second_id == graph_id else f"2nd object ({second_id})"
        return pts, label

    def draw_feature_inset(self) -> None:
        """Draw the Monty section's "Input Feature" inset for the selected channel."""
        if self._inset is None:
            self._inset = FeatureInset(self.fig, self.channel_view)
        monty = self.spec.get_position(self.fig)
        rect = corner_rect(monty, width_frac=0.32, height=0.18, top_pad=0.07)
        self._inset.draw(self.channel_view.channel, rect)

    def _clear_axes(self) -> None:
        """Remove the main Monty axis and any projection axes below it."""
        self._ax.remove()
        for ax in self._proj_axes:
            ax.remove()
        self._proj_axes = []

    def _ensure_projection(self, projection: str | None) -> Axes:
        """Recreate the Monty axis as a single panel with the given projection.

        A matplotlib axis cannot switch between 2D and 3D in place, so the axis is
        removed and re-added whenever the layout or projection changes.

        Args:
            projection: The desired projection (`"3d"` or `None` for 2D).

        Returns:
            The current single Monty axis with the requested projection.
        """
        if self._mode == "single" and self._projection == projection:
            return self._ax
        self._clear_axes()
        self._ax = self.fig.add_subplot(self.spec, projection=projection)
        self._mode = "single"
        self._projection = projection
        return self._ax

    def _ensure_projected(self) -> tuple[Axes, list[Axes]]:
        """Lay the Monty column out as a 3D cloud over a row of three projections.

        Returns:
            The 3D main axis and the three 2D projection axes (XY, XZ, YZ).
        """
        if self._mode == "projected":
            return self._ax, self._proj_axes
        self._clear_axes()
        grid = GridSpecFromSubplotSpec(
            2,
            3,
            subplot_spec=self.spec,
            height_ratios=[3, 1],
            hspace=0.3,
            wspace=0.35,
        )
        self._ax = self.fig.add_subplot(grid[0, :], projection="3d")
        self._proj_axes = [self.fig.add_subplot(grid[1, j]) for j in range(3)]
        self._mode = "projected"
        self._projection = "3d"
        return self._ax, self._proj_axes

    def _draw_selected_channel_2d(
        self, channel: str, pts: npt.NDArray[np.float64]
    ) -> None:
        """Draw a 2D sensor-module channel's buffer as a planar edge cloud.

        Mirrors the inference 2D MLH view: a single flat rectilinear axis of
        hsv-colored dots with dashed segments along the edge tangent where the pose is
        fully defined, rather than the 3D cube and projections used for 3D channels.

        Args:
            channel: The selected 2D sensor-module channel.
            pts: The channel's `(M, 3)` non-padded location points.
        """
        ax = self._ensure_projection(None)
        ax.cla()
        x, y = pts[:, 0], pts[:, 1]
        colors, edge_mask, tangents = planar_style(
            len(x),
            self.channel_view.aligned_feature(channel, "hsv"),
            self.channel_view.aligned_feature(channel, "pose_fully_defined"),
            self.channel_view.aligned_feature(channel, "pose_vectors"),
        )
        draw_2d_segments(ax, x, y, colors, edge_mask, tangents)
        ax.set_title("")

    def _mlh_channel(self, graph_id: str) -> str | None:
        """Choose which channel's stored graph to render for the MLH.

        Args:
            graph_id: The MLH graph id.

        Returns:
            The selected channel when it is part of the graph, else `None` when it is
            not (the graph has no model of that channel). With no channel selected
            yet, the graph's first sensor-module channel (or `None` without one).
        """
        lm = self.channel_view.lm
        channels = lm.get_input_channels_in_graph(graph_id)
        selected = self.channel_view.channel
        if selected is not None:
            return selected if selected in channels else None
        sender_types = lm.buffer.channel_sender_types
        sm_channels = [c for c in channels if sender_types.get(c) == "SM"]
        return sm_channels[0] if sm_channels else None

    @staticmethod
    def _mlh_marker_color(mlh: dict, evidence_threshold: float | None) -> str:
        """Color the MLH marker by whether its evidence clears the threshold.

        Args:
            mlh: The current most likely hypothesis.
            evidence_threshold: The object evidence threshold, or `None`.

        Returns:
            `"red"` when above threshold (or threshold unknown), else `"gray"`.
        """
        if evidence_threshold is None:
            return "red"
        return "red" if mlh["evidence"] > evidence_threshold else "gray"

    def _show_mlh_3d(
        self,
        ax: Axes,
        mlh: dict,
        channel: str,
        pos: npt.NDArray[np.float64],
        mlh_color: str,
        highlights: _NodeHighlights,
    ) -> None:
        """Render a 3D graph as a point cloud with the MLH location marked.

        When the displayed LM holds a current goal targeting this graph, its
        model-frame target location is marked as a labeled green dot, over faint
        green dots at the LM's earlier goal targets on this graph.

        Args:
            ax: The 3D Monty axis.
            mlh: The current most likely hypothesis.
            channel: The input channel whose graph is drawn.
            pos: The graph node positions, shape `(N, 3)`.
            mlh_color: The MLH location marker color.
            highlights: The inhibited or spread-reached nodes to color.
        """
        goal_marker = self._goal_in_model_frame(mlh["graph_id"], channel)
        past_goals = self._past_goals(mlh["graph_id"], channel)
        overlay = self._second_hypothesis_overlay(mlh["graph_id"])
        ax.cla()
        ax.scatter(pos[:, 1], pos[:, 0], pos[:, 2], c="black", s=2)
        highlighted = highlights.draw(ax, pos, three_d=True)
        if overlay is not None:
            second_pts, second_label = overlay
            ax.scatter(
                second_pts[:, 1],
                second_pts[:, 0],
                second_pts[:, 2],
                c="lightskyblue",
                s=3,
                alpha=0.6,
                label=second_label,
                depthshade=False,
            )
        ax.scatter(
            mlh["location"][1],
            mlh["location"][0],
            mlh["location"][2],
            c=mlh_color,
            s=15,
        )
        if len(past_goals):
            ax.scatter(
                past_goals[:, 1],
                past_goals[:, 0],
                past_goals[:, 2],
                c="limegreen",
                alpha=0.8,
                s=70,
                label="past goals",
                depthshade=False,
            )
        if goal_marker is not None:
            location, sender = goal_marker
            ax.scatter(
                location[1],
                location[0],
                location[2],
                c="green",
                s=200,
                label=f"goal ({sender})",
                depthshade=False,
            )
        if (
            goal_marker is not None
            or len(past_goals)
            or overlay is not None
            or highlighted
        ):
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"MLH ({mlh['graph_id']})")
        ax.set_axis_off()
        ax.set_aspect("equal")

    def _show_mlh_2d(
        self,
        ax: Axes,
        mlh: dict,
        channel: str,
        graph: GraphObjectModel,
        pos: npt.NDArray[np.float64],
        mlh_color: str,
        highlights: _NodeHighlights,
    ) -> None:
        """Render a planar graph as hsv-colored, edge-oriented segments.

        When the displayed LM holds a current goal targeting this graph, its
        model-frame target location is marked as a labeled green dot, over faint
        green dots at the LM's earlier goal targets on this graph.

        Args:
            ax: The 2D Monty axis.
            mlh: The current most likely hypothesis.
            channel: The input channel whose graph is drawn.
            graph: The MLH graph object model.
            pos: The graph node positions, shape `(N, >=2)`.
            mlh_color: The MLH location marker color.
            highlights: The inhibited or spread-reached nodes to color.
        """
        goal_marker = self._goal_in_model_frame(mlh["graph_id"], channel)
        past_goals = self._past_goals(mlh["graph_id"], channel)
        overlay = self._second_hypothesis_overlay(mlh["graph_id"])
        ax.cla()
        x, y = pos[:, 0], pos[:, 1]
        fm = graph.feature_mapping
        graph_feature = {
            name: np.asarray(graph.get_values_for_feature(name))
            for name in ("hsv", "pose_fully_defined", "pose_vectors")
            if name in fm
        }
        colors, edge_mask, tangents = planar_style(
            len(x),
            graph_feature.get("hsv"),
            graph_feature.get("pose_fully_defined"),
            graph_feature.get("pose_vectors"),
        )
        draw_2d_segments(ax, x, y, colors, edge_mask, tangents)
        highlighted = highlights.draw(ax, pos, three_d=False)
        if overlay is not None:
            second_pts, second_label = overlay
            ax.plot(
                second_pts[:, 0],
                second_pts[:, 1],
                "o",
                color="lightskyblue",
                markersize=2,
                alpha=0.6,
                linestyle="none",
                label=second_label,
                zorder=2,
            )
        ax.plot(
            mlh["location"][0],
            mlh["location"][1],
            "x",
            color=mlh_color,
            markersize=8,
            markeredgewidth=2,
            zorder=3,
        )
        if len(past_goals):
            ax.plot(
                past_goals[:, 0],
                past_goals[:, 1],
                "o",
                color="limegreen",
                alpha=0.3,
                markersize=5,
                linestyle="none",
                label="past goals",
                zorder=3,
            )
        if goal_marker is not None:
            location, sender = goal_marker
            ax.plot(
                location[0],
                location[1],
                "o",
                color="green",
                markersize=7,
                label=f"goal ({sender})",
                zorder=6,
            )
        if (
            goal_marker is not None
            or len(past_goals)
            or overlay is not None
            or highlighted
        ):
            ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"MLH ({mlh['graph_id']})")


class DetailsPanel:
    """The Details column: per-channel buffer stacks, or inference line plots.

    Owns the Details column's single physical region and its mode state machine: a
    per-channel buffer grid during exploratory steps, the displayed LM's evidence and
    number-of-hypotheses line plots during matching steps, or a centered placeholder.
    Axes are rebuilt only when the active mode or channel count changes, to avoid
    flicker. The selection and per-channel features come from a `ChannelView` and the
    accumulated line-plot series from an `EvidenceHistory`. The "Input Feature" inset
    stacked over each buffer plot is a `FeatureInset`, the same widget the Monty section
    uses.
    """

    def __init__(
        self,
        fig: Figure,
        spec,
        channel_view: ChannelView,
        history: EvidenceHistory,
        show_num_hypotheses: bool = True,
    ) -> None:
        """Bind the Details column to its figure region and data sources.

        Args:
            fig: The figure to draw on.
            spec: The Details column's gridspec subplot spec.
            channel_view: The selected LM/channel and per-channel feature accessors.
            history: The per-LM evidence and hypothesis-count history.
            show_num_hypotheses: Whether the inference view includes the
                number-of-hypotheses line plot below the evidence plot (dropped by the
                attention layout to free vertical space).
        """
        self.fig = fig
        self.spec = spec
        self.channel_view = channel_view
        self.history = history
        self.show_num_hypotheses = show_num_hypotheses
        self._insets: dict[str, FeatureInset] = {}
        self._axes: list[Axes] = []
        self._proj_axes: list[list[Axes]] = []
        self._mode: str | None = None
        self._channel_count: int | None = None
        self._inference_legend = None
        self._evidence_steps: list[int] = []
        self._evidence_history: dict[str, list[float]] = {}
        self._num_hyp_history: dict[str, list[float]] = {}
        self._burst_steps: list[int] = []
        self._terminal_flags: list[bool] = []

    def draw_buffer_grid(
        self, channels: list[str], points: dict[str, npt.NDArray[np.float64]]
    ) -> None:
        """Draw one stacked plot group per channel's buffer (exploratory step).

        Args:
            channels: The channel ids with at least one location.
            points: The per-channel non-padded location points.
        """
        self._ensure_grid(len(channels))
        self._sync_insets(channels)
        for main_ax, proj_axes, channel in zip(self._axes, self._proj_axes, channels):
            groups = self.channel_view.channel_groups(channel, points[channel])
            draw_buffer_series(
                main_ax,
                proj_axes,
                groups,
                title=str(channel),
                title_fontsize=8,
                show_ticks=False,
            )
            cell = main_ax.get_position(self.fig)
            height = (cell.y1 - cell.y0) * 0.45
            rect = corner_rect(cell, width_frac=0.4, height=height)
            self._insets[channel].draw(channel, rect)

    def draw_inference(self) -> None:
        """Draw the displayed LM's evidence (and optional hypothesis-count) plots."""
        self._select_active_history()
        self._ensure_lines()
        self._draw_object_series(
            self._evidence_ax,
            self._evidence_history,
            "Highest evidence per object",
            "evidence",
        )
        self._shade_terminal_steps(self._evidence_ax)
        if self.show_num_hypotheses:
            self._draw_object_series(
                self._num_hyp_ax,
                self._num_hyp_history,
                "Number of hypotheses per object",
                "hypotheses",
            )
        self._draw_inference_legend()

    def draw_placeholder(self, message: str) -> None:
        """Draw a centered placeholder message spanning the Details column.

        Args:
            message: The text to display.
        """
        if self._mode != "placeholder":
            self._clear_insets()
            self._clear_axes()
            self._axes = [self.fig.add_subplot(self.spec)]
            self._mode = "placeholder"
            self._channel_count = None

        ax = self._axes[0]
        ax.cla()
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=ax.transAxes,
        )

    def _clear_axes(self) -> None:
        """Remove every main and projection axis in the Details column."""
        for ax in self._axes:
            ax.remove()
        for axes in self._proj_axes:
            for ax in axes:
                ax.remove()
        self._axes = []
        self._proj_axes = []

    def _sync_insets(self, channels: list[str]) -> None:
        """Match the feature insets to the channels currently stacked in the column.

        Drops insets for channels no longer shown and creates one per new channel, so
        each stacked Details panel carries the same "Input Feature" inset as the Monty
        section.

        Args:
            channels: The channel ids stacked in the Details column this frame.
        """
        wanted = set(channels)
        for channel in list(self._insets):
            if channel not in wanted:
                self._insets.pop(channel).remove()
        for channel in channels:
            if channel not in self._insets:
                self._insets[channel] = FeatureInset(self.fig, self.channel_view)

    def _clear_insets(self) -> None:
        """Remove every feature inset (when leaving the per-channel layout)."""
        for inset in self._insets.values():
            inset.remove()
        self._insets = {}

    def _ensure_grid(self, n: int) -> None:
        """Rebuild the column as `n` stacked per-channel plot groups.

        Each channel group is a 3D cloud over a row of three 2D projections. Rebuilds
        only when the channel count changes, to avoid flicker.

        Args:
            n: The number of stacked per-channel plot groups.
        """
        if self._mode == "grid" and self._channel_count == n:
            return
        self._clear_axes()
        outer = GridSpecFromSubplotSpec(n, 1, subplot_spec=self.spec, hspace=0.25)
        for i in range(n):
            inner = GridSpecFromSubplotSpec(
                2,
                3,
                subplot_spec=outer[i, 0],
                height_ratios=[3, 1],
                hspace=0.1,
                wspace=0.15,
            )
            self._axes.append(self.fig.add_subplot(inner[0, :], projection="3d"))
            self._proj_axes.append(
                [self.fig.add_subplot(inner[1, j]) for j in range(3)]
            )
        self._mode = "grid"
        self._channel_count = n

    def _ensure_lines(self) -> None:
        """Rebuild the column as the line plot(s) with the legend below/between them."""
        if self._mode == "lines":
            return
        self._clear_insets()
        self._clear_axes()
        if self.show_num_hypotheses:
            grid = GridSpecFromSubplotSpec(
                3,
                1,
                subplot_spec=self.spec,
                height_ratios=[1, 0.5, 1],
                hspace=0.4,
            )
            self._evidence_ax = self.fig.add_subplot(grid[0, 0])
            self._legend_ax = self.fig.add_subplot(grid[1, 0])
            self._num_hyp_ax = self.fig.add_subplot(grid[2, 0])
            self._axes = [self._evidence_ax, self._legend_ax, self._num_hyp_ax]
        else:
            grid = GridSpecFromSubplotSpec(
                2,
                1,
                subplot_spec=self.spec,
                height_ratios=[1, 0.45],
                hspace=0.5,
            )
            self._evidence_ax = self.fig.add_subplot(grid[0, 0])
            self._legend_ax = self.fig.add_subplot(grid[1, 0])
            self._num_hyp_ax = None
            self._axes = [self._evidence_ax, self._legend_ax]
        self._legend_ax.set_axis_off()
        self._mode = "lines"
        self._channel_count = None

    def _select_active_history(self) -> None:
        """Point the line-plot history fields at the displayed LM's accumulated data."""
        lm_id = self.channel_view.lm.learning_module_id
        self._evidence_steps = self.history.steps_by_lm.setdefault(lm_id, [])
        self._evidence_history = self.history.evidence_by_lm.setdefault(lm_id, {})
        self._num_hyp_history = self.history.num_hyp_by_lm.setdefault(lm_id, {})
        self._burst_steps = self.history.burst_steps_by_lm.setdefault(lm_id, [])
        self._terminal_flags = self.history.terminal_by_lm.setdefault(lm_id, [])

    def _shade_terminal_steps(self, ax: Axes) -> None:
        """Shade the steps at which the displayed LM is in its "match" terminal state.

        Each contiguous run of matching steps is filled as one vertical band spanning
        half a step either side of the run, so a single matching step is still
        visible. A run ends where the LM drops out of its terminal state, leaving the
        non-matching steps unshaded.

        Args:
            ax: The axis to shade.
        """
        run_start = None
        labeled = False
        for i, is_terminal in enumerate([*self._terminal_flags, False]):
            if is_terminal and run_start is None:
                run_start = i
            elif not is_terminal and run_start is not None:
                ax.axvspan(
                    self._evidence_steps[run_start] - 0.5,
                    self._evidence_steps[i - 1] + 0.5,
                    color="blue",
                    alpha=0.15,
                    linewidth=0,
                    zorder=0,
                    label=None if labeled else "terminal condition met",
                )
                labeled = True
                run_start = None

    def _draw_object_series(
        self,
        ax: Axes,
        history: dict[str, list[float]],
        title: str,
        ylabel: str,
    ) -> None:
        """Redraw one per-object line plot with burst markers.

        Iterating `self._evidence_history` key order in both plots keeps each object's
        color consistent across them. The shared legend is drawn separately beneath the
        MLH panel by `_draw_inference_legend`.

        Args:
            ax: The axis to redraw.
            history: The per-object value history to plot.
            title: The plot title.
            ylabel: The y-axis label.
        """
        ax.cla()
        for graph_id in self._evidence_history:
            ax.plot(self._evidence_steps, history[graph_id], label=graph_id)
        for i, burst_step in enumerate(self._burst_steps):
            ax.axvline(
                burst_step,
                linestyle="--",
                color="red",
                label="burst" if i == 0 else None,
            )
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)

    def _draw_inference_legend(self) -> None:
        """Place the shared per-object legend beneath the evidence plot.

        The legend would otherwise run off the figure's right edge, so it is drawn in a
        dedicated axis below the evidence plot (and above the hypotheses plot when that
        is shown), with the handles gathered from the evidence plot.
        """
        if self._inference_legend is not None:
            self._inference_legend.remove()
            self._inference_legend = None
        handles, labels = self._evidence_ax.get_legend_handles_labels()
        if not handles:
            return
        self._inference_legend = self._legend_ax.legend(
            handles,
            labels,
            loc="center",
            ncol=2,
            fontsize=10,
            borderaxespad=0.0,
        )


class AttentionPanel:
    """The attention section: the `AttentionSystem`'s live voxel grid in 3D space.

    Draws the attention system's persistent sparse voxel grid as a 3D scatter of voxel
    centers in world coordinates, colored by each voxel's attention weight on a
    diverging scale: red for excitation, blue for inhibition, fading to white as a
    weight decays toward zero (and expires). The axis frame is the union of every
    extent seen so far in the episode, so the view stays stable as attention moves.

    This step's proposed goals are overlaid at their world locations, grouped by
    source (the proposing SM or LM id, shown in the legend) and styled by whether
    each fell within the active attention space (stars for goals the attention filter
    kept, crosses for goals it dropped). SM salience goals are dense (one per
    on-object location), so only each SM's top 5% by confidence (its salience) are
    shown — labeled "top ... goals" — drawn small, translucent, and hot pink to avoid
    washing out the voxel weights beneath; the sparse LM goals are all shown, larger
    and opaque (green when kept, red when dropped). The goal the policy selector
    chose this step is drawn as a larger gold star — labeled "enacted goal" when
    monitoring (the actions always execute) or "proposed goal" when interactive (the
    same goal the jump button would enact, which the user may decline). The frame is
    widened to enclose the goals, so a goal outside the attended region stays
    visible.

    When the most recent step's proposed grid carried the inhibit-all signal (e.g. an
    `InhibitAllOnRecognition` region proposer fired under the `InhibitionFlipsGrid`
    merge), a warning is drawn over the panel on that step.

    The world frame is y-up with the camera looking roughly along the z axis, so the
    grid is drawn with world x to the right, world y vertical, and world z as scene
    depth, viewed nearly head-on to match how the object's surface faces the camera.
    That head-on view is only the starting point: dragging to rotate persists across
    steps, and the mouse wheel zooms in and out around the frame center (also
    persisted, and reset with each new episode's figure).

    Shows a placeholder when the model has no real attention system (e.g. the
    `NoopAttentionSystem`, which carries no voxel grid) or the grid is empty.
    """

    # The fraction of an SM's proposed goals shown, keeping only its most salient
    # ones so the dense per-location salience goals don't clutter the voxel grid.
    TOP_SM_GOAL_FRACTION: ClassVar[float] = 0.05

    # How much the projected 3D scene is enlarged within its axes box, eating into
    # the wide internal margins a matplotlib 3D axis reserves around its cube.
    SCENE_ZOOM: ClassVar[float] = 1.3

    # Mouse-wheel zoom: the per-notch scale factor and the allowed zoom range.
    WHEEL_ZOOM_STEP: ClassVar[float] = 1.2
    WHEEL_ZOOM_RANGE: ClassVar[tuple[float, float]] = (0.2, 25.0)

    def __init__(self, fig: Figure, spec, *, interactive: bool = False) -> None:
        """Lay out the 3D voxel axis and its weight colorbar.

        Args:
            fig: The figure to draw on.
            spec: The attention panel's gridspec subplot spec.
            interactive: Whether the plotter is interactive, in which case the
                selected goal is a proposal the user may decline rather than an
                action that will certainly execute, and is labeled accordingly.
        """
        self.fig = fig
        self._selected_goal_label = "proposed goal" if interactive else "enacted goal"
        self._ax = fig.add_subplot(spec, projection="3d")
        # A 3D axis renders its (equal-aspect) scene as a roughly square block centered
        # in its cell (enlarged by SCENE_ZOOM), so the colorbar is pinned just right of
        # that square rather than at the far cell edge.
        cell = spec.get_position(fig)
        fig_w, fig_h = fig.get_size_inches()
        cell_w = cell.x1 - cell.x0
        cell_h = cell.y1 - cell.y0
        side_frac_x = min(cell_w * fig_w, cell_h * fig_h) / fig_w * self.SCENE_ZOOM
        cax_left = (cell.x0 + cell.x1) / 2 + side_frac_x / 2 + 0.02
        cax_height = cell_h * 0.65
        cax_bottom = cell.y0 + (cell_h - cax_height) / 2
        self._cax = fig.add_axes([cax_left, cax_bottom, 0.008, cax_height])
        # Weights live in [MIN_ATTENTION_WEIGHT, MAX_ATTENTION_WEIGHT] (= [-1, 1]):
        # positive excitation decays toward zero, negative weights inhibit. A fixed
        # diverging norm keeps colors comparable across steps and the colorbar static,
        # with red excitation and blue inhibition meeting at (near-white) zero.
        self._norm = plt.Normalize(-1.0, 1.0)
        self._cmap = plt.get_cmap("managua")

        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=self._norm, cmap=self._cmap), cax=self._cax
        )
        self._cax.set_ylabel("attention weight", fontsize=8)
        self._cax.tick_params(labelsize=7)
        self._bounds: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = (
            None
        )
        # The mouse wheel zooms the view by shrinking or widening the data limits
        # around the frame center; the level persists across steps and redraws.
        self._wheel_zoom = 1.0
        self._frame_state: tuple[npt.NDArray[np.float64], float] | None = None
        self._view_initialized = False
        fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    def draw(self, model: Monty) -> None:
        """Draw the current voxel grid, or a placeholder when there is none.

        Args:
            model: The Monty model whose attention system is read.
        """
        attention = getattr(model, "attention_system", None)
        grid = getattr(attention, "grid", None)
        if grid is None:
            self._draw_placeholder("no attention system")
            return
        data = grid.to_pandas()
        if len(data) == 0:
            self._draw_placeholder("voxel grid empty")
            return

        # Voxel indices are the lower corners in units of voxel_size; +0.5 centers.
        voxels = data.index.to_frame(index=False).to_numpy(dtype=float)
        centers = (voxels + 0.5) * grid.voxel_size
        weights = data["weight"].to_numpy(dtype=float)

        ax = self._ax
        ax.cla()
        ax.set_axis_on()
        # Plot axes are (world x, world z, world y) so world-up is vertical and world z
        # is depth; the near-head-on view then matches the camera's perspective.
        # Translucent so the goal markers overlaid on the grid stay visible.
        ax.scatter(
            centers[:, 0],
            centers[:, 2],
            centers[:, 1],
            c=self._cmap(self._norm(weights)),
            marker="s",
            s=16,
            alpha=0.7,
            depthshade=False,
        )
        ax.set_title(f"Attention voxel grid ({len(data)} voxels)")
        if self._inhibit_all_proposed(attention):
            ax.text2D(
                0.5,
                0.97,
                "inhibit-all proposed: grid flipped to inhibition",
                color="red",
                fontweight="bold",
                fontsize=9,
                ha="center",
                va="top",
                transform=ax.transAxes,
            )

        goal_locations = self._draw_goals(ax, model)
        self._frame_state = self._frame(centers, goal_locations)
        self._apply_frame()
        if not self._view_initialized:
            # The near-head-on default view, set only once: `cla` preserves the
            # viewing angles, so the user's mouse rotation survives later redraws.
            ax.view_init(elev=8, azim=-82)
            self._view_initialized = True
        ax.tick_params(labelsize=6)
        # The depth (world z) axis is nearly edge-on in the head-on view, so its tick
        # labels would render as unreadable overlapping text.
        ax.set_yticks([])

    def _apply_frame(self) -> None:
        """Apply the episode frame to the axis at the current wheel-zoom level.

        The frame's half side length is divided by the zoom level, so zooming in
        narrows the limits around the frame center (and zooming out widens them)
        while the scene keeps filling the same enlarged box in the figure.
        """
        center, half = self._frame_state
        half /= self._wheel_zoom
        ax = self._ax
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[2] - half, center[2] + half)
        ax.set_zlim(center[1] - half, center[1] + half)
        ax.set_box_aspect((1, 1, 1), zoom=self.SCENE_ZOOM)

    def _on_scroll(self, event: MouseEvent) -> None:
        """Zoom the voxel view in or out on mouse wheel, around the frame center.

        Only reacts while the pointer is over the attention axis and a frame has
        been drawn. The zoom level persists across steps (each `draw` re-applies
        it) and resets with each new episode's figure.

        Args:
            event: The matplotlib scroll event; its `step` is positive to zoom in.
        """
        if event.inaxes is not self._ax or self._frame_state is None:
            return
        low, high = self.WHEEL_ZOOM_RANGE
        self._wheel_zoom = float(
            np.clip(self._wheel_zoom * self.WHEEL_ZOOM_STEP**event.step, low, high)
        )
        self._apply_frame()
        self.fig.canvas.draw_idle()

    def _draw_goals(self, ax: Axes, model: Monty) -> npt.NDArray[np.float64]:
        """Overlay this step's proposed goals on the voxel grid.

        Goals are grouped by their proposing module and by whether they fell within
        the active attention space, one scatter (and legend entry) per group: stars
        for goals the attention filter kept, crosses for goals it dropped. Dense SM
        salience goals are thinned to each SM's most salient few (see
        `_top_sm_goals`) and drawn small, translucent, and hot pink so the voxel
        weights stay readable beneath them; sparse LM (GSG) goals are all drawn,
        larger and opaque, green when kept and red when dropped. The goal the policy
        selector chose this step is drawn separately as a larger gold star, labeled
        "enacted goal" or "proposed goal" per the plotter's interactivity. Sources
        are named by the goal's sender id, so the legend shows which SM or LM
        proposed each group.

        Args:
            ax: The 3D voxel axis, using (world x, world z, world y) plot axes.
            model: The Monty model whose goals are read.

        Returns:
            The `(G, 3)` world locations of the drawn goals, for widening the frame;
            empty `(0, 3)` when there are no located goals.
        """
        goals = [g for g in proposed_goals(model) if g.location is not None]
        if not goals:
            return np.empty((0, 3))
        goals = self._top_sm_goals(goals)
        enacted = enacted_goal(model)

        groups: dict[tuple[str, str, bool], list[npt.NDArray[np.float64]]] = {}
        for goal in goals:
            if goal is enacted:
                continue
            key = (
                str(goal.sender_id),
                str(goal.sender_type),
                passed_attention_filter(goal),
            )
            groups.setdefault(key, []).append(np.asarray(goal.location, dtype=float))

        for (sender, sender_type, passed), locations in sorted(groups.items()):
            pts = np.asarray(locations)
            style = self._goal_style(sender_type, passed=passed)
            status = "in attention" if passed else "filtered out"
            name = f"top {sender} goals" if sender_type == "SM" else f"{sender} goals"
            ax.scatter(
                pts[:, 0],
                pts[:, 2],
                pts[:, 1],
                label=f"{name} ({status})",
                depthshade=False,
                **style,
            )

        drawn = [np.asarray(g.location, dtype=float) for g in goals]
        # if enacted is not None and enacted.location is not None:
        #     loc = np.asarray(enacted.location, dtype=float)
        #     drawn.append(loc)
        #     ax.scatter(
        #         [loc[0]],
        #         [loc[2]],
        #         [loc[1]],
        #         marker="*",
        #         s=150,
        #         color="gold",
        #         edgecolors="black",
        #         linewidths=0.8,
        #         label=f"{self._selected_goal_label} ({enacted.sender_id})",
        #         depthshade=False,
        #     )
        ax.legend(fontsize=6, loc="upper left")
        return np.asarray(drawn)

    @classmethod
    def _top_sm_goals(cls, goals: list[Goal]) -> list[Goal]:
        """Keep every LM goal but only each SM's most salient few.

        An SM proposes one goal per on-object location with its salience as the
        confidence, so drawing them all buries the voxel grid. Each SM's goals are
        thinned to its top `TOP_SM_GOAL_FRACTION` by confidence (at least one), which
        are also the ones the motor system would choose between. LM goals are rare
        and always kept.

        Args:
            goals: This step's located proposed goals.

        Returns:
            The goals to draw, with each SM's list reduced to its most salient few.
        """
        kept = [g for g in goals if g.sender_type != "SM"]
        by_sender: dict[str, list[Goal]] = {}
        for goal in goals:
            if goal.sender_type == "SM":
                by_sender.setdefault(str(goal.sender_id), []).append(goal)
        for sender_goals in by_sender.values():
            count = max(1, math.ceil(len(sender_goals) * cls.TOP_SM_GOAL_FRACTION))
            ranked = sorted(sender_goals, key=lambda g: g.confidence, reverse=True)
            kept.extend(ranked[:count])
        return kept

    @staticmethod
    def _goal_style(sender_type: str, *, passed: bool) -> dict:
        """Resolve the scatter style for a group of goal markers.

        An SM proposes one salience goal per on-object location, so its markers are
        small, translucent, and hot pink (a color outside the voxel colormap) to
        avoid washing out the attention weights beneath. LM (GSG) goals are rare, so
        they stay larger and opaque: green when kept, red when dropped. Kept goals
        are stars; dropped goals are crosses.

        Args:
            sender_type: The goal's sender type (`"SM"` or `"GSG"`).
            passed: Whether the goal fell within the active attention space.

        Returns:
            The scatter keyword arguments for the group.
        """
        if sender_type == "SM":
            marker = (
                {"marker": "*", "s": 4, "color": "green"}
                if passed
                else {"marker": "x", "s": 2, "color": "hotpink"}
            )
            return {**marker, "alpha": 0.5, "linewidths": 0.5}
        if passed:
            return {"marker": "*", "s": 35, "color": "green"}
        return {"marker": "x", "s": 30, "color": "red", "linewidths": 1.5}

    @staticmethod
    def _inhibit_all_proposed(attention: object) -> bool:
        """Whether this step's proposed grid carried the inhibit-all signal.

        The persistent grid never carries the signal itself; it rides on the per-step
        proposal recorded in the attention system's telemetry, so the last proposed
        grid flags exactly the step a flip takes place.

        Args:
            attention: The model's attention system.

        Returns:
            True when the most recent proposed grid signals inhibit-all.
        """
        proposed = attention.state_dict().get("proposed_grids", [])
        return bool(proposed) and bool(getattr(proposed[-1], "inhibit_all", False))

    def _frame(
        self,
        centers: npt.NDArray[np.float64],
        extra: npt.NDArray[np.float64] | None = None,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Return a stable cubic frame enclosing every voxel seen this episode.

        The bounds only ever grow, so the view doesn't jump around as voxels decay and
        expire.

        Args:
            centers: The `(V, 3)` world-frame voxel centers drawn this step.
            extra: Additional `(G, 3)` world points the frame must also enclose
                (e.g. goal locations outside the attended region), or `None`/empty.

        Returns:
            The frame's per-axis center and half side length.
        """
        if extra is not None and len(extra):
            centers = np.concatenate([centers, extra])
        low = centers.min(axis=0)
        high = centers.max(axis=0)
        if self._bounds is None:
            self._bounds = (low, high)
        else:
            self._bounds = (
                np.minimum(self._bounds[0], low),
                np.maximum(self._bounds[1], high),
            )
        low, high = self._bounds
        center = (low + high) / 2
        half = max(float((high - low).max()) / 2, 0.01)
        return center, half

    def _draw_placeholder(self, message: str) -> None:
        """Show a centered message in place of the voxel scatter.

        Args:
            message: The text to display.
        """
        ax = self._ax
        ax.cla()
        ax.set_axis_off()
        ax.text2D(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)


class SaliencePanel:
    """The salience section: the salience map and the effect of inhibition of return.

    Finds the sensor module carrying a salience strategy (e.g. `Vocus2` on a
    `SalienceSM`) and reads the 2D maps the module recorded in its telemetry for the
    most recent step, drawn side by side as heatmaps with the fixation point (image
    center) marked:

    - Salience: the salience map the strategy extracted from the whole view. Maps
      already normalized to `[0, 1]` (the strategies' default) keep a fixed color
      scale so frames are comparable; any other range is scaled per frame.
    - Inhibition of return: the IoR weight in `[0, 1]` at each on-object pixel, i.e.
      how strongly the decaying kernels left at earlier fixations suppress it.
    - After IoR: the on-object salience minus the IoR weights scaled by the module's
      `ior_weight`, which is what the module turns into goal confidences (before
      adding noise and range-normalizing). It shares the salience's `[0, 1]` scale,
      so fixated spots go dark; values pushed below zero are clipped (shown by the
      colorbar's extension). The inhibited region (IoR weight above
      `IOR_CONTOUR_LEVEL`) is outlined so small inhibited spots stay findable.

    Off-object pixels carry no IoR (goals are only proposed on the object), so they
    are drawn in a neutral gray in the IoR panels.

    The sensor module only records when configured with `save_raw_obs=true` (which
    installs a recording `SalienceSMTelemetry` rather than the no-op one, detected here
    by the telemetry carrying the recorded list), so the panel explains that
    requirement instead of drawing when it is off. It likewise shows a placeholder when
    no sensor module has a salience strategy, and in the IoR panels when the
    telemetry predates IoR recording.
    """

    CMAP: ClassVar[str] = "inferno"
    IOR_CMAP: ClassVar[str] = "Blues"
    OFF_OBJECT_COLOR: ClassVar[str] = "0.5"
    IOR_CONTOUR_LEVEL: ClassVar[float] = 0.05

    def __init__(self, fig: Figure, spec, model: Monty) -> None:
        """Bind the panel to its axes and resolve the salience sensor module.

        Args:
            fig: The figure to draw on.
            spec: The salience panel's gridspec subplot spec.
            model: The Monty model whose sensor modules are searched for a salience
                strategy.
        """
        self.fig = fig
        grid = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.2)
        self._salience_ax, self._ior_ax, self._inhibited_ax = (
            fig.add_subplot(grid[0, i]) for i in range(3)
        )
        # Divider-managed colorbar axes hug the (square, equal-aspect) heatmaps
        # rather than the cell edges, and are created once so redraws don't
        # accumulate colorbars. They sit below the heatmaps so the three maps keep
        # the full width of the narrow cell.
        self._caxes = {
            ax: make_axes_locatable(ax).append_axes("bottom", size="6%", pad=0.05)
            for ax in (self._salience_ax, self._ior_ax, self._inhibited_ax)
        }
        self._sm = next(
            (
                sm
                for sm in model.sensor_modules
                if getattr(sm, "_salience_strategy", None) is not None
            ),
            None,
        )

    def draw(self) -> None:
        """Draw the salience and IoR heatmaps recorded for the most recent step."""
        for ax, cax in self._caxes.items():
            ax.cla()
            ax.set_axis_off()
            cax.cla()
            cax.set_axis_off()
        if self._sm is None:
            self._draw_placeholder(self._salience_ax, "no salience strategy configured")
            return
        telemetry = self._sm._snapshot_telemetry
        maps = getattr(telemetry, "salience_maps", None)
        if maps is None:
            # The no-op telemetry carries no recorded lists at all.
            self._draw_placeholder(
                self._salience_ax,
                "Salience view disabled:\n"
                f"set save_raw_obs=true on sensor module "
                f"{self._sm.sensor_module_id!r}\nso it records its salience maps",
            )
            return
        if not maps:
            self._draw_placeholder(self._salience_ax, "no salience map recorded yet")
            return

        self._draw_salience(np.asarray(maps[-1], dtype=float))

        ior_maps = getattr(telemetry, "ior_maps", None)
        inhibited_maps = getattr(telemetry, "inhibited_salience_maps", None)
        if not ior_maps or not inhibited_maps:
            for ax in (self._ior_ax, self._inhibited_ax):
                self._draw_placeholder(ax, "no IoR recorded")
            return
        ior = np.asarray(ior_maps[-1], dtype=float)
        self._draw_ior(ior)
        self._draw_inhibited(np.asarray(inhibited_maps[-1], dtype=float), ior)

    def _draw_salience(self, salience: npt.NDArray[np.float64]) -> None:
        """Draw the raw salience map.

        Args:
            salience: The 2D salience map.
        """
        ax = self._salience_ax
        strategy_name = type(self._sm._salience_strategy).__name__
        ax.set_title(
            f"Salience\n({strategy_name} on {self._sm.sensor_module_id})", fontsize=8
        )
        # The strategies range-normalize to [0, 1] by default; keep that scale fixed
        # across frames so the same color means the same salience from step to step.
        finite = salience[np.isfinite(salience)]
        if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None
        image = ax.imshow(salience, cmap=self.CMAP, vmin=vmin, vmax=vmax)
        self._draw_colorbar(ax, image)
        self._mark_fixation(ax, salience.shape, "cyan")

    def _draw_ior(self, ior: npt.NDArray[np.float64]) -> None:
        """Draw the inhibition-of-return weights.

        Args:
            ior: The 2D IoR weights, NaN off-object.
        """
        ax = self._ior_ax
        finite = ior[np.isfinite(ior)]
        peak = float(finite.max()) if finite.size else 0.0
        ax.set_title(f"Inhibition of return\n(max {peak:.2f})", fontsize=8)
        image = ax.imshow(
            ior, cmap=self._with_off_object(self.IOR_CMAP), vmin=0, vmax=1
        )
        self._draw_colorbar(ax, image)
        self._mark_fixation(ax, ior.shape, "red")

    def _draw_inhibited(
        self, inhibited: npt.NDArray[np.float64], ior: npt.NDArray[np.float64]
    ) -> None:
        """Draw the on-object salience left after inhibition of return.

        Args:
            inhibited: The 2D inhibited salience, NaN off-object.
            ior: The 2D IoR weights, NaN off-object, outlined over the map.
        """
        ax = self._inhibited_ax
        ior_weight = getattr(self._sm, "_ior_weight", None)
        suffix = f"\n(ior_weight {ior_weight:g})" if ior_weight is not None else ""
        ax.set_title(f"Salience after IoR{suffix}", fontsize=8)
        image = ax.imshow(
            inhibited, cmap=self._with_off_object(self.CMAP), vmin=0, vmax=1
        )
        self._draw_colorbar(ax, image, extend="min")
        inhibited_region = np.nan_to_num(ior, nan=0.0)
        if inhibited_region.max() > self.IOR_CONTOUR_LEVEL:
            ax.contour(
                inhibited_region,
                levels=[self.IOR_CONTOUR_LEVEL],
                colors="deepskyblue",
                linewidths=0.8,
            )
        self._mark_fixation(ax, inhibited.shape, "cyan")

    def _with_off_object(self, name: str) -> Colormap:
        """A copy of a colormap that draws NaN (off-object) pixels in gray.

        Args:
            name: The colormap name.

        Returns:
            The colormap with its "bad" color set to `OFF_OBJECT_COLOR`.
        """
        cmap = plt.get_cmap(name).copy()
        cmap.set_bad(self.OFF_OBJECT_COLOR)
        return cmap

    def _draw_colorbar(
        self, ax: Axes, image: AxesImage, extend: str = "neither"
    ) -> None:
        """Draw a heatmap's horizontal colorbar in its divider axis below it.

        Args:
            ax: The heatmap axis.
            image: The heatmap.
            extend: Which colorbar ends to extend, marking clipped values.
        """
        cax = self._caxes[ax]
        cax.set_axis_on()
        self.fig.colorbar(image, cax=cax, orientation="horizontal", extend=extend)
        # Few ticks, so neighboring colorbars' end labels don't run together.
        cax.xaxis.set_major_locator(MaxNLocator(nbins=2))
        cax.tick_params(labelsize=6)

    @staticmethod
    def _mark_fixation(ax: Axes, shape: tuple[int, ...], color: str) -> None:
        """Mark the fixation point; the salience sensor module fixates the center.

        Args:
            ax: The heatmap axis.
            shape: The heatmap's shape.
            color: The marker color.
        """
        h, w = shape[:2]
        ax.plot(w // 2, h // 2, "+", color=color, markersize=8, markeredgewidth=1.5)

    @staticmethod
    def _draw_placeholder(ax: Axes, message: str) -> None:
        """Show a centered message in place of a heatmap.

        Args:
            ax: The heatmap axis to write in.
            message: The text to display.
        """
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            fontsize=8,
            transform=ax.transAxes,
        )


class SegmentationPanel:
    """The segmentation section: the model-free SM's segmented region over its view.

    Finds the sensor module carrying a segmentation strategy (e.g. `SlicMerge` on a
    `SalienceSM`) and reads the 2D segmentation mask and matching camera snapshot the
    module recorded in its telemetry, so the overlay shows exactly the mask Monty acted
    on. The panel shows the camera image with everything outside the segmented region
    dimmed, the region boundary outlined, and the fixation point (image center) marked.

    The sensor module only records when configured with `save_raw_obs=true` (which
    installs a recording `SalienceSMTelemetry` rather than the no-op one, detected here
    by the telemetry carrying the recorded lists), so the panel explains that
    requirement instead of drawing when it is off. It likewise shows a placeholder when
    no sensor module has a segmentation strategy.
    """

    def __init__(self, fig: Figure, spec, model: Monty) -> None:
        """Bind the panel to its axis and resolve the segmenting sensor module.

        Args:
            fig: The figure to draw on.
            spec: The segmentation panel's gridspec subplot spec.
            model: The Monty model whose sensor modules are searched for a
                segmentation strategy.
        """
        self.fig = fig
        self._ax = fig.add_subplot(spec)
        self._sm = next(
            (
                sm
                for sm in model.sensor_modules
                if getattr(sm, "_segmentation_strategy", None) is not None
            ),
            None,
        )

    def draw(self) -> None:
        """Draw the segmented-region overlay recorded for the most recent step."""
        ax = self._ax
        ax.cla()
        ax.set_axis_off()
        if self._sm is None:
            self._draw_placeholder("no segmentation strategy configured")
            return
        telemetry = self._sm._snapshot_telemetry
        maps = getattr(telemetry, "segmentation_maps", None)
        raw_observations = getattr(telemetry, "raw_observations", None)
        if maps is None or raw_observations is None:
            # The no-op telemetry carries no recorded lists at all.
            self._draw_placeholder(
                "Segmentation view disabled:\n"
                f"set save_raw_obs=true on sensor module "
                f"{self._sm.sensor_module_id!r}\nso it records its segmentation maps"
            )
            return
        if not maps or not raw_observations:
            self._draw_placeholder("no segmentation recorded yet")
            return

        mask = maps[-1]
        rgba = np.asarray(raw_observations[-1]["rgba"])
        strategy_name = type(self._sm._segmentation_strategy).__name__
        ax.set_title(
            f"Segmented region ({strategy_name} on {self._sm.sensor_module_id})"
        )
        ax.imshow(rgba)
        if mask is not None:
            dim = np.zeros((*mask.shape, 4))
            dim[mask == 0] = (0.0, 0.0, 0.0, 0.55)
            ax.imshow(dim)
            ax.contour(mask, levels=[0.5], colors="cyan", linewidths=1.2)
        # The segmentation strategies fixate at the image center.
        h, w = rgba.shape[:2]
        ax.plot(w // 2, h // 2, "+", color="red", markersize=10, markeredgewidth=2)

    def _draw_placeholder(self, message: str) -> None:
        """Show a centered message in place of the overlay.

        Args:
            message: The text to display.
        """
        self._ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            wrap=True,
            transform=self._ax.transAxes,
        )
