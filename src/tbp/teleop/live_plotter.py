# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

from typing import TYPE_CHECKING
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.no_reset_evidence_matching import (
    MontyForNoResetEvidenceGraphMatching,
)

from tbp.teleop.controls import (
    ActionButtons,
    SelectorBar,
    SpeedSlider,
)
from tbp.teleop.helpers import (
    ChannelView,
    EvidenceHistory,
    draw_section_dividers,
    is_interactive_backend,
)
from tbp.teleop.panels import (
    DetailsPanel,
    MontyPanel,
    SimulatorPanel,
    MemoryPanel
)
from tbp.teleop.plotter import Plotter
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    apply_rf_transform_to_points,
)

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        Monty,
        Observations,
    )


class LivePlotter(Plotter):
    """Live plotter implementing the `Plotter` Protocol for training and inference.

    Renders a 3-section view of the selected input channel of the displayed
    learning module. You can switch both the displayed learning module and the selected
    input channel at runtime with two cycling buttons under the `Step N` title. The
    learning module and channel are discovered from the model rather than configured by
    id.

    - Simulator: view finder and RGB patch of the sensor module feeding the selected
      (or, for an LM channel, the displayed LM's first sensor) channel.
    - Monty: the main graph of the selected channel, with an "Input Feature" corner
      inset showing the live feature on that channel (a 2D edge, a 3D surface, or, for a
      learning-module channel, the name of the object being passed). During an
      exploratory step the main graph is the channel's buffer points; during a
      matching step it is the most likely hypothesis graph plus its location marker.
    - Details: during an exploratory step, a per-channel stack of the other channels'
      buffers, each carrying the same "Input Feature" inset as the Monty section; during
      a matching step, the displayed LM's per-object evidence and number-of-hypotheses
      line plots.

    A non-interactive plotter exposes a `Speed` slider; an interactive plotter instead
    shows the four exploration heading buttons (plus a jump button when one is offered
    and "End episode") and overrides the executed action via `override_action`.

    When the displayed learning module lacks the evidence-LM inference API, the Monty
    and Details matching-step panels degrade to a placeholder rather than raising.
    """

    _channel_view: ChannelView
    _controls: ActionButtons | SpeedSlider | None
    _history: EvidenceHistory
    _last_observations: Observations | None
    _last_step: int | None
    _supervised_lm_ids: list[str]

    def __init__(
        self,
        interactive: bool = False,
        min_delay: float = 0.001,
        max_delay: float = 2.0,
        figsize: tuple[float, float] = (16, 8),
    ) -> None:
        """Initialize the plotter.

        Args:
            interactive: Whether to render action buttons and override the executed
                action with the user's choice (otherwise render a speed slider).
            min_delay: Non-interactive pause in seconds at full speed.
            max_delay: Maximum non-interactive pause in seconds at the slowest speed.
            figsize: Figure size in inches.
        """
        # Turn interactive plotting off so the plotter controls when figures are
        # drawn and when execution blocks, via its own canvas event loop.
        plt.ioff()

        self.interactive = interactive
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.figsize = figsize

        self.fig = None
        self._controls = None

        # Graph-memory objects on which this plotter has installed its optional
        # merge-animation callback.
        self._merge_hook_memories = []

    def _lm_building_graph(self, lm: LearningModule) -> bool:
        """Whether a learning module is building a graph this step.

        The model's `step_type` is shared across all learning modules, so it can't
        distinguish per-LM phases in a heterarchy. On an exploratory step every LM is
        building. On a matching step (the default, and what partially-supervised
        heterarchy training stays on) an LM is still building its graph when it is one
        of the supervised LMs in a training episode: its buffer is converted to a graph
        from the ground-truth label at episode end. Every other LM is matching.

        Args:
            lm: The learning module whose phase decides which panels to draw.

        Returns:
            True when `lm` is building a graph (exploratory panels), False when it is
            matching (inference panels).
        """
        if self.model.step_type == "exploratory_step":
            return True
        return (
            self.model.experiment_mode is ExperimentMode.TRAIN
            and lm.learning_module_id in self._supervised_lm_ids
        )

    @staticmethod
    def _is_continual(model: Monty) -> bool:
        """Whether the model carries its hypotheses across episodes.

        No-reset / continual experiments do not reset their hypotheses (and therefore
        their evidence) between episodes, so their evidence history is plotted as one
        continuous trajectory. Every other experiment resets per episode and its plot
        is cleared at each episode boundary.

        Args:
            model: The Monty model being plotted.

        Returns:
            True when the model preserves hypotheses across episodes.
        """
        return isinstance(model, MontyForNoResetEvidenceGraphMatching)

    def initialize(self, model: Monty, supervised_lm_ids: list[str]) -> None:
        """Resolve the displayed LM and build the figure, axes, and widgets.

        Detects whether the displayed learning module supports the inference and buffer
        panels and resets the per-episode history. When interactive, derives the action
        buttons from the model's interactive policy. Must be called once per episode.

        Args:
            model: The Monty model whose sensor and learning modules are plotted.
            supervised_lm_ids: The list of supervised learning module IDs.
        """
        self._remove_merge_animation_hooks()
        self.model = model
        self._channel_view = ChannelView(model)
        self._controls = (
            ActionButtons(model)
            if self.interactive
            else SpeedSlider(self.min_delay, self.max_delay)
        )
        self._history = EvidenceHistory()
        self._last_observations = None
        self._last_step = None
        self._supervised_lm_ids = supervised_lm_ids

        self._build_figure()
        self._install_merge_animation_hooks()

    def _build_figure(self) -> None:
        """Build this episode's figure, axes, and widgets.

        Closes the previous episode's figure first so figures don't accumulate across a
        multi-object / multi-rotation run.

        Raises:
            RuntimeError: When interactive but the active backend cannot run the
                blocking event loop (e.g. the headless `Agg` backend).
        """
        if self.interactive and not is_interactive_backend():
            raise RuntimeError(
                "Interactive plotter requires a GUI matplotlib backend (e.g. TkAgg / "
                f"QtAgg); the current backend {mpl.get_backend()!r} cannot run "
                "the blocking event loop."
            )

        if self.fig is not None:
            plt.close(self.fig)

        self.fig = plt.figure(figsize=self.figsize)
        self.fig.subplots_adjust(
            bottom=0.16, top=0.9, left=0.04, right=0.97, wspace=0.25
        )
        outer = self.fig.add_gridspec(1, 4)
        self._sim_spec = outer[0, 0]
        self._monty_spec = outer[0, 1]
        self._details_spec = outer[0, 2]
        # NEW: this assignment is the piece your traceback says is missing.
        self._memory_spec = outer[0, 3]

        self._simulator = SimulatorPanel(self.fig, self._sim_spec)
        self._monty = MontyPanel(self.fig, self._monty_spec, self._channel_view)
        self._details = DetailsPanel(
            self.fig,
            self._details_spec,
            self._channel_view,
            self._history,
        )

        # NEW: importantly, pass the SAME ChannelView as Monty/Details.
        #
        # That is what makes the Memory panel follow the existing LM/channel
        # selector without introducing another selector or callback.
        self._memory = MemoryPanel(
            self.fig,
            self._memory_spec,
            self._channel_view,
        )

        # CHANGED: include the fourth section when drawing separators.
        draw_section_dividers(
            self.fig,
            self._sim_spec,
            self._monty_spec,
            self._details_spec,
            self._memory_spec,
        )

        self._selector = SelectorBar(
            self.fig, self._monty_spec, self._channel_view, self._redraw
        )
        if self.interactive:
            self._controls.build(self.fig, self._simulator.ax_rgb)
        else:
            self._controls.build(self.fig)

        if is_interactive_backend():
            self.fig.show()

    def update(self, observations: Observations, step: int) -> None:
        """Draw the current state, never blocking for an action.

        The frame is stashed so selector-button clicks can repaint it without new
        observations. An interactive plotter draws no choice buttons here (they are
        built on demand in `override_action`); a non-interactive plotter applies its
        speed-slider pause/halt here. Interactive blocking lives only in
        `override_action`.

        Args:
            observations: The observations from the most recent step.
            step: The index of the current step within the episode.
        """
        if self.fig is None:
            return
        prev_step = self._last_step
        self._last_observations = observations
        self._last_step = step
        if self._is_continual(self.model):
            history_step = self.model.total_steps
        else:
            if prev_step is None or step <= prev_step:
                self._history.clear()
            history_step = step
        self._history.accumulate(self.model.learning_modules, history_step)
        self._render(observations, step)
        if not self.interactive:
            self._controls.pause()

    def _render(self, observations: Observations, step: int) -> None:
        """Draw every section for one frame.

        Args:
            observations: The observations to draw.
            step: The index of the current step within the episode.
        """
        self.fig.suptitle(f"Step {step}")
        if self._channel_view.ensure_channel():
            self._selector.refresh_labels()

        sm, sm_id = self._channel_view.simulator_sm()
        self._simulator.draw(observations, sm, sm_id)

        if self._lm_building_graph(self._channel_view.lm):
            self._draw_training()
        else:
            self._draw_inference()

        self._monty.draw_feature_inset()

        # NEW:
        #
        # Memory is independent of whether this particular step is exploratory
        # or matching, so do NOT add it separately to _draw_training() and
        # _draw_inference().
        #
        # Putting it here means:
        #
        #   * it appears during both training and inference,
        #   * it refreshes every experiment step,
        #   * it refreshes after the existing LM/channel selector changes,
        #   * there is only one new rendering call to maintain.
        self._memory.draw()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _redraw(self) -> None:
        """Repaint the last frame after a selection change, without re-accumulating.

        Called by the selector buttons so a new learning module or channel is visible
        immediately, even while a blocking event loop is running.
        """
        if self.fig is None or self._last_observations is None:
            return
        self._render(self._last_observations, self._last_step)

    def _install_merge_animation_hooks(self) -> None:
        """Attach this plotter to graph memories that support merge telemetry."""
        self._merge_hook_memories = []

        for lm in self.model.learning_modules:
            memory = getattr(lm, "graph_memory", None)

            if memory is None:
                continue

            setter = getattr(
                memory,
                "set_merge_animation_callback",
                None,
            )

            if not callable(setter):
                continue

            setter(self._animate_graph_merge)
            self._merge_hook_memories.append(memory)


    def _remove_merge_animation_hooks(self) -> None:
        """Detach merge callbacks previously installed by this plotter."""
        for memory in getattr(
            self,
            "_merge_hook_memories",
            [],
        ):
            setter = getattr(
                memory,
                "set_merge_animation_callback",
                None,
            )

            if callable(setter):
                setter(None)

        self._merge_hook_memories = []


    def _animate_graph_merge(
        self,
        *,
        memory,
        first_graph_id: str,
        new_graph_id: str,
        old_graph_ids: tuple[str, ...],
        location_rel_model: np.ndarray,
        channels: dict,
    ) -> None:
        """Animate the exact RF transforms being used by a Janus graph merge."""

        # Animation requires a GUI event loop.
        if self.fig is None or not is_interactive_backend():
            return

        # Only animate merges for the LM currently selected in Teleop.
        selected_memory = getattr(
            self._channel_view.lm,
            "graph_memory",
            None,
        )

        if memory is not selected_memory:
            return

        if not channels:
            return

        # Match the Memory panel's existing channel preference:
        #
        #   selected channel if the merged graph has it,
        #   otherwise first available merge channel.
        channel = self._channel_view.channel

        if channel not in channels:
            channel = next(iter(channels))

        event = channels[channel]

        target_points = np.asarray(
            event["target_points"],
            dtype=float,
        )

        merged_points = np.asarray(
            event["merged_points"],
            dtype=float,
        )

        location_rel_model = np.asarray(
            location_rel_model,
            dtype=float,
        )

        # ------------------------------------------------------------
        # Precompute the EXACT t=1 RF result for each source.
        #
        # This calls the exact same function GridObjectModel.update_model() uses.
        # ------------------------------------------------------------
        prepared_entries = []

        bounds_parts = [
            target_points,
            merged_points,
        ]

        for entry in event["entries"]:
            locations = np.asarray(
                entry["locations"],
                dtype=float,
            )

            features = copy.deepcopy(entry["features"])

            object_location_rel_body = np.asarray(
                entry["object_location_rel_body"],
                dtype=float,
            )

            object_rotation = entry["object_rotation"]

            final_locations, _final_features = apply_rf_transform_to_points(
                locations=locations.copy(),
                features=copy.deepcopy(features),
                location_rel_model=location_rel_model.copy(),
                object_location_rel_body=object_location_rel_body.copy(),
                object_rotation=object_rotation,
            )

            final_locations = np.asarray(
                final_locations,
                dtype=float,
            )

            prepared_entries.append(
                {
                    "source_graph_id": entry["source_graph_id"],
                    "locations": locations,
                    "features": features,
                    "object_location_rel_body": object_location_rel_body,
                    "object_rotation": object_rotation,
                    "final_locations": final_locations,
                }
            )

            # Use both endpoints when computing fixed animation bounds.
            bounds_parts.append(locations)
            bounds_parts.append(final_locations)

        non_empty_bounds = [
            points
            for points in bounds_parts
            if points.size
        ]

        if non_empty_bounds:
            bounds_points = np.vstack(non_empty_bounds)
        else:
            bounds_points = np.empty((0, 3))

        # ------------------------------------------------------------
        # Animation timing
        #
        # First 80%:
        #     replay apply_rf_transform_to_points as the object rotates/translates
        #
        # Last 20%:
        #     crossfade from transformed raw observations to the ACTUAL
        #     voxel/grid-built merged model.
        # ------------------------------------------------------------
        n_frames = 40
        # n_frames = 16

        transform_end = 0.80

        initial_source_points = {
            entry["source_graph_id"]: entry["locations"].copy()
            for entry in prepared_entries
        }

        self._memory.begin_merge_animation(
            first_graph_id=first_graph_id,
            new_graph_id=new_graph_id,
            source_points=initial_source_points,
            merged_points=merged_points,
            bounds_points=bounds_points,
        )

        for frame in range(n_frames + 1):
            overall_u = frame / n_frames

            # Phase 1: transformation.
            transform_u = min(
                overall_u / transform_end,
                1.0,
            )

            # Smoothstep.
            transform_t = (
                transform_u
                * transform_u
                * (3.0 - 2.0 * transform_u)
            )

            # Phase 2: crossfade to actual GridObjectModel output.
            if overall_u <= transform_end:
                merged_u = 0.0
            else:
                merged_u = (
                    overall_u - transform_end
                ) / (1.0 - transform_end)

            merged_alpha = (
                merged_u
                * merged_u
                * (3.0 - 2.0 * merged_u)
            )

            overall_t = (
                overall_u
                * overall_u
                * (3.0 - 2.0 * overall_u)
            )

            frame_source_points = {}

            for entry in prepared_entries:
                source_graph_id = entry["source_graph_id"]
                locations = entry["locations"]
                object_rotation = entry["object_rotation"]
                object_location_rel_body = entry[
                    "object_location_rel_body"
                ]

                if transform_u <= 0.0:
                    # Guarantee the very first frame is exactly the original model.
                    frame_locations = locations.copy()

                elif transform_u >= 1.0:
                    # Guarantee the end of the RF phase is exactly the same result
                    # as the real apply_rf_transform_to_points() call.
                    frame_locations = entry["final_locations"].copy()

                else:
                    # --------------------------------------------------------
                    # Rotation interpolation
                    #
                    # object_rotation supports as_rotvec()/from_rotvec().
                    # Scaling the rotation vector gives identity at t=0 and
                    # the real object rotation at t=1.
                    # --------------------------------------------------------
                    rotation_type = type(object_rotation)

                    partial_rotation = rotation_type.from_rotvec(
                        object_rotation.as_rotvec()
                        * transform_t
                    )

                    # --------------------------------------------------------
                    # Translation interpolation
                    #
                    # At t=0:
                    #
                    #   partial_location_rel_model
                    #       == object_location_rel_body
                    #
                    # Combined with identity rotation, the RF transform returns
                    # the original point coordinates.
                    #
                    # At t=1:
                    #
                    #   partial_location_rel_model
                    #       == real location_rel_model
                    #
                    # so this becomes the exact transform used by Janus.
                    # --------------------------------------------------------
                    partial_location_rel_model = (
                        (1.0 - transform_t)
                        * object_location_rel_body
                        + transform_t
                        * location_rel_model
                    )

                    frame_locations, _frame_features = (
                        apply_rf_transform_to_points(
                            locations=locations.copy(),
                            features=copy.deepcopy(
                                entry["features"]
                            ),
                            location_rel_model=(
                                partial_location_rel_model
                            ),
                            object_location_rel_body=(
                                object_location_rel_body.copy()
                            ),
                            object_rotation=partial_rotation,
                        )
                    )

                    frame_locations = np.asarray(
                        frame_locations,
                        dtype=float,
                    )

                frame_source_points[source_graph_id] = (
                    frame_locations
                )

            self._memory.update_merge_animation(
                source_points=frame_source_points,
                progress=overall_t,
                merged_alpha=merged_alpha,
            )

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # 40 frames × 25 ms ≈ one-second animation.
            # self.fig.canvas.start_event_loop(0.025)
            self.fig.canvas.start_event_loop(0.01)

    def awaits_choice(self, proposed: list[Action]) -> bool:
        """Whether the user should choose this step's action.

        Delegates to the interactive `ActionButtons` control's policy.

        Args:
            proposed: The actions the model computed for this step.

        Returns:
            True when this step is a user choice point.
        """
        return self._controls.awaits_choice(proposed)

    def override_action(
        self, ctx: RuntimeContext, proposed: list[Action]
    ) -> list[Action]:
        """Block until a button is clicked, then return the user's chosen action.

        Delegates to the interactive `ActionButtons` control, which draws this step's
        buttons, blocks on the figure's event loop, asks the interactive policy to
        compute the chosen action (or returns the proposed jump when "jump" is
        clicked), and raises `StopIteration` on "End episode".

        Args:
            ctx: The runtime context supplying the random state.
            proposed: The actions the Monty computed for this step.

        Returns:
            The actions to execute next, built from the user's button choice.
        """
        return self._controls.override_action(ctx, proposed)

    def close(self) -> None:
        """Close the final figure and drop widget references."""
        self._remove_merge_animation_hooks()

        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

        if self._controls is not None:
            self._controls.close()

        self._selector = None
        self._simulator = None
        self._monty = None
        self._details = None
        self._memory = None

    def _draw_training(self) -> None:
        """Draw the exploratory-step panels from the LM buffer.

        The Monty panel shows the selected channel's buffered points; the Details panel
        stacks one plot group per other channel. While the graph is still being built we
        cannot tell whether the points are planar, so every buffer view is drawn in 3D
        with three head-on 2D projections beneath it. Falls back to a placeholder when
        there are no observations yet.
        """
        locations = self._channel_view.lm.buffer.locations
        points = {
            c: self._channel_view.channel_points(np.asarray(locs))
            for c, locs in locations.items()
        }
        channels = [c for c in points if points[c].size]
        if not channels:
            self._monty.draw_placeholder("no observations yet")
            self._details.draw_placeholder("no observations yet")
            return

        selected = self._channel_view.channel
        selected_pts = points.get(selected) if selected is not None else None
        if selected is None or selected_pts is None or not selected_pts.size:
            label = selected if selected is not None else "channel"
            self._monty.draw_placeholder(f"no observations on {label}")
        else:
            self._monty.draw_selected_channel(selected, selected_pts)

        others = [c for c in channels if c != selected]
        if others:
            self._details.draw_buffer_grid(others, points)
        else:
            self._details.draw_placeholder("no other channels")

    def _draw_inference(self) -> None:
        """Draw the matching-step panels (MLH graph and line plots).

        Renders the displayed LM's MLH graph and line plots from the per-LM evidence
        history accumulated in `update`. Degrades to placeholders when the displayed LM
        lacks the evidence-LM inference API.
        """
        if not self._channel_view.supports_evidence:
            lm_name = type(self._channel_view.lm).__name__
            placeholder = f"Inference view not available for {lm_name}"
            self._monty.draw_placeholder(placeholder)
            self._details.draw_placeholder(placeholder)
            return
        self._monty.draw_mlh()
        self._details.draw_inference()
