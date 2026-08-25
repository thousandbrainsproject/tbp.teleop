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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.no_reset_evidence_matching import (
    MontyForNoResetEvidenceGraphMatching,
)

from tbp.teleop.controls import (
    ActionButtons,
    SelectorBar,
    SpeedSlider,
)
from tbp.teleop.goals import JumpWatcher, goal_status
from tbp.teleop.helpers import (
    ChannelView,
    EvidenceHistory,
    draw_section_dividers,
    is_interactive_backend,
)
from tbp.teleop.panels import (
    AttentionPanel,
    DetailsPanel,
    MontyPanel,
    SegmentationPanel,
    SimulatorPanel,
)
from tbp.teleop.plotter import Plotter

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        Monty,
        Observations,
    )


# How many steps a failed-jump banner stays visible after the move-back step.
FAILED_JUMP_BANNER_STEPS = 5


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

    With `attention_vis` enabled, the three sections are compressed into a top row
    (about half the figure height) and a second row is added along the bottom with two
    attention-debugging panels: the `AttentionSystem`'s live voxel grid in 3D world
    space, and the segmented region proposed by the model-free sensor module (e.g.
    `SlicMerge`) overlaid on its camera view. To make room, the "Input Feature" inset
    and the "Number of hypotheses per object" plot are dropped in this layout.

    Goals emitted by SMs and LMs are surfaced in several ways: a status line in the
    top-left corner names the enacted goal's action and source module each step; with
    `attention_vis` enabled, the attention panel overlays every proposed goal on the
    voxel grid, styled by whether it fell within the active attention space; the
    matching-step MLH view marks the displayed LM's current goal target on the
    hypothesized model; and a red top-right banner reports for a few steps when a
    goal was unsuccessful because no object was visible at its location and Monty
    moved back.
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
        attention_vis: bool = False,
    ) -> None:
        """Initialize the plotter.

        Args:
            interactive: Whether to render action buttons and override the executed
                action with the user's choice (otherwise render a speed slider).
            min_delay: Non-interactive pause in seconds at full speed.
            max_delay: Maximum non-interactive pause in seconds at the slowest speed.
            figsize: Figure size in inches.
            attention_vis: Whether to add the bottom row with the attention voxel-grid
                and segmented-region panels (dropping the feature inset and the
                hypotheses plot to make room).
        """
        # Turn interactive plotting off so the plotter controls when figures are
        # drawn and when execution blocks, via its own canvas event loop.
        plt.ioff()

        self.interactive = interactive
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.figsize = figsize
        self.attention_vis = attention_vis

        self.fig = None
        self._controls = None

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
        self._jump_watcher = JumpWatcher()
        self._banner_message: str | None = None
        self._banner_until: int | None = None

        self._build_figure()

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
        if self.attention_vis:
            # Two rows: the three regular sections on top at roughly half height, the
            # attention voxel grid and segmented region along the bottom. The top
            # margin is lowered (vs. the figure-wide 0.9) so the compressed top row's
            # axis titles clear the selector buttons at 0.91.
            outer = self.fig.add_gridspec(
                2,
                1,
                height_ratios=[1.0, 0.95],
                hspace=0.4,
                top=0.86,
                bottom=0.16,
                left=0.04,
                right=0.97,
            )
            top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 0], wspace=0.25)
            bottom = GridSpecFromSubplotSpec(
                1,
                3 if self.interactive else 2,
                subplot_spec=outer[1, 0],
                wspace=0.25,
            )
            self._sim_spec = top[0, 0]
            self._monty_spec = top[0, 1]
            self._details_spec = top[0, 2]
            if self.interactive:
                # The interactive step-multiplier slider sits below the Simulator
                # column's RGB patch, so the bottom-left third is left free for it and
                # the two panels align under the Monty and Details columns.
                self._attention_spec = bottom[0, 1]
                self._segmentation_spec = bottom[0, 2]
            else:
                self._attention_spec = bottom[0, 0]
                self._segmentation_spec = bottom[0, 1]
        else:
            outer = self.fig.add_gridspec(1, 3)
            self._sim_spec = outer[0, 0]
            self._monty_spec = outer[0, 1]
            self._details_spec = outer[0, 2]

        self._simulator = SimulatorPanel(self.fig, self._sim_spec)
        self._monty = MontyPanel(self.fig, self._monty_spec, self._channel_view)
        self._details = DetailsPanel(
            self.fig,
            self._details_spec,
            self._channel_view,
            self._history,
            show_num_hypotheses=not self.attention_vis,
        )
        if self.attention_vis:
            self._attention = AttentionPanel(
                self.fig, self._attention_spec, interactive=self.interactive
            )
            self._segmentation = SegmentationPanel(
                self.fig, self._segmentation_spec, self.model
            )
        else:
            self._attention = None
            self._segmentation = None
        draw_section_dividers(
            self.fig, self._sim_spec, self._monty_spec, self._details_spec
        )
        self._selector = SelectorBar(
            self.fig, self._monty_spec, self._channel_view, self._redraw
        )
        if self.interactive:
            self._controls.build(self.fig, self._simulator.ax_rgb)
        else:
            self._controls.build(self.fig)

        # Two figure-level status texts flanking the `Step N` title: the enacted
        # goal's action and source on the left, the failed-jump banner on the right.
        self._goal_text = self.fig.text(
            0.02, 0.99, "", ha="left", va="top", fontsize=9, color="0.25"
        )
        self._banner_text = self.fig.text(
            0.98,
            0.99,
            "",
            ha="right",
            va="top",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

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
        self._observe_jump(
            step, new_episode=prev_step is not None and step <= prev_step
        )
        self._render(observations, step)
        if not self.interactive:
            self._controls.pause()

    def _observe_jump(self, step: int, *, new_episode: bool) -> None:
        """Track this step's jump outcome and manage the failed-jump banner.

        Observes the motor system's jump state exactly once per step (repaints via
        `_redraw` never re-observe). When a jump failed because no object was visible
        at the goal location and Monty moved back, the failure message is shown for
        `FAILED_JUMP_BANNER_STEPS` steps. An episode restart drops the watcher's
        cross-step state along with any leftover banner.

        Args:
            step: The index of the current step within the episode.
            new_episode: Whether this step starts a new episode.
        """
        if new_episode:
            self._jump_watcher = JumpWatcher()
            self._banner_message = None
            self._banner_until = None
        message = self._jump_watcher.observe(self.model)
        if message is not None:
            self._banner_message = message
            self._banner_until = step + FAILED_JUMP_BANNER_STEPS
        elif self._banner_until is not None and step > self._banner_until:
            self._banner_message = None
            self._banner_until = None

    def _render(self, observations: Observations, step: int) -> None:
        """Draw every section for one frame.

        Args:
            observations: The observations to draw.
            step: The index of the current step within the episode.
        """
        self.fig.suptitle(f"Step {step}")
        self._goal_text.set_text(goal_status(self.model))
        self._banner_text.set_text(self._banner_message or "")
        if self._channel_view.ensure_channel():
            self._selector.refresh_labels()

        sm, sm_id = self._channel_view.simulator_sm()
        self._simulator.draw(observations, sm, sm_id)

        if self._lm_building_graph(self._channel_view.lm):
            self._draw_training()
        else:
            self._draw_inference()

        if self.attention_vis:
            self._attention.draw(self.model)
            self._segmentation.draw()
        else:
            self._monty.draw_feature_inset()

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
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        if self._controls is not None:
            self._controls.close()
        self._selector = None
        self._simulator = None
        self._monty = None
        self._details = None
        self._attention = None
        self._segmentation = None

    def _draw_training(self) -> None:
        """Draw the exploratory-step panels from the LM buffer.

        The Monty panel shows the selected channel's buffered points; the Details panel
        stacks one plot group per other channel. The buffer stores a single global
        location per step (the mean of the sensor module locations), so every channel's
        cloud is drawn on those shared coordinates, masked to the steps that channel was
        active; learning-module channels therefore sit on the sensor-mean locations
        rather than their own object-frame coordinates. While the graph is still being
        built we cannot tell whether the points are planar, so every buffer view is
        drawn in 3D with three head-on 2D projections beneath it. Falls back to a
        placeholder when there are no observations yet.
        """
        buffer_channels = self._channel_view.lm_channels()
        points = {c: self._channel_view.channel_points(c) for c in buffer_channels}
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
