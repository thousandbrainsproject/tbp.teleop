# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Browser-based live visualization and teleoperation for Monty experiments."""

from __future__ import annotations

import copy
import time
import webbrowser
from typing import TYPE_CHECKING, Any

import numpy as np

from tbp.monty.frameworks.actions.actions import SetAgentPose
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.no_reset_evidence_matching import (
    MontyForNoResetEvidenceGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    apply_rf_transform_to_points,
)

from tbp.teleop.policies import HEADINGS, interactive_policy_for
from tbp.teleop.web.server import CommandBridge, WebServer
from tbp.teleop.web.snapshot import SnapshotBuilder, SnapshotHistory, WebChannelView

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        LearningModule,
        Monty,
        Observations,
    )
    from tbp.monty.frameworks.utils.spatial_arithmetics import (
        apply_rf_transform_to_points,
    )
    from tbp.teleop.policies import InteractivePolicy


END_EPISODE = "End episode"
JUMP = "jump"


class WebPlotter:
    """Plotter implementation that moves presentation into a local browser UI.

    The class intentionally preserves the existing Plotter lifecycle. ``update`` never
    blocks for an action; interactive blocking remains exclusively in
    ``override_action``. Network callbacks never touch Monty state directly.
    """

    def __init__(
        self,
        interactive: bool = False,
        min_delay: float = 0.001,
        max_delay: float = 2.0,
        host: str = "127.0.0.1",
        port: int = 0,
        open_browser: bool = True,
    ) -> None:
        """Configure web presentation and control behavior.

        Args:
            interactive: Whether choice points should wait for browser input.
            min_delay: Monitor delay at maximum speed.
            max_delay: Monitor delay at the slow end of the speed control.
            host: Interface on which the local web server listens.
            port: TCP port, or ``0`` to reserve an ephemeral port.
            open_browser: Whether initialization opens the UI in a browser tab.
        """
        self.interactive = interactive
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.host = host
        self.port = port
        self.open_browser = open_browser

        self.model: Monty | None = None
        self._channel_view: WebChannelView | None = None
        self._history = SnapshotHistory()
        self._builder: SnapshotBuilder | None = None
        self._policy: InteractivePolicy | None = None
        self._bridge = CommandBridge()
        self._server: WebServer | None = None
        self._last_observations: Observations | None = None
        self._last_step: int | None = None
        self._supervised_lm_ids: list[str] = []
        self._request_serial = 0
        self._active_request_id: str | None = None
        self._merge_hook_memories: list[Any] = []

    @property
    def url(self) -> str | None:
        """Return the local UI URL once initialized."""
        return self._server.url if self._server is not None else None

    def initialize(self, model: Monty, supervised_lm_ids: list[str]) -> None:
        """Bind live Monty state and start the browser transport."""
        self._remove_merge_animation_hooks()
        self.model = model
        self._bridge.reset()
        self._channel_view = WebChannelView(model)
        self._history = SnapshotHistory()
        self._builder = SnapshotBuilder(model, self._channel_view, self._history)
        self._policy = interactive_policy_for(model) if self.interactive else None
        self._last_observations = None
        self._last_step = None
        self._supervised_lm_ids = supervised_lm_ids
        if self._server is None:
            self._server = WebServer(self._bridge, host=self.host, port=self.port)
            self._server.start()
            if self.open_browser:
                webbrowser.open(self._server.url, new=2)
    
        self._install_merge_animation_hooks()

    def update(self, observations: Observations, step: int) -> None:
        """Publish one frame and apply monitor pacing without action blocking."""
        model = self._model()
        self._server_or_raise()
        self._apply_queued_navigation()
        previous_step = self._last_step
        self._last_observations = observations
        self._last_step = step
        if self._is_continual(model):
            history_step = model.total_steps
        else:
            if previous_step is None or step <= previous_step:
                self._history.clear()
            history_step = step
        self._history.accumulate(model.learning_modules, history_step)
        self._publish_frame()
        if not self.interactive:
            self._monitor_pause()

    def _install_merge_animation_hooks(self) -> None:
        """Attach merge telemetry to graph memories that expose the callback."""
        self._merge_hook_memories = []

        for lm in self._model().learning_modules:
            memory = getattr(lm, "graph_memory", None)
            setter = getattr(memory, "set_merge_animation_callback", None)

            if callable(setter):
                setter(self._animate_graph_merge)
                self._merge_hook_memories.append(memory)

    def _remove_merge_animation_hooks(self) -> None:
        """Detach merge callbacks installed by this plotter."""
        for memory in self._merge_hook_memories:
            setter = getattr(memory, "set_merge_animation_callback", None)

            if callable(setter):
                setter(None)

        self._merge_hook_memories = []

    @staticmethod
    def _memory_display_points(
        points: np.ndarray,
        is_3d: bool,
    ) -> list[list[float]]:
        """Convert Monty coordinates to the Memory thumbnail display convention."""
        points = np.asarray(points, dtype=float)

        if points.ndim != 2:
            return []

        if is_3d and points.shape[1] >= 3:
            # Match MemoryPanel / MontyPanel: Y, X, Z.
            return points[:, [1, 0, 2]].tolist()

        return points[:, :2].tolist()

    def _animate_graph_merge(
        self,
        *,
        memory: Any,
        first_graph_id: str,
        new_graph_id: str,
        old_graph_ids: tuple[str, ...],
        location_rel_model: np.ndarray,
        channels: dict[str, Any],
    ) -> None:
        """Stream the same RF-transform merge animation as LivePlotter."""
        del old_graph_ids

        if self._channel_view is None or self._server is None or not channels:
            return

        # Just like the Matplotlib version, only animate merges belonging
        # to the LM currently selected in Teleop.
        selected_memory = getattr(
            self._channel_view.lm,
            "graph_memory",
            None,
        )

        if memory is not selected_memory:
            return

        # Prefer the selected input channel. If the merged graph does not
        # contain it, use the first channel represented by the merge event.
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

        is_3d = (
            target_points.ndim == 2
            and target_points.shape[1] >= 3
            and not np.allclose(target_points[:, 2], 0.0)
        )

        # ------------------------------------------------------------
        # Prepare every source graph and calculate its EXACT final RF
        # transform using the same function Monty uses for the real merge.
        # ------------------------------------------------------------
        prepared: list[dict[str, Any]] = []

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

            prepared.append(
                {
                    "source_graph_id": entry["source_graph_id"],
                    "locations": locations,
                    "features": features,
                    "object_location_rel_body": object_location_rel_body,
                    "object_rotation": object_rotation,
                    "final_locations": final_locations,
                }
            )

            # Include both ends of the animation when calculating a fixed
            # bounding box, preventing zoom jitter while objects rotate.
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

        # Tell the browser a merge animation is starting.
        #
        # The browser gets only display-ready coordinates; all actual
        # Monty/RF transformation logic remains here in Python.
        self._server.publish(
            {
                "type": "memory_merge_begin",
                "target_id": first_graph_id,
                "new_graph_id": new_graph_id,
                "is_3d": is_3d,
                "target_points": self._memory_display_points(
                    target_points,
                    is_3d,
                ),
                "source_points": {
                    entry["source_graph_id"]: self._memory_display_points(
                        entry["locations"],
                        is_3d,
                    )
                    for entry in prepared
                },
                "merged_points": self._memory_display_points(
                    merged_points,
                    is_3d,
                ),
                "bounds_points": self._memory_display_points(
                    bounds_points,
                    is_3d,
                ),
            }
        )

        # ------------------------------------------------------------
        # Animation timing.
        #
        # First 80%:
        #     replay the RF transform
        #
        # Last 20%:
        #     crossfade transformed observations into the actual merged
        #     GridObjectModel.
        # ------------------------------------------------------------
        n_frames = 40
        transform_end = 0.80
        frame_delay = 0.06

        for frame in range(n_frames + 1):
            overall_u = frame / n_frames

            # Phase 1: RF transformation.
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

            # Phase 2: reveal the actual merged graph.
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

            # Used to fade the source Memory thumbnails away.
            overall_t = (
                overall_u
                * overall_u
                * (3.0 - 2.0 * overall_u)
            )

            frame_source_points: dict[str, list[list[float]]] = {}

            for entry in prepared:
                source_graph_id = entry["source_graph_id"]
                locations = entry["locations"]
                object_rotation = entry["object_rotation"]
                object_location_rel_body = entry[
                    "object_location_rel_body"
                ]

                if transform_u <= 0.0:
                    # Guarantee frame zero is exactly the source model.
                    frame_locations = locations.copy()

                elif transform_u >= 1.0:
                    # Guarantee the end of the RF phase is exactly the
                    # result of the real apply_rf_transform_to_points().
                    frame_locations = entry["final_locations"].copy()

                else:
                    # Interpolate rotation from identity to the real
                    # object rotation.
                    rotation_type = type(object_rotation)

                    partial_rotation = rotation_type.from_rotvec(
                        object_rotation.as_rotvec()
                        * transform_t
                    )

                    # At t=0 this equals object_location_rel_body, which
                    # cancels the translation and leaves the original
                    # coordinates. At t=1 it equals location_rel_model,
                    # making this the real merge transform.
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
                    self._memory_display_points(
                        frame_locations,
                        is_3d,
                    )
                )

            self._server.publish(
                {
                    "type": "memory_merge_frame",
                    "source_points": frame_source_points,
                    "progress": overall_t,
                    "merged_alpha": merged_alpha,
                }
            )

            # 41 frames at ~10 ms is about the timing of your current
            # Matplotlib implementation.
            time.sleep(frame_delay)

        # Leave the fully merged model on screen for a moment before Monty
        # is allowed to continue into the next episode.
        time.sleep(1.0)


        self._server.publish(
            {
                "type": "memory_merge_end",
            }
        )

    def awaits_choice(self, proposed: list[Action]) -> bool:
        """Return whether this step is one of the existing policy choice points."""
        if self._is_jump(proposed):
            return True
        if self._policy is None:
            return False
        return self._policy.awaits_choice(proposed)

    def override_action(
        self,
        ctx: RuntimeContext,
        proposed: list[Action],
    ) -> list[Action]:
        """Wait for a browser choice and preserve existing policy/log semantics."""
        model = self._model()
        server = self._server_or_raise()
        if self._policy is None:
            return proposed
        specials = [JUMP, END_EPISODE] if self._is_jump(proposed) else [END_EPISODE]
        self._request_serial += 1
        request_id = f"{self._last_step}:{self._request_serial}"
        self._active_request_id = request_id
        server.publish(
            {
                "type": "action_request",
                "request_id": request_id,
                "headings": list(HEADINGS),
                "specials": specials,
                "keyboard": {
                    "w": "up",
                    "ArrowUp": "up",
                    "s": "down",
                    "ArrowDown": "down",
                    "a": "left",
                    "ArrowLeft": "left",
                    "d": "right",
                    "ArrowRight": "right",
                    "Space": JUMP,
                    "Delete": END_EPISODE,
                },
            }
        )
        selected = self._wait_for_action(request_id, set(HEADINGS) | set(specials))
        self._active_request_id = None
        server.publish({"type": "action_resolved", "request_id": request_id})
        if selected == END_EPISODE:
            model.deal_with_time_out()
            raise StopIteration
        if selected == JUMP:
            return proposed
        state = model.motor_system.action_sequence[-1][1]
        chosen = self._policy.compute(ctx, selected, state, self._bridge.step_scale)
        self._policy.feedback(chosen)
        model.motor_system.action_sequence[-1] = (chosen, state)
        return chosen

    def close(self) -> None:
        """Release the local server and browser-side wait state."""
        self._remove_merge_animation_hooks()

        if self._server is not None:
            self._server.publish({"type": "session_closed"})
            self._server.stop()
            self._server = None
        self._active_request_id = None
        self._last_observations = None
        self.model = None
        self._builder = None
        self._channel_view = None
        self._policy = None

    def _wait_for_action(self, request_id: str, valid: set[str]) -> str:
        while True:
            command = self._bridge.get(timeout=0.1)
            if command is None:
                continue
            kind = command.get("type")
            if kind == "action":
                if command.get("request_id") != request_id:
                    continue
                name = str(command.get("name", ""))
                if name in valid:
                    return name
                continue
            if kind in {"select_lm", "select_channel"}:
                if self._apply_navigation(command):
                    self._publish_frame()

    def _monitor_pause(self) -> None:
        while True:
            self._apply_queued_navigation(redraw=True)
            speed = self._bridge.speed
            delay = self._pause_seconds(speed)
            if delay is None:
                self._bridge.wait_for_change(0.1)
                continue
            deadline = time.monotonic() + delay
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._bridge.wait_for_change(min(remaining, 0.05))
                # A speed change should take effect immediately rather than after an
                # old, slower delay finishes.
                if self._bridge.speed != speed:
                    break
                self._apply_queued_navigation(redraw=True)
            # Recompute the delay from the new speed.

    def _pause_seconds(self, speed: float) -> float | None:
        if speed <= 0.0:
            return None
        speed = min(speed, 1.0)
        return self.min_delay + (self.max_delay - self.min_delay) * (1.0 - speed)

    def _apply_queued_navigation(self, *, redraw: bool = False) -> None:
        changed = False
        for command in self._bridge.drain():
            if command.get("type") in {"select_lm", "select_channel"}:
                changed = self._apply_navigation(command) or changed
        # Action commands are only meaningful while override_action is blocking. If
        # they arrive outside a request, discard them; request IDs prevent stale clicks
        # from crossing choice points.
        if redraw and changed:
            self._publish_frame()

    def _apply_navigation(self, command: dict[str, Any]) -> bool:
        if self._channel_view is None:
            return False
        kind = command.get("type")
        if kind == "select_lm":
            return self._channel_view.select_lm(str(command.get("id", "")))
        if kind == "select_channel":
            return self._channel_view.select_channel(str(command.get("id", "")))
        return False

    def _publish_frame(self) -> None:
        if (
            self._builder is None
            or self._channel_view is None
            or self._server is None
            or self._last_observations is None
            or self._last_step is None
        ):
            return
        frame = self._builder.build(
            self._last_observations,
            self._last_step,
            building_graph=self._lm_building_graph(self._channel_view.lm),
            interactive=self.interactive,
            speed=self._bridge.speed,
            step_scale=self._bridge.step_scale,
            url=self._server.url,
        )
        self._server.publish(frame)

    def _lm_building_graph(self, lm: LearningModule) -> bool:
        model = self._model()
        if model.step_type == "exploratory_step":
            return True
        return (
            model.experiment_mode is ExperimentMode.TRAIN
            and lm.learning_module_id in self._supervised_lm_ids
        )

    @staticmethod
    def _is_jump(proposed: list[Action]) -> bool:
        return bool(proposed) and isinstance(proposed[0], SetAgentPose)

    @staticmethod
    def _is_continual(model: Monty) -> bool:
        return isinstance(model, MontyForNoResetEvidenceGraphMatching)

    def _model(self) -> Monty:
        if self.model is None:
            raise RuntimeError("WebPlotter.initialize must be called before use")
        return self.model

    def _server_or_raise(self) -> WebServer:
        if self._server is None:
            raise RuntimeError("WebPlotter.initialize must be called before use")
        return self._server
