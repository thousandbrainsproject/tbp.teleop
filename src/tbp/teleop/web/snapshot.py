# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Translate live Monty state into a stable, renderer-neutral web protocol.

Nothing in this module knows about DOM layout. It produces semantic visualization
records so Monty representation changes can be absorbed here without changing the UI.
"""

from __future__ import annotations

import base64
import colorsys
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import quaternion as qt
from tbp.monty.frameworks.models.evidence_matching.burst_sampling import (
    BurstSamplingHypothesesUpdater,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.two_d_sensor_module import TwoDSensorModule
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.plot_utils import add_patch_outline_to_view_finder

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import (
        AgentObservations,
        LearningModule,
        Monty,
        Observations,
        SensorModule,
    )


PROTOCOL_VERSION = 1


def current_mlh(lm: object) -> dict[str, Any] | None:
    """Call the MLH accessor across current and upcoming tbp.monty APIs."""
    getter = getattr(lm, "_get_current_mlh", None)
    if getter is None:
        getter = getattr(lm, "get_current_mlh", None)
    if getter is None:
        return None
    value = getter()
    return value if isinstance(value, dict) else None


class SnapshotHistory:
    """Per-LM evidence history with compatibility for both MLH accessor names."""

    def __init__(self) -> None:
        """Initialize empty, per-learning-module history buffers."""
        self.steps_by_lm: dict[str, list[int]] = {}
        self.evidence_by_lm: dict[str, dict[str, list[float]]] = {}
        self.num_hyp_by_lm: dict[str, dict[str, list[float]]] = {}
        self.burst_steps_by_lm: dict[str, list[int]] = {}
        self._last_accumulated_step: int | None = None

    def clear(self) -> None:
        """Drop all accumulated history and restart step de-duplication."""
        self.steps_by_lm.clear()
        self.evidence_by_lm.clear()
        self.num_hyp_by_lm.clear()
        self.burst_steps_by_lm.clear()
        self._last_accumulated_step = None

    def accumulate(self, learning_modules: Sequence[LearningModule], step: int) -> None:
        """Record one evidence-history sample for every compatible LM.

        Args:
            learning_modules: Learning modules in the live Monty model.
            step: Episode or continual-learning step used on the history x-axis.
        """
        if self._last_accumulated_step == step:
            return
        self._last_accumulated_step = step
        for lm in learning_modules:
            if isinstance(lm, EvidenceGraphLM):
                self._append(lm, step)

    @staticmethod
    def _num_hypotheses(lm: EvidenceGraphLM) -> npt.NDArray[np.float64]:
        graph_ids = lm.get_all_known_object_ids()
        if not graph_ids or graph_ids[0] not in lm._hypotheses:
            return np.array([0.0])
        counts = []
        for graph_id in graph_ids:
            evidence = lm._hypotheses[graph_id].evidence
            if len(evidence):
                counts.append(float(len(evidence)))
        return np.asarray(counts, dtype=float)

    def _append(self, lm: EvidenceGraphLM, step: int) -> None:
        mlh = current_mlh(lm)
        if not mlh or mlh.get("graph_id") == "no_observations_yet":
            return
        graph_ids, evidences = lm.evidence_for_each_graph()
        num_hyps = self._num_hypotheses(lm)
        evidence_by_id = dict(zip(graph_ids, evidences))
        num_hyp_by_id = dict(zip(graph_ids, num_hyps))
        lm_id = lm.learning_module_id
        steps = self.steps_by_lm.setdefault(lm_id, [])
        evidence_history = self.evidence_by_lm.setdefault(lm_id, {})
        num_hyp_history = self.num_hyp_by_lm.setdefault(lm_id, {})
        burst_steps = self.burst_steps_by_lm.setdefault(lm_id, [])
        steps.append(step)
        n = len(steps)
        for graph_id in graph_ids:
            if graph_id not in evidence_history:
                evidence_history[graph_id] = [np.nan] * (n - 1)
                num_hyp_history[graph_id] = [np.nan] * (n - 1)
        for graph_id in evidence_history:
            evidence_history[graph_id].append(
                float(evidence_by_id.get(graph_id, np.nan))
            )
            num_hyp_history[graph_id].append(
                float(num_hyp_by_id.get(graph_id, np.nan))
            )
        updater = lm.hypotheses_updater
        if (
            isinstance(updater, BurstSamplingHypothesesUpdater)
            and updater.sampling_burst_steps > 0
        ):
            burst_steps.append(step)


class WebChannelView:
    """LM/channel selection plus a compatibility boundary around LM buffer layouts."""

    def __init__(self, model: Monty) -> None:
        """Select the first LM and defer channel choice until data is available.

        Args:
            model: Monty model whose modules and buffers feed visualization data.
        """
        self.model = model
        self.lm_index = 0
        self.lm = model.learning_modules[0]
        self.channel: str | None = None
        self.supports_evidence = isinstance(self.lm, EvidenceGraphLM)

    def ensure_channel(self) -> bool:
        """Choose the default channel once the selected LM exposes channels.

        Returns:
            True when a channel was newly selected.
        """
        if self.channel is None:
            self.channel = self.default_channel()
            return True
        return False

    def select_lm(self, lm_id: str) -> bool:
        """Select a learning module by id and reset its channel selection.

        Args:
            lm_id: Learning-module id sent by the browser.

        Returns:
            True when the selected learning module changed.
        """
        for index, lm in enumerate(self.model.learning_modules):
            if lm.learning_module_id == lm_id:
                if index == self.lm_index:
                    return False
                self.lm_index = index
                self.lm = lm
                self.supports_evidence = isinstance(lm, EvidenceGraphLM)
                self.channel = self.default_channel()
                return True
        return False

    def select_channel(self, channel: str) -> bool:
        """Select one input channel of the displayed learning module.

        Args:
            channel: Channel id sent by the browser.

        Returns:
            True when the selected channel changed.
        """
        if channel not in self.lm_channels() or channel == self.channel:
            return False
        self.channel = channel
        return True

    def lm_channels(self) -> list[str]:
        """Return the selected LM's input channels in buffer order.

        Returns:
            Channel ids known to the displayed LM buffer.
        """
        return list(self.lm.buffer.channel_sender_types)

    def default_channel(self) -> str | None:
        """Prefer the first sensor-module channel, then any available channel.

        Returns:
            Default channel id, or None while the LM has no channels.
        """
        channels = self.lm_channels()
        sender_types = self.lm.buffer.channel_sender_types
        sm_channels = [
            channel for channel in channels if sender_types.get(channel) == "SM"
        ]
        if sm_channels:
            return sm_channels[0]
        return channels[0] if channels else None

    def resolve_sm_channel(self, channel: str | None) -> SensorModule | None:
        """Resolve an SM channel id to the matching sensor-module instance.

        Args:
            channel: Channel id to resolve.

        Returns:
            Matching sensor module, or None for a non-SM or unknown channel.
        """
        if channel is None or self.lm.buffer.channel_sender_types.get(channel) != "SM":
            return None
        return next(
            (
                sm
                for sm in self.model.sensor_modules
                if sm.sensor_module_id == channel
            ),
            None,
        )

    def resolve_lm_channel(self, channel: str | None) -> LearningModule | None:
        """Resolve an LM channel id to the source learning-module instance.

        Args:
            channel: Channel id to resolve.

        Returns:
            Matching learning module, or None for a non-LM or unknown channel.
        """
        if channel is None or self.lm.buffer.channel_sender_types.get(channel) != "LM":
            return None
        return next(
            (
                lm
                for lm in self.model.learning_modules
                if lm.learning_module_id == channel
            ),
            None,
        )

    def simulator_sm(self) -> tuple[SensorModule | None, str | None]:
        """Resolve the sensor module that should drive the Simulator column.

        Returns:
            Sensor module and its id, or ``(None, None)`` when none is available.
        """
        sm = self.resolve_sm_channel(self.channel)
        if sm is not None:
            return sm, self.channel
        sender_types = self.lm.buffer.channel_sender_types
        for channel in self.lm_channels():
            if sender_types.get(channel) == "SM":
                sm = self.resolve_sm_channel(channel)
                if sm is not None:
                    return sm, channel
        return None, None

    def channel_points(self, channel: str) -> npt.NDArray[np.float64]:
        """Return valid points under per-channel and shared-location buffers.

        Args:
            channel: Input channel whose aligned locations should be returned.

        Returns:
            Valid location rows as an ``(N, 3)`` floating-point array.
        """
        locations = self.lm.buffer.locations
        if isinstance(locations, Mapping):
            if channel not in locations:
                return np.empty((0, 3), dtype=float)
            array = np.asarray(locations[channel], dtype=float)
            if array.ndim != 2 or not len(array) or array.shape[1] < 3:
                return np.empty((0, 3), dtype=float)
            return array[~np.isnan(array[:, 0]), :3]

        array = np.asarray(locations, dtype=float)
        if array.ndim != 2 or not len(array) or array.shape[1] < 3:
            return np.empty((0, 3), dtype=float)
        mask = self._shared_location_mask(channel, array.shape[0])
        return array[mask, :3]

    def aligned_feature(
        self,
        channel: str,
        attr: str,
    ) -> npt.NDArray[np.float64] | None:
        """Return one channel feature aligned to the channel's valid locations.

        Args:
            channel: Input channel to inspect.
            attr: Buffer feature name.

        Returns:
            Aligned feature rows, or None when absent, malformed, or incomplete.
        """
        channel_feats = self.lm.buffer.features.get(channel)
        if not channel_feats or attr not in channel_feats:
            return None
        locations = self.lm.buffer.locations
        raw = np.asarray(channel_feats[attr], dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        if raw.ndim != 2:
            return None

        if isinstance(locations, Mapping):
            channel_locations = np.asarray(locations.get(channel, []), dtype=float)
            if (
                channel_locations.ndim != 2
                or raw.shape[0] != channel_locations.shape[0]
            ):
                return None
            valid = raw[~np.isnan(channel_locations[:, 0])]
        else:
            shared = np.asarray(locations, dtype=float)
            if shared.ndim != 2:
                return None
            padded = self._pad_rows(raw, shared.shape[0])
            valid = padded[self._shared_location_mask(channel, shared.shape[0])]

        if valid.size == 0 or np.isnan(valid).any():
            return None
        return valid

    def _shared_location_mask(
        self,
        channel: str,
        target_length: int,
    ) -> npt.NDArray[np.bool_]:
        channel_feats = self.lm.buffer.features.get(channel, {})
        on_object = channel_feats.get("on_object")
        if on_object is None:
            # A conservative compatibility fallback. If a branch exposes shared
            # locations before exposing an on_object validity feature, retain rows
            # whose first location coordinate is finite rather than crashing.
            locations = np.asarray(self.lm.buffer.locations, dtype=float)
            return ~np.isnan(locations[:target_length, 0])
        raw = np.asarray(on_object, dtype=float)
        if raw.ndim == 1:
            raw = raw[:, None]
        padded = self._pad_rows(raw, target_length)
        return ~np.isnan(padded[:, 0])

    def _pad_rows(
        self,
        array: npt.NDArray[np.float64],
        target_length: int,
    ) -> npt.NDArray[np.float64]:
        padder = getattr(self.lm.buffer, "_pad_to_target_length", None)
        if padder is not None:
            try:
                padded = np.asarray(padder(array), dtype=float)
                if padded.ndim == 2 and padded.shape[0] == target_length:
                    return padded
            except (TypeError, ValueError):
                pass
        width = array.shape[1] if array.ndim == 2 else 1
        padded = np.full((target_length, width), np.nan, dtype=float)
        count = min(target_length, array.shape[0])
        padded[:count] = array[:count]
        return padded

    def object_id_names(self, channel: str) -> dict[int, str]:
        """Invert object-id feature hashes for an LM-fed input channel.

        Args:
            channel: Learning-module input channel.

        Returns:
            Mapping from numeric object id to human-readable graph id.
        """
        source_lm = self.resolve_lm_channel(channel)
        if not isinstance(source_lm, EvidenceGraphLM):
            return {}
        return {
            sum(ord(char) for char in graph_id): graph_id
            for graph_id in source_lm.graph_memory.get_memory_ids()
        }

    def point_groups(
        self,
        channel: str,
        points: npt.NDArray[np.float64],
    ) -> list[dict[str, Any]]:
        """Create semantic point groups using the current LivePlotter precedence.

        Args:
            channel: Channel whose features determine grouping and color.
            points: Valid channel locations.

        Returns:
            Renderer-neutral point groups with colors or stable series indices.
        """
        hsv = self.aligned_feature(channel, "hsv")
        if hsv is not None and hsv.shape[0] == points.shape[0] and hsv.shape[1] >= 3:
            colors = [_rgb_css(_hsv_to_rgb(row[:3])) for row in hsv]
            return [{"label": None, "points": points.tolist(), "colors": colors}]
        object_id = self.aligned_feature(channel, "object_id")
        if object_id is not None and object_id.shape[0] == points.shape[0]:
            ids = object_id[:, 0]
            names = self.object_id_names(channel)
            groups = []
            for series_index, uid in enumerate(np.unique(ids)):
                selected = points[ids == uid]
                groups.append(
                    {
                        "label": names.get(int(uid), f"object {int(uid)}"),
                        "points": selected.tolist(),
                        "series_index": series_index,
                    }
                )
            return groups
        return [{"label": None, "points": points.tolist(), "series_index": 0}]


class SnapshotBuilder:
    """Build complete frames from observations, channel selection, and history."""

    def __init__(
        self,
        model: Monty,
        channel_view: WebChannelView,
        history: SnapshotHistory,
    ) -> None:
        """Bind model state, channel selection, and accumulated history.

        Args:
            model: Live Monty model.
            channel_view: Runtime LM/channel selection and buffer compatibility layer.
            history: Episode evidence history shared across frame rebuilds.
        """
        self.model = model
        self.channel_view = channel_view
        self.history = history

    def _memory_extras(self, limit: int = 8) -> list[dict[str, Any]]:
        """Build Memory thumbnails for the currently selected learning module."""
        lm = self.channel_view.lm

        required = (
            "get_all_known_object_ids",
            "get_input_channels_in_graph",
            "get_graph",
        )
        if not all(hasattr(lm, name) for name in required):
            return [
                {
                    "id": "memory",
                    "title": "Memory",
                    **_placeholder("No graph memory"),
                }
            ]

        model_ids = list(lm.get_all_known_object_ids())

        if hasattr(lm, "evidence_for_each_graph"):
            evidence_ids, evidences = lm.evidence_for_each_graph()
            evidence_by_id = dict(zip(evidence_ids, evidences))

            model_ids.sort(
                key=lambda model_id: evidence_by_id.get(model_id, -np.inf),
                reverse=True,
            )

        extras = []

        for model_id in model_ids[:limit]:
            channels = list(lm.get_input_channels_in_graph(model_id))

            if not channels:
                continue

            selected = self.channel_view.channel
            channel = selected if selected in channels else channels[0]

            graph = lm.get_graph(model_id, channel)
            pos = np.asarray(graph.pos, dtype=float)

            if pos.ndim != 2 or not pos.size:
                continue

            # ------------------------------------------------------------
            # Read the per-node HSV color stored in the graph.
            # ------------------------------------------------------------
            hsv = None

            if "hsv" in graph.feature_mapping:
                hsv = np.asarray(
                    graph.get_values_for_feature("hsv"),
                    dtype=float,
                )

            # Convert Monty's HSV values into browser-ready RGB strings.
            #
            # Keep the colors in exactly the same node order as graph.pos,
            # so colors[i] belongs to pos[i].
            colors = None

            if (
                hsv is not None
                and hsv.ndim == 2
                and hsv.shape[0] == pos.shape[0]
                and hsv.shape[1] >= 3
            ):
                colors = [
                    _rgb_css(_hsv_to_rgb(row[:3]))
                    for row in hsv
                ]


            if _is_3d(pos):
                # Match MemoryPanel / Monty inference orientation: Y, X, Z.
                #
                # This only rearranges coordinate dimensions; it does NOT
                # rearrange node order, so colors still line up by index.
                display = pos[:, [1, 0, 2]]

                visualization = {
                    "kind": "point_cloud",
                    "groups": [
                        {
                            "label": None,
                            "points": display.tolist(),

                            # NEW
                            "colors": colors,
                        }
                    ],
                    "projections": False,
                    "frame": _frame(display),
                }

            else:
                display = pos[:, :2]

                visualization = {
                    "kind": "planar_cloud",
                    "points": display.tolist(),

                    # NEW
                    "colors": colors,

                    "frame": _frame(display),
                }

            extras.append(
                {
                    "id": f"memory:{model_id}",
                    "title": str(model_id),
                    "memory_channel": str(channel),
                    **visualization,
                }
            )

        if extras:
            return extras

        return [
            {
                "id": "memory",
                "title": "Memory",
                **_placeholder("Memory is empty"),
            }
        ]

    def build(
        self,
        observations: Observations,
        step: int,
        *,
        building_graph: bool,
        interactive: bool,
        speed: float,
        step_scale: float,
        url: str,
    ) -> dict[str, Any]:
        """Build one complete, semantic browser frame.

        Args:
            observations: Most recent Monty observations.
            step: Current episode step.
            building_graph: Whether the selected LM should show its live buffer.
            interactive: Whether the run uses browser action choice points.
            speed: Current monitor speed control.
            step_scale: Current interactive movement multiplier.
            url: Local browser URL shown in session metadata.

        Returns:
            JSON-serializable protocol-v1 visualization frame.
        """
        self.channel_view.ensure_channel()
        lm = self.channel_view.lm
        simulator = self._simulator(observations)
        if building_graph:
            monty, details = self._training_views()
            phase = "exploring"
        else:
            monty, details = self._inference_views()
            phase = "matching"
        feature = self._feature(self.channel_view.channel)
        return {
            "type": "frame",
            "protocol": PROTOCOL_VERSION,
            "session": {
                "step": int(step),
                "phase": phase,
                "interactive": interactive,
                "experiment_mode": _enumish(self.model.experiment_mode),
                "monty_done": bool(getattr(self.model, "is_done", False)),
                "url": url,
            },
            "selectors": {
                "learning_modules": [
                    {
                        "id": item.learning_module_id,
                        "type": type(item).__name__,
                    }
                    for item in self.model.learning_modules
                ],
                "active_lm": lm.learning_module_id,
                "channels": [
                    {
                        "id": channel,
                        "sender_type": lm.buffer.channel_sender_types.get(channel),
                    }
                    for channel in self.channel_view.lm_channels()
                ],
                "active_channel": self.channel_view.channel,
            },
            "simulator": simulator,
            "monty": {**monty, "feature": feature},
            "details": details,
            "controls": {
                "mode": "interactive" if interactive else "monitor",
                "speed": speed,
                "step_scale": step_scale,
                "step_scale_min": 0.1,
                "step_scale_max": 3.0,
            },
            # Extension lane: new semantic widgets can be appended here without
            # changing the core three-column protocol. The browser resolves each
            # record through its renderer registry.
            "extras": self._memory_extras(),
        }

    def _simulator(
        self,
        observations: Observations,
    ) -> dict[str, Any]:
        sm, sm_id = self.channel_view.simulator_sm()
        agent_obs: AgentObservations = {}
        if sm_id is not None:
            for candidate in observations.values():
                if sm_id in candidate:
                    agent_obs = candidate
                    break
        view_finder = None
        if SensorID("view_finder") in agent_obs:
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
                    rgba,
                    np.array(raw[-1]["pixel_loc"]),
                    patch_size,
                )
            view_finder = _encode_image(rgba)
            view_finder["has_precise_outline"] = bool(has_outline)
        patch = None
        patch_obs = agent_obs.get(sm_id) if sm_id is not None else None
        if patch_obs is not None and "rgba" in patch_obs:
            patch = _encode_image(np.asarray(patch_obs["rgba"]))
        return {
            "sensor_module_id": sm_id,
            "view_finder": view_finder,
            "patch": patch,
        }

    def _training_views(self) -> tuple[dict[str, Any], dict[str, Any]]:
        channels = self.channel_view.lm_channels()
        points = {
            channel: self.channel_view.channel_points(channel)
            for channel in channels
        }
        populated = [channel for channel in channels if points[channel].size]
        if not populated:
            placeholder = _placeholder("no observations yet")
            return placeholder, placeholder
        selected = self.channel_view.channel
        if selected is None or selected not in points or not points[selected].size:
            monty = _placeholder(f"no observations on {selected or 'channel'}")
        else:
            monty = self._channel_visualization(
                selected, points[selected], selected=True
            )
        other_channels = [channel for channel in populated if channel != selected]
        if not other_channels:
            details = _placeholder("no other channels")
        else:
            details = {
                "kind": "channel_stack",
                "items": [
                    {
                        "channel": channel,
                        "visualization": self._channel_visualization(
                            channel,
                            points[channel],
                            selected=False,
                        ),
                        "feature": self._feature(channel),
                    }
                    for channel in other_channels
                ],
            }
        return monty, details

    def _channel_visualization(
        self,
        channel: str,
        points: npt.NDArray[np.float64],
        *,
        selected: bool,
    ) -> dict[str, Any]:
        sm = self.channel_view.resolve_sm_channel(channel)
        if selected and isinstance(sm, TwoDSensorModule):
            return self._planar_buffer(channel, points)
        return {
            "kind": "point_cloud",
            "title": "" if selected else channel,
            "groups": self.channel_view.point_groups(channel, points),
            "projections": True,
            "frame": _frame(points),
        }

    def _planar_buffer(
        self,
        channel: str,
        points: npt.NDArray[np.float64],
    ) -> dict[str, Any]:
        hsv = self.channel_view.aligned_feature(channel, "hsv")
        flags = self.channel_view.aligned_feature(channel, "pose_fully_defined")
        pose = self.channel_view.aligned_feature(channel, "pose_vectors")
        return _planar_visualization(points, hsv, flags, pose, title="")

    def _inference_views(self) -> tuple[dict[str, Any], dict[str, Any]]:
        lm = self.channel_view.lm
        if not isinstance(lm, EvidenceGraphLM):
            message = f"Inference view not available for {type(lm).__name__}"
            placeholder = _placeholder(message)
            return placeholder, placeholder
        monty = self._mlh_view(lm)
        lm_id = lm.learning_module_id
        details = {
            "kind": "inference_history",
            "steps": self.history.steps_by_lm.get(lm_id, []),
            "evidence": _series_json(self.history.evidence_by_lm.get(lm_id, {})),
            "hypotheses": _series_json(self.history.num_hyp_by_lm.get(lm_id, {})),
            "burst_steps": self.history.burst_steps_by_lm.get(lm_id, []),
        }
        return monty, details

    def _mlh_view(self, lm: EvidenceGraphLM) -> dict[str, Any]:
        mlh = current_mlh(lm)
        graph = None
        if mlh and mlh.get("graph_id") not in (None, "no_observations_yet"):
            graph_id = str(mlh["graph_id"])
            if graph_id in lm.graph_memory.get_memory_ids():
                channel = self._mlh_channel(lm, graph_id)
                if channel is not None:
                    graph = lm.graph_memory.get_graph(graph_id, channel)
        if graph is None or getattr(graph, "pos", None) is None:
            return _placeholder("No MLH")
        pos = np.asarray(graph.pos, dtype=float)
        if not len(pos):
            return _placeholder("No MLH")
        threshold = lm.object_evidence_threshold
        evidence = float(mlh.get("evidence", 0.0))
        above_threshold = threshold is None or evidence > float(threshold)
        marker = np.asarray(mlh["location"], dtype=float)
        title = f"MLH ({mlh['graph_id']})"
        if _is_3d(pos):
            # Preserve the current LivePlotter's 3D display convention, which swaps
            # x/y for inference graphs only.
            display_pos = pos[:, [1, 0, 2]]
            display_marker = marker[[1, 0, 2]]
            return {
                "kind": "point_cloud",
                "title": title,
                "groups": [{"label": None, "points": display_pos.tolist()}],
                "marker": {
                    "point": display_marker.tolist(),
                    "above_threshold": above_threshold,
                    "evidence": evidence,
                    "threshold": threshold,
                },
                "projections": False,
                "frame": _frame(display_pos),
            }
        feature_mapping = graph.feature_mapping
        graph_features = {
            name: np.asarray(graph.get_values_for_feature(name), dtype=float)
            for name in ("hsv", "pose_fully_defined", "pose_vectors")
            if name in feature_mapping
        }
        result = _planar_visualization(
            pos,
            graph_features.get("hsv"),
            graph_features.get("pose_fully_defined"),
            graph_features.get("pose_vectors"),
            title=title,
        )
        result["marker"] = {
            "point": marker[:2].tolist(),
            "above_threshold": above_threshold,
            "evidence": evidence,
            "threshold": threshold,
        }
        return result

    def _mlh_channel(self, lm: EvidenceGraphLM, graph_id: str) -> str | None:
        channels = lm.get_input_channels_in_graph(graph_id)
        if self.channel_view.channel in channels:
            return self.channel_view.channel
        sender_types = lm.buffer.channel_sender_types
        sm_channels = [
            channel for channel in channels if sender_types.get(channel) == "SM"
        ]
        return sm_channels[0] if sm_channels else None

    def _feature(self, channel: str | None) -> dict[str, Any]:
        if channel is None:
            return _feature_message("no channel")
        sender_type = self.channel_view.lm.buffer.channel_sender_types.get(channel)
        if sender_type == "LM":
            source_lm = self.channel_view.resolve_lm_channel(channel)
            name = "-"
            if source_lm is not None:
                mlh = current_mlh(source_lm)
                graph_id = mlh.get("graph_id") if mlh else None
                if graph_id and graph_id != "no_observations_yet":
                    name = str(graph_id)
            return {"kind": "object_feature", "name": name}
        if sender_type != "SM":
            return _feature_message("no channel")
        sm = self.channel_view.resolve_sm_channel(channel)
        processed = sm.processed_obs if sm is not None else []
        if not processed:
            return _feature_message("no pose")
        obs = processed[-1]
        morph = obs["morphological_features"]
        non_morph = obs["non_morphological_features"]
        pose_vectors = morph.get("pose_vectors")
        if pose_vectors is None:
            return _feature_message("off object")
        pose = np.asarray(pose_vectors, dtype=float)
        hsv = non_morph.get("hsv")
        color = _rgb_css(_hsv_to_rgb(hsv)) if hsv is not None else "rgb(128 128 128)"
        if isinstance(sm, TwoDSensorModule):
            tangent = np.asarray(pose[1][:2], dtype=float)
            norm = float(np.linalg.norm(tangent))
            defined = bool(morph.get("pose_fully_defined", False)) and norm >= 1e-9
            if defined:
                tangent = tangent / norm
            return {
                "kind": "edge_feature",
                "defined": defined,
                "tangent": tangent.tolist(),
                "color": color,
            }
        rotation = sm.state.rotation if sm.state is not None else None
        if rotation is not None:
            pose = qt.rotate_vectors(rotation.inverse(), pose)
        vectors = [_unit(row) for row in pose[:3]]
        return {
            "kind": "surface_feature",
            "normal": vectors[0].tolist(),
            "tangent_u": vectors[1].tolist(),
            "tangent_v": vectors[2].tolist(),
            "color": color,
        }


def _planar_visualization(
    points: npt.NDArray[np.float64],
    hsv: npt.NDArray[np.float64] | None,
    flags: npt.NDArray[np.float64] | None,
    pose: npt.NDArray[np.float64] | None,
    *,
    title: str,
) -> dict[str, Any]:
    n = len(points)
    if hsv is not None and hsv.shape[0] == n and hsv.shape[1] >= 3:
        colors = [_rgb_css(_hsv_to_rgb(row[:3])) for row in hsv]
    else:
        colors = ["rgb(51 51 51)"] * n
    if flags is not None and flags.shape[0] == n:
        edge_mask = flags[:, 0].astype(bool)
    else:
        edge_mask = np.zeros(n, dtype=bool)
    tangents: list[list[float] | None] = [None] * n
    if edge_mask.any() and pose is not None and pose.shape[0] == n:
        raw = pose[:, 3:5]
        for index in np.flatnonzero(edge_mask):
            tangent = np.asarray(raw[index], dtype=float)
            norm = float(np.linalg.norm(tangent))
            tangents[index] = (tangent / max(norm, 1e-9)).tolist()
    return {
        "kind": "planar_cloud",
        "title": title,
        "points": points[:, :2].tolist(),
        "colors": colors,
        "edge_mask": edge_mask.tolist(),
        "tangents": tangents,
        "frame": _frame(points[:, :2]),
    }


def _frame(points: npt.NDArray[np.float64]) -> dict[str, Any]:
    dims = min(points.shape[1], 3)
    values = points[:, :dims]
    low = values.min(axis=0)
    high = values.max(axis=0)
    center = (low + high) / 2.0
    span = float((high - low).max())
    size = 0.05
    if span > size:
        size += 0.05 * math.ceil((span - 0.05) / 0.05)
    return {"center": center.tolist(), "half": size / 2.0}


def _is_3d(points: npt.NDArray[np.float64]) -> bool:
    return points.shape[1] >= 3 and not np.allclose(points[:, 2], 0.0)


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _hsv_to_rgb(value: object) -> tuple[float, float, float]:
    array = np.clip(np.asarray(value, dtype=float).reshape(-1)[:3], 0.0, 1.0)
    if len(array) < 3:
        return (0.5, 0.5, 0.5)
    return colorsys.hsv_to_rgb(float(array[0]), float(array[1]), float(array[2]))


def _rgb_css(value: Sequence[float]) -> str:
    red, green, blue = [int(round(max(0.0, min(1.0, item)) * 255)) for item in value]
    return f"rgb({red} {green} {blue})"


def _encode_image(array: npt.NDArray[Any]) -> dict[str, Any]:
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return {"width": 0, "height": 0, "rgba_b64": ""}
    if np.issubdtype(image.dtype, np.floating):
        finite_max = float(np.nanmax(image)) if image.size else 1.0
        scale = 255.0 if finite_max <= 1.0 + 1e-6 else 1.0
        image = np.nan_to_num(image * scale, nan=0.0)
    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    if image.shape[2] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    image = np.ascontiguousarray(image)
    return {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "rgba_b64": base64.b64encode(image.tobytes()).decode("ascii"),
    }


def _series_json(
    series: Mapping[str, Sequence[float]],
) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    for name, values in series.items():
        result[str(name)] = [
            None if not np.isfinite(value) else float(value)
            for value in values
        ]
    return result


def _placeholder(message: str) -> dict[str, Any]:
    return {"kind": "placeholder", "message": message}


def _feature_message(message: str) -> dict[str, Any]:
    return {"kind": "feature_message", "message": message}


def _enumish(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)
