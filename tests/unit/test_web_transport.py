# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from tbp.teleop.web.server import CommandBridge
from tbp.teleop.web.snapshot import WebChannelView, current_mlh
from tbp.teleop.web_plotter import WebPlotter

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty


def test_command_bridge_keeps_continuous_controls_out_of_command_queue() -> None:
    bridge = CommandBridge()
    bridge.put({"type": "set_speed", "value": 0.25})
    bridge.put({"type": "set_step_scale", "value": 2.5})

    assert bridge.speed == 0.25
    assert bridge.step_scale == 2.5
    assert bridge.get(timeout=0.0) is None


def test_shared_location_buffer_uses_channel_validity_mask() -> None:
    buffer = SimpleNamespace(
        locations=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        channel_sender_types={"patch": "SM"},
        features={"patch": {"on_object": np.array([[1.0], [np.nan], [1.0]])}},
    )
    lm = SimpleNamespace(learning_module_id="learning_module_0", buffer=buffer)
    model = SimpleNamespace(learning_modules=[lm], sensor_modules=[])
    view = WebChannelView(cast("Monty", model))

    np.testing.assert_array_equal(
        view.channel_points("patch"),
        np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]),
    )


def test_legacy_per_channel_location_buffer_remains_supported() -> None:
    buffer = SimpleNamespace(
        locations={
            "patch": np.array(
                [[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan], [7.0, 8.0, 9.0]]
            )
        },
        channel_sender_types={"patch": "SM"},
        features={"patch": {}},
    )
    lm = SimpleNamespace(learning_module_id="learning_module_0", buffer=buffer)
    model = SimpleNamespace(learning_modules=[lm], sensor_modules=[])
    view = WebChannelView(cast("Monty", model))

    np.testing.assert_array_equal(
        view.channel_points("patch"),
        np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]),
    )


def test_current_mlh_prefers_new_private_accessor() -> None:
    lm = SimpleNamespace(
        _get_current_mlh=lambda: {"graph_id": "new"},
        get_current_mlh=lambda: {"graph_id": "old"},
    )
    assert current_mlh(lm) == {"graph_id": "new"}


def test_web_plotter_keeps_existing_speed_mapping() -> None:
    plotter = WebPlotter(min_delay=0.001, max_delay=2.0)
    assert plotter._pause_seconds(0.0) is None
    assert plotter._pause_seconds(1.0) == pytest.approx(0.001)
    assert plotter._pause_seconds(0.5) == pytest.approx(1.0005)
