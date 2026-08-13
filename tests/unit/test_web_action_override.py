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

from tbp.teleop.web.server import WebServer
from tbp.teleop.web_plotter import WebPlotter

if TYPE_CHECKING:
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.actions.actions import Action
    from tbp.monty.frameworks.models.abstract_monty_classes import Monty
    from tbp.teleop.policies import InteractivePolicy


class _Server:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def publish(self, message: dict[str, object]) -> None:
        self.messages.append(message)


class _Policy:
    def __init__(self) -> None:
        self.feedback_value: object | None = None
        self.compute_args: tuple[object, str, object, float] | None = None

    def compute(
        self, ctx: object, name: str, state: object, scale: float
    ) -> list[str]:
        self.compute_args = (ctx, name, state, scale)
        return [f"chosen:{name}:{scale}"]

    def feedback(self, chosen: object) -> None:
        self.feedback_value = chosen


def test_override_replaces_logged_proposal_and_feeds_back() -> None:
    plotter = WebPlotter(interactive=True)
    server = _Server()
    policy = _Policy()
    motor_state = object()
    model = SimpleNamespace(
        motor_system=SimpleNamespace(action_sequence=[(["proposed"], motor_state)]),
        deal_with_time_out=lambda: None,
    )
    plotter.model = cast("Monty", model)
    plotter._server = cast("WebServer", server)
    plotter._policy = cast("InteractivePolicy", policy)
    plotter._last_step = 7
    plotter._bridge.put({"type": "set_step_scale", "value": 1.5})
    plotter._bridge.put({"type": "action", "request_id": "7:1", "name": "left"})

    ctx = cast("RuntimeContext", SimpleNamespace(rng=object()))
    proposed = cast("list[Action]", ["proposed"])
    chosen = plotter.override_action(ctx, proposed)

    assert chosen == ["chosen:left:1.5"]
    assert policy.feedback_value == chosen
    assert model.motor_system.action_sequence[-1] == (chosen, motor_state)


def test_stale_action_request_is_ignored() -> None:
    plotter = WebPlotter(interactive=True)
    server = _Server()
    policy = _Policy()
    motor_state = object()
    model = SimpleNamespace(
        motor_system=SimpleNamespace(action_sequence=[(["proposed"], motor_state)]),
        deal_with_time_out=lambda: None,
    )
    plotter.model = cast("Monty", model)
    plotter._server = cast("WebServer", server)
    plotter._policy = cast("InteractivePolicy", policy)
    plotter._last_step = 9
    plotter._bridge.put({"type": "action", "request_id": "8:1", "name": "right"})
    plotter._bridge.put({"type": "action", "request_id": "9:1", "name": "up"})

    ctx = cast("RuntimeContext", SimpleNamespace(rng=object()))
    proposed = cast("list[Action]", ["proposed"])
    chosen = plotter.override_action(ctx, proposed)

    assert chosen == ["chosen:up:1.0"]
