# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Small local HTTP/WebSocket server used by the web plotter.

The server deliberately owns no Monty objects. Browser messages are copied into a
thread-safe command bridge and are consumed by the Monty thread at explicit safe
points. This keeps browser/network concurrency out of model state.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import json
from pathlib import Path
import queue
import socket
import threading
from typing import Any
import uuid

from aiohttp import WSMsgType, web


STATIC_DIR = Path(__file__).with_name("static")


class CommandBridge:
    """Thread-safe mailbox between aiohttp's event loop and the Monty thread."""

    def __init__(self) -> None:
        """Initialize an empty command mailbox and default control values."""
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._state_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._speed = 1.0
        self._step_scale = 1.0

    @property
    def speed(self) -> float:
        """Return the latest monitor speed value."""
        with self._state_lock:
            return self._speed

    @property
    def step_scale(self) -> float:
        """Return the latest interactive step multiplier."""
        with self._state_lock:
            return self._step_scale

    def reset(self) -> None:
        """Reset per-episode controls and discard commands from an older episode."""
        with self._state_lock:
            self._speed = 1.0
            self._step_scale = 1.0
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        self._wakeup.set()

    def put(self, message: Mapping[str, Any]) -> None:
        """Accept one validated-enough browser message without touching Monty."""
        kind = str(message.get("type", ""))
        if kind == "set_speed":
            value = _clamp_float(message.get("value"), 0.0, 1.0, self.speed)
            with self._state_lock:
                self._speed = value
            self._wakeup.set()
            return
        if kind == "set_step_scale":
            value = _clamp_float(message.get("value"), 0.1, 3.0, self.step_scale)
            with self._state_lock:
                self._step_scale = value
            self._wakeup.set()
            return
        if kind in {"select_lm", "select_channel", "action"}:
            self._commands.put(dict(message))
            self._wakeup.set()

    def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Take one queued command, returning ``None`` on timeout."""
        try:
            command = self._commands.get(timeout=timeout)
        except queue.Empty:
            return None
        if self._commands.empty():
            self._wakeup.clear()
        return command

    def drain(self) -> list[dict[str, Any]]:
        """Drain currently queued commands without blocking."""
        commands: list[dict[str, Any]] = []
        while True:
            command = self.get(timeout=0.0)
            if command is None:
                return commands
            commands.append(command)

    def wait_for_change(self, timeout: float) -> None:
        """Sleep until any browser control changes or timeout expires."""
        self._wakeup.wait(timeout)
        self._wakeup.clear()


def _clamp_float(value: object, low: float, high: float, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(low, min(high, numeric))


class WebServer:
    """Serve the UI and broadcast semantic snapshots over one WebSocket."""

    def __init__(
        self,
        bridge: CommandBridge,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        """Initialize server state without binding a socket yet.

        Args:
            bridge: Thread-safe mailbox receiving browser control messages.
            host: Interface on which the local UI server listens.
            port: TCP port, or ``0`` to reserve an ephemeral port.
        """
        self.bridge = bridge
        self.host = host
        self.port = port
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._controller: web.WebSocketResponse | None = None
        self._latest_lock = threading.Lock()
        self._latest_frame: dict[str, Any] | None = None
        self._latest_action: dict[str, Any] | None = None
        self._frame_flush_scheduled = False

    @property
    def url(self) -> str:
        """Return the browser URL after ``start`` succeeds."""
        return f"http://{self.host}:{self.port}/"

    def start(self) -> None:
        """Start aiohttp in a daemon thread and wait until its socket is bound."""
        if self._thread is not None:
            return
        if self.port == 0:
            self.port = _reserve_ephemeral_port(self.host)
        self._thread = threading.Thread(
            target=self._thread_main,
            name="tbp-teleop-web",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "Could not start tbp.teleop web server"
            ) from self._startup_error

    def publish(self, message: dict[str, Any]) -> None:
        """Store and broadcast state without letting slow browsers backlog Monty."""
        loop = self._loop
        kind = message.get("type")
        schedule_frame = False
        with self._latest_lock:
            if kind == "frame":
                self._latest_frame = message
                if not self._frame_flush_scheduled:
                    self._frame_flush_scheduled = True
                    schedule_frame = True
            elif kind == "action_request":
                self._latest_action = message
            elif kind == "action_resolved":
                if (
                    self._latest_action is not None
                    and self._latest_action.get("request_id")
                    == message.get("request_id")
                ):
                    self._latest_action = None
        if loop is None or not loop.is_running():
            return
        if kind == "frame":
            if schedule_frame:
                asyncio.run_coroutine_threadsafe(self._flush_latest_frames(), loop)
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), loop)

    def stop(self) -> None:
        """Shut down clients and aiohttp; safe to call more than once."""
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            try:
                future.result(timeout=2.0)
            except (TimeoutError, RuntimeError):
                pass
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2.0)
        self._thread = None
        self._loop = None

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._startup())
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                loop.run_until_complete(self._shutdown())
                loop.close()

    async def _startup(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/healthz", self._health)
        app.router.add_get("/ws", self._websocket)
        app.router.add_static("/static/", STATIC_DIR, show_index=False)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    async def _shutdown(self) -> None:
        clients = list(self._clients)
        self._clients.clear()
        self._controller = None
        for ws in clients:
            if not ws.closed:
                await ws.close(code=1001, message=b"teleop server stopped")
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _index(self, request: web.Request) -> web.FileResponse:  # noqa: ARG002
        return web.FileResponse(STATIC_DIR / "index.html")

    async def _health(self, request: web.Request) -> web.Response:  # noqa: ARG002
        return web.json_response({"ok": True, "protocol": 1})

    async def _websocket(self, request: web.Request) -> web.WebSocketResponse:
        origin = request.headers.get("Origin")
        allowed_origins = {f"http://{request.host}", f"https://{request.host}"}
        if origin is not None and origin not in allowed_origins:
            raise web.HTTPForbidden(text="WebSocket origin does not match teleop UI")

        ws = web.WebSocketResponse(heartbeat=15.0, max_msg_size=2 * 1024 * 1024)
        await ws.prepare(request)
        client_id = uuid.uuid4().hex[:8]
        self._clients.add(ws)
        if self._controller is None:
            self._controller = ws
        role = "controller" if ws is self._controller else "observer"
        await ws.send_json(
            {
                "type": "hello",
                "client_id": client_id,
                "protocol": 1,
                "role": role,
            }
        )
        with self._latest_lock:
            latest_frame = self._latest_frame
            latest_action = self._latest_action
        if latest_frame is not None:
            await ws.send_json(latest_frame)
        if latest_action is not None:
            await ws.send_json(latest_action)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "message": "Invalid JSON"})
                        continue
                    if not isinstance(payload, dict):
                        continue
                    if ws is not self._controller:
                        await ws.send_json(
                            {
                                "type": "error",
                                "message": (
                                    "This browser is observing; another tab "
                                    "controls Monty."
                                ),
                            }
                        )
                        continue
                    self.bridge.put(payload)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
            if ws is self._controller:
                self._controller = next(iter(self._clients), None)
                if self._controller is not None and not self._controller.closed:
                    await self._controller.send_json(
                        {"type": "role", "role": "controller"}
                    )
        return ws

    async def _flush_latest_frames(self) -> None:
        """Send only the newest frame when rendering falls behind production."""
        while True:
            with self._latest_lock:
                frame = self._latest_frame
            if frame is not None:
                await self._broadcast(frame)
            with self._latest_lock:
                if frame is self._latest_frame:
                    self._frame_flush_scheduled = False
                    return

    async def _broadcast(self, message: dict[str, Any]) -> None:
        dead: list[web.WebSocketResponse] = []
        for ws in tuple(self._clients):
            if ws.closed:
                dead.append(ws)
                continue
            try:
                await ws.send_json(message)
            except (ConnectionResetError, RuntimeError):
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)


def _reserve_ephemeral_port(host: str) -> int:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
