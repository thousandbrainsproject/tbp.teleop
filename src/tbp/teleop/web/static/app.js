const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const SVG_NS = "http://www.w3.org/2000/svg";

const el = {
  lmSelect: $("#lm-select"),
  channelSelect: $("#channel-select"),
  experimentMode: $("#experiment-mode"),
  phasePill: $("#phase-pill"),
  stepValue: $("#step-value"),
  connection: $("#connection"),
  sensorSource: $("#sensor-source"),
  outlineBadge: $("#outline-badge"),
  viewFinder: $("#view-finder"),
  viewEmpty: $("#view-empty"),
  rgbPatch: $("#rgb-patch"),
  patchEmpty: $("#patch-empty"),
  choiceNote: $("#choice-note"),
  montyTitle: $("#monty-title"),
  montyCaption: $("#monty-caption"),
  montyViz: $("#monty-viz"),
  montyFeature: $("#monty-feature"),
  detailsViz: $("#details-viz"),
  extras: $("#extras"),
  resetCamera: $("#reset-camera"),
  monitorControls: $("#monitor-controls"),
  interactiveControls: $("#interactive-controls"),
  controlMode: $("#control-mode"),
  controlDescription: $("#control-description"),
  pauseToggle: $("#pause-toggle"),
  speedSlider: $("#speed-slider"),
  speedOutput: $("#speed-output"),
  stepSlider: $("#step-slider"),
  stepOutput: $("#step-output"),
  jumpButton: $("#jump-button"),
  endButton: $("#end-button"),
  toast: $("#toast"),
};

const state = {
  socket: null,
  frame: null,
  actionRequest: null,
  reconnectTimer: null,
  camera: new Map(),
  lastNonzeroSpeed: 1,
  role: "unknown",
  memoryMerge: null,
};

const rendererRegistry = new Map();

function registerRenderer(kind, mount) {
  rendererRegistry.set(kind, mount);
}

window.TBPTeleop = Object.freeze({ registerRenderer });

function connect() {
  clearTimeout(state.reconnectTimer);
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  state.socket = socket;
  setConnection(false, "Connecting");

  socket.addEventListener("open", () => setConnection(true, "Live"));
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    handleMessage(message);
  });
  socket.addEventListener("close", () => {
    setConnection(false, "Reconnecting");
    if (state.socket === socket) {
      state.reconnectTimer = setTimeout(connect, 800);
    }
  });
  socket.addEventListener("error", () => socket.close());
}

function handleMessage(message) {
  if (message.type === "hello" || message.type === "role") {
    state.role = message.role || state.role;
    setConnection(true, state.role === "controller" ? "Live" : "Observer");
    if (state.frame) renderFrame(state.frame);
    renderActionState();
    return;
  }
  if (message.type === "error") {
    toast(message.message || "Browser command rejected");
    return;
  }
  if (message.type === "frame") {
    state.frame = message;
    renderFrame(message);
    return;
  }
  if (message.type.startsWith("memory_merge_")) {
    handleMemoryMerge(message);
    return;
  }
  if (message.type === "action_request") {
    state.actionRequest = message;
    renderActionState();
    return;
  }
  if (message.type === "action_resolved") {
    if (state.actionRequest?.request_id === message.request_id) {
      state.actionRequest = null;
      renderActionState();
    }
    return;
  }
  if (message.type === "session_closed") {
    setConnection(false, "Closed");
    state.actionRequest = null;
    renderActionState();
    toast("Monty session closed");
  }
}

function send(message) {
  if (state.role !== "controller") return;
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message));
  }
}

function renderFrame(frame) {
  const { session, selectors, simulator, monty, details, controls } = frame;
  el.stepValue.textContent = session.step;
  el.experimentMode.textContent = `${session.experiment_mode} · ${session.interactive ? "interactive" : "monitor"}`;
  renderPhase(session.phase);
  renderSelectors(selectors);
  renderSimulator(simulator);

  el.montyTitle.textContent = monty.title || titleForKind(monty.kind);
  el.montyCaption.textContent = session.phase === "exploring"
    ? "Live learning-module buffer"
    : "Current most-likely hypothesis";
  renderVisualization(el.montyViz, monty, `monty:${selectors.active_lm}:${selectors.active_channel}`);
  renderFeature(el.montyFeature, monty.feature);
  renderDetails(details);
  renderExtras(frame.extras || []);
  renderControls(controls);
  renderActionState();
}

function renderPhase(phase) {
  el.phasePill.className = "status-pill";
  if (phase === "exploring") {
    el.phasePill.classList.add("live");
    el.phasePill.querySelector("span").textContent = "Exploring";
  } else if (phase === "matching") {
    el.phasePill.classList.add("matching");
    el.phasePill.querySelector("span").textContent = "Matching";
  } else {
    el.phasePill.querySelector("span").textContent = phase || "Idle";
  }
}

function renderSelectors(selectors) {
  syncSelect(
    el.lmSelect,
    selectors.learning_modules.map((item) => ({ value: item.id, label: `${item.id} · ${item.type}` })),
    selectors.active_lm,
  );
  syncSelect(
    el.channelSelect,
    selectors.channels.map((item) => ({ value: item.id, label: `${item.id} · ${item.sender_type || "?"}` })),
    selectors.active_channel,
  );
}

function syncSelect(select, options, value) {
  const signature = JSON.stringify(options);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...options.map((item) => {
      const option = document.createElement("option");
      option.value = item.value;
      option.textContent = item.label;
      return option;
    }));
    select.dataset.signature = signature;
  }
  select.disabled = options.length === 0 || state.role !== "controller";
  if (value != null) select.value = value;
}

function renderSimulator(simulator) {
  el.sensorSource.textContent = simulator.sensor_module_id || "No sensor selected";
  drawRawImage(el.viewFinder, simulator.view_finder, el.viewEmpty);
  drawRawImage(el.rgbPatch, simulator.patch, el.patchEmpty);
  const precise = simulator.view_finder?.has_precise_outline;
  el.outlineBadge.textContent = precise ? "tracked" : "fallback";
  el.outlineBadge.classList.toggle("live", Boolean(precise));
}

function drawRawImage(canvas, image, empty) {
  if (!image?.width || !image?.height || !image?.rgba_b64) {
    empty.hidden = false;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  empty.hidden = true;
  const bytes = base64Bytes(image.rgba_b64);
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.putImageData(new ImageData(new Uint8ClampedArray(bytes.buffer), image.width, image.height), 0, 0);
  if (image.has_precise_outline === false) {
    ctx.save();
    ctx.strokeStyle = "white";
    ctx.lineWidth = Math.max(1, Math.round(image.width / 300));
    ctx.strokeRect(image.width * .45, image.height * .45, image.width * .1, image.height * .1);
    ctx.restore();
  }
}

function base64Bytes(value) {
  const raw = atob(value);
  const result = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) result[i] = raw.charCodeAt(i);
  return result;
}

function renderVisualization(host, record, key) {
  const kind = record?.kind || "placeholder";
  const mount = rendererRegistry.get(kind) || rendererRegistry.get("placeholder");
  if (host._rendererKind !== kind || !host._rendererHandle) {
    host._rendererHandle?.cleanup?.();
    host.replaceChildren();
    host._rendererKind = kind;
    host._rendererHandle = mount(host, record, key);
  } else {
    host._rendererHandle.update(record, key);
  }
}

registerRenderer("placeholder", (host, record) => {
  const node = document.createElement("div");
  node.className = "placeholder";
  host.append(node);
  const update = (next) => { node.textContent = next.message || "No data"; };
  update(record);
  return { update };
});

registerRenderer("point_cloud", (host, initial, key) => {
  const shell = document.createElement("div");
  shell.className = "cloud-shell";

  const wrap = document.createElement("div");
  wrap.className = "canvas-wrap";

  const canvas = document.createElement("canvas");
  wrap.append(canvas);

  const legend = document.createElement("div");
  legend.className = "legend";
  wrap.append(legend);

  const projectionRow = document.createElement("div");
  projectionRow.className = "projection-row";

  const projections = ["XY", "XZ", "YZ"].map((name) => {
    const node = document.createElement("div");
    node.className = "projection";

    const child = document.createElement("canvas");

    const label = document.createElement("span");
    label.textContent = name;

    node.append(child, label);
    projectionRow.append(node);

    return child;
  });

  shell.append(wrap, projectionRow);
  host.append(shell);

  let data = initial;
  let cameraKey = key;
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  let camera = cameraFor(cameraKey);

  /*
   * ------------------------------------------------------------
   * Memory thumbnail auto-rotation
   * ------------------------------------------------------------
   *
   * Only Memory point clouds rotate automatically.
   *
   * Other users of this renderer -- such as the main Monty panel
   * and Details point clouds -- continue behaving exactly as they
   * did before.
   */
  const AUTO_ROTATE_SPEED = 0.08;
  const AUTO_ROTATE_IDLE_MS = 900;

  let animationFrame = null;
  let previousTime = performance.now();
  let lastInteractionTime = previousTime;

  const isMemoryThumbnail = () =>
    cameraKey.startsWith("extra:memory:");

  const noteInteraction = () => {
    lastInteractionTime = performance.now();
  };

  const draw = () => {
    drawCloud(canvas, data, camera);

    projectionRow.hidden = !data.projections;

    if (data.projections) {
      drawProjection(
        projections[0],
        data,
        0,
        1,
      );

      drawProjection(
        projections[1],
        data,
        0,
        2,
      );

      drawProjection(
        projections[2],
        data,
        1,
        2,
      );
    }

    renderLegend(
      legend,
      data.groups || [],
    );
  };

  /*
   * Run a lightweight animation loop for this renderer.
   *
   * The camera yaw is changed rather than the underlying points,
   * so the stored object geometry remains untouched.
   */
  const animate = (now) => {
    /*
     * Convert elapsed time to seconds.
     *
     * Clamp large jumps so switching browser tabs does not cause
     * the object to suddenly rotate a large amount when returning.
     */
    const dt = Math.min(
      (now - previousTime) / 1000,
      0.05,
    );

    previousTime = now;

    const idleLongEnough =
      now - lastInteractionTime
      >= AUTO_ROTATE_IDLE_MS;

    /*
     * Auto-rotate only when:
     *
     *   1. this renderer belongs to a Memory thumbnail,
     *   2. the user is not currently dragging it,
     *   3. no Memory merge animation is active,
     *   4. enough time has passed since manual interaction,
     *   5. the browser tab is visible.
     */
    const shouldRotate =
      isMemoryThumbnail()
      && !dragging
      && !state.memoryMerge
      && idleLongEnough
      && !document.hidden;

    if (shouldRotate) {
      camera.yaw +=
        AUTO_ROTATE_SPEED * dt;

      draw();
    }

    animationFrame =
      requestAnimationFrame(animate);
  };

  const resize =
    new ResizeObserver(draw);

  resize.observe(wrap);
  resize.observe(projectionRow);

  /*
   * ------------------------------------------------------------
   * Manual camera interaction
   * ------------------------------------------------------------
   *
   * Touching/dragging the model immediately stops auto-rotation.
   */
  canvas.addEventListener(
    "pointerdown",
    (event) => {
      dragging = true;
      noteInteraction();

      lastX = event.clientX;
      lastY = event.clientY;

      canvas.setPointerCapture(
        event.pointerId,
      );
    },
  );

  canvas.addEventListener(
    "pointermove",
    (event) => {
      if (!dragging) return;

      noteInteraction();

      camera.yaw +=
        (event.clientX - lastX)
        * 0.008;

      camera.pitch = clamp(
        camera.pitch
          + (event.clientY - lastY)
          * 0.008,
        -1.45,
        1.45,
      );

      lastX = event.clientX;
      lastY = event.clientY;

      draw();
    },
  );

  canvas.addEventListener(
    "pointerup",
    () => {
      dragging = false;
      noteInteraction();
    },
  );

  canvas.addEventListener(
    "pointercancel",
    () => {
      dragging = false;
      noteInteraction();
    },
  );

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      noteInteraction();

      camera.zoom = clamp(
        camera.zoom
          * Math.exp(
            -event.deltaY * 0.001,
          ),
        0.55,
        2.5,
      );

      draw();
    },
    {
      passive: false,
    },
  );

  /*
   * Normal renderer update.
   *
   * If this renderer begins displaying a different object, switch
   * to that object's persistent camera just as before.
   */
  const update = (
    next,
    nextKey,
  ) => {
    data = next;

    if (nextKey !== cameraKey) {
      cameraKey = nextKey;
      camera = cameraFor(cameraKey);

      /*
       * Treat switching objects as an interaction so a newly
       * displayed object rests briefly before beginning to turn.
       */
      noteInteraction();
    }

    draw();
  };

  /*
   * Reset the view to Teleop's normal default camera.
   *
   * Auto-rotation waits briefly after reset before resuming.
   */
  const reset = () => {
    camera.yaw = -0.72;
    camera.pitch = 0.52;
    camera.zoom = 1;

    noteInteraction();
    draw();
  };

  /*
   * Draw immediately, then start the idle animation loop.
   */
  draw();

  animationFrame =
    requestAnimationFrame(animate);

  return {
    update,
    reset,

    cleanup() {
      resize.disconnect();

      /*
       * Important when a Memory card disappears or its renderer
       * gets replaced: don't leave an orphaned RAF loop running.
       */
      if (animationFrame !== null) {
        cancelAnimationFrame(
          animationFrame,
        );

        animationFrame = null;
      }
    },
  };
});


registerRenderer("planar_cloud", (host, initial) => {
  const shell = document.createElement("div");
  shell.className = "planar-shell";
  const wrap = document.createElement("div");
  wrap.className = "canvas-wrap";
  const canvas = document.createElement("canvas");
  wrap.append(canvas);
  shell.append(wrap);
  host.append(shell);
  let data = initial;
  const draw = () => drawPlanar(canvas, data);
  const resize = new ResizeObserver(draw);
  resize.observe(wrap);
  draw();
  return {
    update(next) { data = next; draw(); },
    reset: draw,
    cleanup() { resize.disconnect(); },
  };
});

function cameraFor(key) {
  if (!state.camera.has(key)) {
    state.camera.set(key, { yaw: -0.72, pitch: 0.52, zoom: 1 });
  }
  return state.camera.get(key);
}

function drawMemoryMergeFlyer(
  canvas,
  points,
  colors,
  merge,
  seriesIndex,
  frame,
  camera,
) {
  const {
    ctx,
    width,
    height,
    dpr,
  } = canvasContext(canvas);

  ctx.clearRect(
    0,
    0,
    width,
    height,
  );

  if (!points?.length) return;

  if (merge.is_3d) {
    drawMemoryMergeFlyer3D(
      ctx,
      width,
      height,
      dpr,
      points,
      colors,
      frame,
      camera,
      seriesIndex,
    );
  } else {
    drawMemoryMergeFlyer2D(
      ctx,
      width,
      height,
      dpr,
      points,
      colors,
      frame,
      seriesIndex,
    );
  }
}


function drawMemoryMergeFlyer3D(
  ctx,
  width,
  height,
  dpr,
  points,
  colors,
  frame,
  camera,
  seriesIndex,
) {
  const center = pad3(
    frame.center || [0, 0, 0],
  );

  const half = Math.max(
    Number(frame.half) || 0.025,
    1e-9,
  );

  const baseScale =
    Math.min(width, height)
    * 0.34
    * camera.zoom;

  const project = (point) =>
    project3D(
      point,
      center,
      half,
      camera,
      width,
      height,
      baseScale,
    );

  const marks = points.map(
    (point, pointIndex) => {
      const projected = project(point);

      return {
        ...projected,
        color:    
          colors?.[pointIndex]
          || seriesColor(seriesIndex),
      };
    },
  );

  marks.sort(
    (a, b) => a.depth - b.depth,
  );

  for (const mark of marks) {
    ctx.beginPath();

    ctx.arc(
      mark.x,
      mark.y,
      2.2 * dpr,
      0,
      Math.PI * 2,
    );

    ctx.fillStyle = mark.color;
    ctx.globalAlpha = 1;
    ctx.fill();
  }

  ctx.globalAlpha = 1;
}

function drawMemoryMergeFlyer2D(
  ctx,
  width,
  height,
  dpr,
  points,
  colors,
  frame,
  seriesIndex,
) {
  const center =
    frame.center || [0, 0];

  const half = Math.max(
    Number(frame.half) || 0.025,
    1e-9,
  );

  const scale =
    Math.min(width, height)
    * 0.42
    / half;

  for (
    let pointIndex = 0;
    pointIndex < points.length;
    pointIndex += 1
  ) {
    const point = points[pointIndex];

    const x =
      width / 2
      + (point[0] - center[0])
      * scale;

    const y =
      height / 2
      - (point[1] - center[1])
      * scale;

    ctx.beginPath();

    ctx.arc(
      x,
      y,
      2.3 * dpr,
      0,
      Math.PI * 2,
    );

    ctx.fillStyle =
      colors?.[pointIndex]
      || seriesColor(seriesIndex);

    ctx.globalAlpha = 1;
    ctx.fill();
  }

  ctx.globalAlpha = 1;
}

function drawCloud(canvas, data, camera) {
  const { ctx, width, height, dpr } = canvasContext(canvas);
  ctx.clearRect(0, 0, width, height);
  const frame = data.frame || inferFrame(flatPoints(data.groups || []));
  const center = pad3(frame.center || [0, 0, 0]);
  const half = Math.max(Number(frame.half) || 0.025, 1e-9);
  const baseScale = Math.min(width, height) * 0.34 * camera.zoom;
  const project = (point) => project3D(point, center, half, camera, width, height, baseScale);

  drawCube(ctx, center, half, project);
  const marks = [];
  (data.groups || []).forEach((group, groupIndex) => {
    (group.points || []).forEach((point, pointIndex) => {
      const projected = project(point);
      marks.push({
        ...projected,
        color: group.colors?.[pointIndex] || seriesColor(group.series_index ?? groupIndex),
        radius: 2.2 * dpr,
        alpha: Number(group.alpha ?? 0.9),
      });
    });
  });
  marks.sort((a, b) => a.depth - b.depth);
  for (const mark of marks) {
    ctx.beginPath();
    ctx.arc(mark.x, mark.y, mark.radius, 0, Math.PI * 2);
    ctx.fillStyle = mark.color;
    ctx.globalAlpha = mark.alpha;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (data.marker?.point) {
    const marker = project(data.marker.point);
    const color = data.marker.above_threshold ? css("--danger") : css("--muted");
    drawCross(ctx, marker.x, marker.y, 7 * dpr, color, 2 * dpr);
  }
}

function drawCube(ctx, center, half, project) {
  const corners = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
    corners.push([center[0] + x * half, center[1] + y * half, center[2] + z * half]);
  }
  const edges = [
    [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7],
  ];
  ctx.strokeStyle = "rgba(180, 199, 210, .13)";
  ctx.lineWidth = 1;
  for (const [a, b] of edges) {
    const p = project(corners[a]);
    const q = project(corners[b]);
    ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  }
}

function project3D(point, center, half, camera, width, height, scale) {
  const p = pad3(point);
  let x = (p[0] - center[0]) / half;
  let y = (p[1] - center[1]) / half;
  let z = (p[2] - center[2]) / half;
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw);
  const x1 = cy * x + sy * z;
  const z1 = -sy * x + cy * z;
  const cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const y2 = cp * y - sp * z1;
  const z2 = sp * y + cp * z1;
  return { x: width / 2 + x1 * scale, y: height / 2 - y2 * scale, depth: z2 };
}

function drawProjection(canvas, data, a, b) {
  const { ctx, width, height, dpr } = canvasContext(canvas);
  ctx.clearRect(0, 0, width, height);
  const frame = data.frame || inferFrame(flatPoints(data.groups || []));
  const center = pad3(frame.center || [0,0,0]);
  const half = Math.max(Number(frame.half) || .025, 1e-9);
  const scale = Math.min(width, height) * .39 / half;
  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.strokeRect(width/2-half*scale, height/2-half*scale, half*2*scale, half*2*scale);
  (data.groups || []).forEach((group, groupIndex) => {
    (group.points || []).forEach((point, pointIndex) => {
      const x = width/2 + (point[a] - center[a]) * scale;
      const y = height/2 - (point[b] - center[b]) * scale;
      ctx.beginPath(); ctx.arc(x, y, 1.6*dpr, 0, Math.PI*2);
      ctx.fillStyle = group.colors?.[pointIndex] || seriesColor(group.series_index ?? groupIndex);
      ctx.fill();
    });
  });
}

function drawPlanar(canvas, data) {
  const { ctx, width, height, dpr } = canvasContext(canvas);
  ctx.clearRect(0, 0, width, height);

  // Normal planar records have data.points.
  // Merge-animation records have data.groups.
  const grouped = Array.isArray(data.groups);

  const groups = grouped
    ? data.groups
    : [{
        points: data.points || [],
        colors: data.colors,
        edge_mask: data.edge_mask,
        tangents: data.tangents,
        alpha: 1,
        useMutedDefault: true,
      }];

  const points = flatPoints(groups);

  if (!points.length) return;

  const frame = data.frame || inferFrame(points);
  const center = frame.center;
  const half = Math.max(Number(frame.half) || 0.025, 1e-9);
  const scale = Math.min(width, height) * 0.42 / half;

  const project = (point) => [
    width / 2 + (point[0] - center[0]) * scale,
    height / 2 - (point[1] - center[1]) * scale,
  ];

  ctx.strokeStyle = "rgba(255,255,255,.08)";
  ctx.lineWidth = 1;
  ctx.strokeRect(
    width / 2 - half * scale,
    height / 2 - half * scale,
    half * 2 * scale,
    half * 2 * scale,
  );

  groups.forEach((group, groupIndex) => {
    (group.points || []).forEach((point, index) => {
      const [x, y] = project(point);

      const color =
        group.colors?.[index]
        || (
          group.useMutedDefault
            ? css("--muted")
            : seriesColor(group.series_index ?? groupIndex)
        );

      const tangent = group.tangents?.[index];

      ctx.globalAlpha = Number(group.alpha ?? 1);

      if (group.edge_mask?.[index] && tangent) {
        const length = 0.04 * half * scale;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.6 * dpr;
        ctx.setLineDash([4 * dpr, 3 * dpr]);

        ctx.beginPath();
        ctx.moveTo(
          x - tangent[0] * length,
          y + tangent[1] * length,
        );
        ctx.lineTo(
          x + tangent[0] * length,
          y - tangent[1] * length,
        );
        ctx.stroke();

        ctx.restore();
      } else {
        ctx.beginPath();
        ctx.arc(
          x,
          y,
          2.3 * dpr,
          0,
          Math.PI * 2,
        );
        ctx.fillStyle = color;
        ctx.fill();
      }

      ctx.globalAlpha = 1;
    });
  });

  if (data.marker?.point) {
    const [x, y] = project(data.marker.point);

    drawCross(
      ctx,
      x,
      y,
      7 * dpr,
      data.marker.above_threshold
        ? css("--danger")
        : css("--muted"),
      2 * dpr,
    );
  }
}

function renderLegend(host, groups) {
  const labeled = groups.filter((group) => group.label);
  host.replaceChildren(...labeled.map((group, index) => {
    const item = document.createElement("span");
    item.className = "legend-item";
    const dot = document.createElement("i");
    dot.style.background = seriesColor(group.series_index ?? index);
    const label = document.createElement("span");
    label.textContent = group.label;
    item.append(dot, label);
    return item;
  }));
}

function renderFeature(host, feature) {
  host.replaceChildren();
  if (!feature || feature.kind === "feature_message") {
    const node = document.createElement("div");
    node.className = "feature-message";
    node.textContent = feature?.message || "No feature";
    host.append(node);
    return;
  }
  if (feature.kind === "object_feature") {
    const node = document.createElement("div");
    node.className = "object-feature";
    node.innerHTML = `Source LM<strong>${escapeHtml(feature.name || "-")}</strong>`;
    host.append(node);
    return;
  }
  if (feature.kind === "edge_feature") {
    host.append(edgeFeatureSvg(feature));
    return;
  }
  if (feature.kind === "surface_feature") {
    host.append(surfaceFeatureSvg(feature));
    return;
  }
  const unknown = document.createElement("div");
  unknown.className = "feature-message";
  unknown.textContent = feature.kind;
  host.append(unknown);
}

function edgeFeatureSvg(feature) {
  const svg = svgNode("svg", { viewBox: "0 0 120 86", role: "img", "aria-label": "Detected 2D edge" });
  if (!feature.defined) {
    const text = svgNode("text", { x: 60, y: 45, "text-anchor": "middle", fill: css("--muted"), "font-size": 10 });
    text.textContent = "No edge detected";
    svg.append(text);
    return svg;
  }
  const [tx, ty] = feature.tangent || [1, 0];
  const length = 34;
  svg.append(svgNode("line", {
    x1: 60-tx*length, y1: 43+ty*length,
    x2: 60+tx*length, y2: 43-ty*length,
    stroke: feature.color || css("--accent"), "stroke-width": 4, "stroke-linecap": "round",
  }));
  return svg;
}

function surfaceFeatureSvg(feature) {
  const svg = svgNode("svg", { viewBox: "0 0 120 90", role: "img", "aria-label": "Local 3D surface and normal" });
  const u = feature.tangent_u || [1,0,0];
  const v = feature.tangent_v || [0,1,0];
  const normal = feature.normal || [0,0,1];
  const project = (p) => [60 + (p[0] - p[1]) * 27, 48 + (p[0]+p[1]) * 10 - p[2]*24];
  const corners = [[1,1], [1,-1], [-1,-1], [-1,1]].map(([a,b]) => project([
    .55*(a*u[0]+b*v[0]), .55*(a*u[1]+b*v[1]), .55*(a*u[2]+b*v[2]),
  ]));
  svg.append(svgNode("polygon", {
    points: corners.map((p) => p.join(",")).join(" "), fill: feature.color || css("--accent"),
    "fill-opacity": .38, stroke: "rgba(255,255,255,.45)", "stroke-width": 1,
  }));
  const start = project([0,0,0]);
  const end = project(normal.map((n) => n*.85));
  svg.append(svgNode("line", { x1:start[0], y1:start[1], x2:end[0], y2:end[1], stroke: css("--text"), "stroke-width":2.5, "stroke-linecap":"round" }));
  svg.append(svgNode("circle", { cx:end[0], cy:end[1], r:3.5, fill:css("--text") }));
  return svg;
}

function renderDetails(details) {
  if (!details) return;
  if (el.detailsViz.dataset.kind !== details.kind) {
    el.detailsViz._detailsCleanup?.();
    el.detailsViz.replaceChildren();
    el.detailsViz.dataset.kind = details.kind;
    if (details.kind === "inference_history") mountInferenceDetails(details);
    else if (details.kind === "channel_stack") mountChannelDetails(details);
    else mountDetailsPlaceholder(details);
    return;
  }
  if (details.kind === "inference_history") updateInferenceDetails(details);
  else if (details.kind === "channel_stack") updateChannelDetails(details);
  else mountDetailsPlaceholder(details);
}

function mountInferenceDetails(details) {
  const panel = document.createElement("article");
  panel.className = "panel inference-panel";
  for (const [id, title] of [["evidence", "Highest evidence per object"], ["hypotheses", "Number of hypotheses per object"]]) {
    const block = document.createElement("section");
    block.className = "chart-block";
    const heading = document.createElement("h2");
    heading.className = "chart-title";
    heading.textContent = title;
    const host = document.createElement("div");
    host.className = "chart-host";
    host.dataset.chart = id;
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    legend.dataset.legend = id;
    block.append(heading, host, legend);
    panel.append(block);
  }
  el.detailsViz.append(panel);
  updateInferenceDetails(details);
}

function updateInferenceDetails(details) {
  drawLineChart(el.detailsViz.querySelector('[data-chart="evidence"]'), details.steps || [], details.evidence || {}, details.burst_steps || []);
  drawLineChart(el.detailsViz.querySelector('[data-chart="hypotheses"]'), details.steps || [], details.hypotheses || {}, details.burst_steps || []);
  renderChartLegend(el.detailsViz.querySelector('[data-legend="evidence"]'), details.evidence || {});
  renderChartLegend(el.detailsViz.querySelector('[data-legend="hypotheses"]'), details.hypotheses || {});
}

function mountChannelDetails(details) {
  updateChannelDetails(details);
}

function updateChannelDetails(details) {
  const signature = (details.items || []).map((item) => item.channel).join("|");
  if (el.detailsViz.dataset.channels !== signature) {
    el.detailsViz.replaceChildren();
    el.detailsViz.dataset.channels = signature;
    for (const item of details.items || []) {
      const card = document.createElement("article");
      card.className = "panel channel-card";
      card.dataset.channel = item.channel;
      const head = document.createElement("div");
      head.className = "channel-card-head";
      const title = document.createElement("strong");
      title.textContent = item.channel;
      const feature = document.createElement("div");
      feature.className = "feature-host";
      feature.style.width = "76px";
      feature.style.height = "54px";
      feature.dataset.feature = "true";
      head.append(title, feature);
      const viz = document.createElement("div");
      viz.className = "viz-host mini-viz";
      viz.dataset.viz = "true";
      card.append(head, viz);
      el.detailsViz.append(card);
    }
  }
  for (const item of details.items || []) {
    const card = [...el.detailsViz.children].find((node) => node.dataset.channel === item.channel);
    if (!card) continue;
    renderFeature(card.querySelector('[data-feature="true"]'), item.feature);
    renderVisualization(card.querySelector('[data-viz="true"]'), item.visualization, `details:${item.channel}`);
  }
}

function mountDetailsPlaceholder(details) {
  el.detailsViz.replaceChildren();
  const panel = document.createElement("article");
  panel.className = "panel";
  panel.style.flex = "1";
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder";
  placeholder.textContent = details.message || "No details";
  panel.append(placeholder);
  el.detailsViz.append(panel);
}

function drawLineChart(host, steps, series, burstSteps) {
  if (!host) return;
  host.replaceChildren();
  const svg = svgNode("svg", { viewBox: "0 0 420 220", preserveAspectRatio: "none", role: "img", "aria-label": "History chart" });
  const margin = { left: 38, right: 12, top: 14, bottom: 28 };
  const width = 420 - margin.left - margin.right;
  const height = 220 - margin.top - margin.bottom;
  const finite = Object.values(series).flat().filter((value) => Number.isFinite(value));
  if (!steps.length || !finite.length) {
    const text = svgNode("text", { x:210, y:112, "text-anchor":"middle", class:"chart-label" });
    text.textContent = "No inference history yet";
    svg.append(text);
    host.append(svg);
    return;
  }
  const xMin = Math.min(...steps), xMax = Math.max(...steps);
  let yMin = Math.min(...finite), yMax = Math.max(...finite);
  if (yMin === yMax) { yMin -= .5; yMax += .5; }
  const yPad = (yMax-yMin)*.08;
  yMin -= yPad; yMax += yPad;
  const x = (value) => margin.left + ((value-xMin)/Math.max(1, xMax-xMin))*width;
  const y = (value) => margin.top + (1-(value-yMin)/(yMax-yMin))*height;

  for (let i=0; i<4; i+=1) {
    const yy = margin.top + i*height/3;
    svg.append(svgNode("line", { x1:margin.left, y1:yy, x2:margin.left+width, y2:yy, class:"chart-grid" }));
    const label = svgNode("text", { x:margin.left-6, y:yy+3, "text-anchor":"end", class:"chart-label" });
    label.textContent = formatNumber(yMax - i*(yMax-yMin)/3);
    svg.append(label);
  }
  for (const burst of burstSteps) {
    if (burst < xMin || burst > xMax) continue;
    svg.append(svgNode("line", { x1:x(burst), y1:margin.top, x2:x(burst), y2:margin.top+height, class:"chart-burst" }));
  }
  Object.entries(series).forEach(([name, values], index) => {
    let d = "";
    let inSegment = false;
    values.forEach((value, i) => {
      if (!Number.isFinite(value) || !Number.isFinite(steps[i])) { inSegment = false; return; }
      d += `${inSegment ? "L" : "M"}${x(steps[i]).toFixed(2)},${y(value).toFixed(2)}`;
      inSegment = true;
    });
    svg.append(svgNode("path", { d, fill:"none", stroke:seriesColor(index), "stroke-width":1.8, "vector-effect":"non-scaling-stroke" }));
  });
  svg.append(svgNode("line", { x1:margin.left, y1:margin.top+height, x2:margin.left+width, y2:margin.top+height, class:"chart-axis" }));
  const first = svgNode("text", { x:margin.left, y:210, class:"chart-label" }); first.textContent = xMin;
  const last = svgNode("text", { x:margin.left+width, y:210, "text-anchor":"end", class:"chart-label" }); last.textContent = xMax;
  svg.append(first, last);
  host.append(svg);
}

function renderChartLegend(host, series) {
  host.replaceChildren(...Object.keys(series).map((name, index) => {
    const item = document.createElement("span");
    const line = document.createElement("i");
    line.style.background = seriesColor(index);
    item.append(line, document.createTextNode(name));
    return item;
  }));
}

function renderExtras(extras) {
  /*
   * Record the current position of every existing card.
   *
   * This is the "First" part of FLIP.
   */
  const before = new Map();

  for (
    const card
    of [...el.extras.children]
  ) {
    if (!card.dataset.extra) {
      continue;
    }

    before.set(
      card.dataset.extra,
      card.getBoundingClientRect(),
    );
  }

  const wantedIds = new Set(
    extras.map(
      (item) => item.id,
    ),
  );

  /*
   * Remove models that genuinely disappeared.
   *
   * In a merge, absorbed source cards disappear here
   * when the next normal Memory snapshot arrives.
   */
  for (
    const card
    of [...el.extras.children]
  ) {
    const id =
      card.dataset.extra;

    if (!wantedIds.has(id)) {
      const host =
        card.querySelector(
          '[data-extra-viz="true"]',
        );

      host?._rendererHandle
        ?.cleanup?.();

      card.remove();
    }
  }

  /*
   * Update existing cards and create new cards.
   */
  for (const item of extras) {
    let card =
      [...el.extras.children].find(
        (node) =>
          node.dataset.extra ===
          item.id,
      );

    if (!card) {
      card =
        document.createElement(
          "article",
        );

      card.className =
        "panel channel-card";

      card.dataset.extra =
        item.id;

      const head =
        document.createElement(
          "div",
        );

      head.className =
        "channel-card-head";

      const title =
        document.createElement(
          "strong",
        );

      head.append(title);

      const host =
        document.createElement(
          "div",
        );

      host.className =
        "viz-host mini-viz";

      host.dataset.extraViz =
        "true";

      card.append(
        head,
        host,
      );

      /*
       * Initially append new cards.
       *
       * The ordering pass below will place them
       * into their correct evidence-ranked slot.
       */
      el.extras.append(card);
    }

    const title =
      card.querySelector(
        "strong",
      );

    if (title) {
      title.textContent =
        item.title || item.id;
    }

    const host =
      card.querySelector(
        '[data-extra-viz="true"]',
      );

    renderVisualization(
      host,
      item,
      `extra:${item.id}`,
    );
  }

  /*
   * IMPORTANT:
   *
   * `extras` arrives from Python already ordered by
   * evidence, highest evidence first.
   *
   * Reorder the existing DOM nodes to match that order.
   *
   * append(existingNode) MOVES the node; it does not
   * recreate it, so canvases/renderers/camera state
   * remain intact.
   */
  const cardsById = new Map(
    [...el.extras.children].map(
      (card) => [
        card.dataset.extra,
        card,
      ],
    ),
  );

  for (const item of extras) {
    const card =
      cardsById.get(item.id);

    if (!card) continue;

    el.extras.append(card);
  }

  /*
   * FLIP animation.
   *
   * At this point both:
   *
   *   1. removed Memory models
   *   2. evidence-driven reordering
   *
   * have already changed the layout.
   *
   * Animate directly from each card's previous
   * location to its final location.
   */
  requestAnimationFrame(() => {
    for (
      const card
      of [...el.extras.children]
    ) {
      const first =
        before.get(
          card.dataset.extra,
        );

      /*
       * Newly-created cards did not have an old position.
       */
      if (!first) continue;

      const last =
        card.getBoundingClientRect();

      const dx =
        first.left - last.left;

      const dy =
        first.top - last.top;

      if (
        Math.abs(dx) < 0.5
        && Math.abs(dy) < 0.5
      ) {
        continue;
      }

      /*
       * Cancel an older layout animation if another
       * evidence update arrives before it finishes.
       */
      card._layoutAnimation?.cancel?.();

      card._layoutAnimation =
        card.animate(
          [
            {
              transform:
                `translate(${dx}px, ${dy}px)`,
            },
            {
              transform:
                "translate(0, 0)",
            },
          ],
          {
            duration: 220,
            easing:
              "cubic-bezier(0.22, 1, 0.36, 1)",
          },
        );

      card._layoutAnimation.addEventListener(
        "finish",
        () => {
          card._layoutAnimation = null;
        },
        {
          once: true,
        },
      );
    }
  });
}

function memoryExtraCard(graphId) {
  return [...el.extras.children].find(
    (node) => node.dataset.extra === `memory:${graphId}`,
  );
}

function memoryExtraViz(graphId) {
  return memoryExtraCard(graphId)?.querySelector(
    '[data-extra-viz="true"]',
  ) || null;
}


function memoryExtraCanvas(graphId) {
  return memoryExtraViz(graphId)?.querySelector(
    "canvas",
  ) || null;
}


function memoryExtraRecord(graphId) {
  return (state.frame?.extras || []).find(
    (item) => item.id === `memory:${graphId}`,
  ) || null;
}


function lerp(a, b, t) {
  return a + (b - a) * t;
}


function interpolateMergeFrame(
  fromFrame,
  toFrame,
  t,
  is3d,
) {
  const fromCenter = pad3(
    fromFrame?.center || [0, 0, 0],
  );

  const toCenter = pad3(
    toFrame?.center || [0, 0, 0],
  );

  const dimensions = is3d ? 3 : 2;

  return {
    center: Array.from(
      { length: dimensions },
      (_, index) =>
        lerp(
          fromCenter[index],
          toCenter[index],
          t,
        ),
    ),

    half: lerp(
      Number(fromFrame?.half) || 0.025,
      Number(toFrame?.half) || 0.025,
      t,
    ),
  };
}


function copyCamera(key) {
  const camera = cameraFor(key);

  return {
    yaw: camera.yaw,
    pitch: camera.pitch,
    zoom: camera.zoom,
  };
}


function interpolateCamera(from, to, t) {
  return {
    yaw: lerp(from.yaw, to.yaw, t),
    pitch: lerp(from.pitch, to.pitch, t),
    zoom: lerp(from.zoom, to.zoom, t),
  };
}




function cleanupMemoryMergeFlyers(
  restoreSources = false,
) {
  const merge =
    state.memoryMerge;

  if (!merge?.flyers) return;

  for (
    const flyer
    of merge.flyers.values()
  ) {
    flyer.canvas?.remove();

    if (
      restoreSources
      && flyer.sourceHost
    ) {
      flyer.sourceHost.style.visibility =
        "";
    }
  }

  merge.flyers.clear();
}


function beginMemoryMergeFlyers(message) {
  const merge = state.memoryMerge;

  if (!merge) return;

  /*
   * Measure the ACTUAL rendered target canvas,
   * not the surrounding mini-viz host.
   */
  const targetCanvas =
    memoryExtraCanvas(
      message.target_id,
    );

  if (!targetCanvas) return;

  const targetRect =
    targetCanvas.getBoundingClientRect();

  merge.flyers = new Map();

  Object.entries(
    message.source_points || {},
  ).forEach(
    ([sourceId, points], index) => {
      const sourceHost =
        memoryExtraViz(sourceId);

      const sourceCanvas =
        memoryExtraCanvas(sourceId);

      if (
        !sourceHost
        || !sourceCanvas
      ) {
        return;
      }

      /*
       * Measure the actual source canvas.
       */
      const sourceRect =
        sourceCanvas.getBoundingClientRect();

      if (
        sourceRect.width <= 0
        || sourceRect.height <= 0
        || targetRect.width <= 0
        || targetRect.height <= 0
      ) {
        return;
      }

      /*
       * Get the EXACT frame the source Memory thumbnail
       * was using before the merge.
       */
      const sourceRecord =
        memoryExtraRecord(sourceId);

      const sourceFrame =
        sourceRecord?.frame
        || inferFrame(points);

      /*
       * Preserve the camera the source thumbnail
       * was actually using.
       *
       * renderExtras() uses:
       *
       *   extra:${item.id}
       *
       * and item.id is memory:<graph id>,
       *
       * so this is the normal source camera key.
       */
      const sourceCamera =
        copyCamera(
          `extra:memory:${sourceId}`,
        );

      /*
       * This is the camera the destination merge
       * renderer will use.
       */
      const targetCamera =
        copyCamera(
          `extra:memory:${message.target_id}`,
        );

      const canvas =
        document.createElement(
          "canvas",
        );

      canvas.className =
        "memory-merge-model-flyer";

      /*
       * START EXACTLY ON THE SOURCE CANVAS.
       *
       * No target sizing yet.
       */
      canvas.style.left =
        `${sourceRect.left}px`;

      canvas.style.top =
        `${sourceRect.top}px`;

      canvas.style.width =
        `${sourceRect.width}px`;

      canvas.style.height =
        `${sourceRect.height}px`;

      document.body.append(canvas);

      /*
       * At frame zero:
       *
       *   position = source position
       *   size     = source canvas size
       *   frame    = source model frame
       *   camera   = source camera
       *
       * Therefore it should visually match the source
       * thumbnail before that thumbnail disappears.
       */
      drawMemoryMergeFlyer(
        canvas,
        points,
        merge.source_colors?.[sourceId],
        merge,
        index + 1,
        sourceFrame,
        sourceCamera,
      );

      /*
       * Hide only the old model drawing.
       * The card remains completely stationary.
       */
      sourceHost.style.visibility =
        "hidden";

      merge.flyers.set(
        sourceId,
        {
          id: sourceId,

          canvas,
          sourceHost,
          colors:
            merge.source_colors?.[sourceId],

          /*
           * Screen-space endpoints.
           */
          startLeft:
            sourceRect.left,

          startTop:
            sourceRect.top,

          startWidth:
            sourceRect.width,

          startHeight:
            sourceRect.height,

          targetLeft:
            targetRect.left,

          targetTop:
            targetRect.top,

          targetWidth:
            targetRect.width,

          targetHeight:
            targetRect.height,

          /*
           * World-space endpoints.
           */
          sourceFrame,

          targetFrame:
            merge.frame,

          /*
           * Camera endpoints.
           */
          sourceCamera,
          targetCamera,

          seriesIndex:
            index + 1,
        },
      );
    },
  );
}


function updateMemoryMergeFlyers(
  sourcePoints,
  transformProgress,
  docked,
) {
  const merge =
    state.memoryMerge;

  if (!merge?.flyers) return;

  const travel = clamp(
    Number(
      transformProgress,
    ) || 0,
    0,
    1,
  );

  for (
    const [
      sourceId,
      flyer,
    ]
    of merge.flyers
  ) {
    /*
     * --------------------------------------------------
     * 1. SCREEN-SPACE INTERPOLATION
     * --------------------------------------------------
     *
     * Start as the exact source canvas.
     * Finish as the exact destination canvas.
     */
    const left = lerp(
      flyer.startLeft,
      flyer.targetLeft,
      travel,
    );

    const top = lerp(
      flyer.startTop,
      flyer.targetTop,
      travel,
    );

    const width = lerp(
      flyer.startWidth,
      flyer.targetWidth,
      travel,
    );

    const height = lerp(
      flyer.startHeight,
      flyer.targetHeight,
      travel,
    );

    flyer.canvas.style.left =
      `${left}px`;

    flyer.canvas.style.top =
      `${top}px`;

    flyer.canvas.style.width =
      `${width}px`;

    flyer.canvas.style.height =
      `${height}px`;

    /*
     * No CSS scale().
     *
     * We resize the actual canvas and redraw it instead.
     * This avoids scaling already-rasterized dots.
     */
    flyer.canvas.style.transform = "";

    /*
     * --------------------------------------------------
     * 2. WORLD-FRAME INTERPOLATION
     * --------------------------------------------------
     *
     * At t=0:
     *     source's normal fitted frame
     *
     * At t=1:
     *     exact merge.frame
     */
    const frame =
      interpolateMergeFrame(
        flyer.sourceFrame,
        flyer.targetFrame,
        travel,
        merge.is_3d,
      );

    /*
     * --------------------------------------------------
     * 3. CAMERA INTERPOLATION
     * --------------------------------------------------
     *
     * Mostly important for 3D.
     *
     * This also means a zoomed source object won't suddenly
     * inherit a different destination zoom on frame one.
     */
    const camera =
      interpolateCamera(
        flyer.sourceCamera,
        flyer.targetCamera,
        travel,
      );

    /*
     * --------------------------------------------------
     * 4. REDRAW THE ACTUAL RF-TRANSFORMED POINTS
     * --------------------------------------------------
     */
    drawMemoryMergeFlyer(
      flyer.canvas,

      sourcePoints?.[
        sourceId
      ] || [],

      flyer.colors,

      merge,

      flyer.seriesIndex,

      frame,

      camera,
    );

    /*
     * At the exact docking frame, drawMemoryMerge()
     * already places these transformed source points
     * into the real target canvas.
     *
     * Remove the temporary flying copy.
     */
    flyer.canvas.style.opacity =
      docked ? "0" : "1";
  }
}


function drawMemoryMerge(
  sourcePoints,
  transformProgress,
  mergedAlpha,
) {
  const merge = state.memoryMerge;

  if (!merge) return;

  const target = memoryExtraCard(
    merge.target_id,
  );

  const host = target?.querySelector(
    '[data-extra-viz="true"]',
  );

  if (!host) return;

  /*
   * Gradually tighten the animation frame to the
   * final merged model during the final crossfade.
   */
  const finalTransition =
    clamp(
      Number(mergedAlpha) || 0,
      0,
      1,
    );

  const displayFrame =
    interpolateMergeFrame(
      merge.frame,
      merge.finalFrame,
      finalTransition,
      merge.is_3d,
    );

  /*
   * Has the moving source reached the target?
   */
  const docked =
    Number(transformProgress || 0) >= 0.999;

  const oldModelAlpha =
    1 - mergedAlpha;

  /*
   * Existing target model.
   */
  const groups = [
    {
      label: merge.target_id,
      points: merge.target_points,
      colors: merge.target_colors,
      series_index: 0,
      alpha: oldModelAlpha,
    },
  ];

  /*
   * Once the moving source reaches the target,
   * draw its transformed points directly in the
   * target canvas.
   */
  if (docked) {
    Object.entries(
      sourcePoints || {},
    ).forEach(
      ([id, points], index) => {
        groups.push({
          label: id,
          points,
          colors:
            merge.source_colors?.[id],
          series_index:
            index + 1,
          alpha:
            oldModelAlpha,
        });
      },
    );
  }

  /*
   * Final merged model.
   *
   * It fades in during the final portion of
   * the animation.
   */
  groups.push({
    label:
      merge.new_graph_id,

    points:
      merge.merged_points,

    colors:
      merge.merged_colors,

    series_index:
      Object.keys(
        sourcePoints || {},
      ).length + 1,

    alpha:
      mergedAlpha,
  });

  /*
   * Use displayFrame rather than merge.frame.
   *
   * displayFrame gradually changes from the broad
   * animation bounds to the tight final model bounds.
   */
  const visualization =
    merge.is_3d
      ? {
          kind:
            "point_cloud",

          groups,

          projections:
            false,

          frame:
            displayFrame,
        }
      : {
          kind:
            "planar_cloud",

          groups,

          frame:
            displayFrame,
        };

  /*
   * Draw the destination first.
   */
  renderVisualization(
    host,
    visualization,
    `extra:memory:${merge.target_id}`,
  );

  /*
   * Then update the travelling source model.
   */
  updateMemoryMergeFlyers(
    sourcePoints,
    transformProgress,
    docked,
  );
}


function handleMemoryMerge(message) {
  if (message.type === "memory_merge_begin") {
    // If an old animation somehow got interrupted, clean it up first.
    if (state.memoryMerge) {
      cleanupMemoryMergeFlyers(true);
    }

    const target = memoryExtraCard(message.target_id);

    // Same behavior as before: only animate if this Memory graph
    // is actually visible in the current UI.
    if (!target) return;

    state.memoryMerge = {
      ...message,

      // Fixed target frame prevents zoom jitter while the final model
      // crossfades into place.
      frame: inferFrame(
        message.bounds_points || [],
      ),
      // NEW:
      finalFrame: inferFrame(
        message.merged_points || [],
      ),
      flyers: new Map(),
    };

    const title = target.querySelector("strong");

    if (title) {
      title.textContent =
        `${message.target_id} + sources → ${message.new_graph_id}`;
    }

    // Create the source model(s) at their real current positions
    // in the Memory grid.
    beginMemoryMergeFlyers(message);

    // Draw frame zero.
    drawMemoryMerge(
      message.source_points,
      0,
      0,
    );

    return;
  }

  if (!state.memoryMerge) return;

  if (message.type === "memory_merge_frame") {
    drawMemoryMerge(
      message.source_points,

      // Physical travel and RF transform are now synchronized.
      Number(
        message.transform_progress || 0,
      ),

      Number(
        message.merged_alpha || 0,
      ),
    );

    return;
  }

  if (message.type === "memory_merge_end") {
    const merge =
      state.memoryMerge;

    if (!merge) return;

    /*
    * The target card already contains the final merged
    * visualization at this point.
    *
    * Rather than throwing that card away and creating a new
    * one on the next normal frame, simply rename it to the
    * final graph ID.
    */
    const targetCard =
      memoryExtraCard(
        merge.target_id,
      );

    if (targetCard) {
      const oldCameraKey =
        `extra:memory:${merge.target_id}`;

      const newCameraKey =
        `extra:memory:${merge.new_graph_id}`;

      state.camera.set(
        newCameraKey,
        cameraFor(oldCameraKey),
      );
      targetCard.dataset.extra =
        `memory:${merge.new_graph_id}`;

      const title =
        targetCard.querySelector(
          "strong",
        );

      if (title) {
        title.textContent =
          merge.new_graph_id;
      }
    }

    
    
    
    /*
    * IMPORTANT:
    *
    * Do NOT remove the source cards here.
    *
    * They're already visually hidden by the merge animation.
    * Leave their DOM boxes in place until the next normal
    * snapshot reconciles Memory.
    *
    * This prevents an immediate CSS-grid reflow at merge_end.
    */

    // /*
    // * Remove the source Memory cards that were absorbed.
    // */
    // for (
    //   const sourceId
    //   of Object.keys(
    //     merge.source_points || {},
    //   )
    // ) {
    //   const sourceCard =
    //     memoryExtraCard(sourceId);

    //   if (!sourceCard) continue;

    //   const host =
    //     sourceCard.querySelector(
    //       '[data-extra-viz="true"]',
    //     );

    //   host?._rendererHandle
    //     ?.cleanup?.();

    //   sourceCard.remove();
    // }

    /*
    * Remove only the temporary travelling canvases.
    */
    cleanupMemoryMergeFlyers(
      false,
    );

    state.memoryMerge = null;
    return;
  }
}

function renderControls(controls) {
  const interactive = controls.mode === "interactive";
  el.monitorControls.classList.toggle("hidden", interactive);
  el.interactiveControls.classList.toggle("hidden", !interactive);
  el.controlMode.textContent = interactive ? "Teleoperate" : "Monitor";
  el.controlDescription.textContent = state.role === "observer"
    ? "Read-only observer · the first connected browser owns controls."
    : (interactive
      ? "Automatic corrections pass through; choice points wait for you."
      : "Adjust pacing without changing Monty actions.");
  const canControl = state.role === "controller";
  el.speedSlider.disabled = !canControl;
  el.pauseToggle.disabled = !canControl;
  el.stepSlider.disabled = !canControl;
  if (!interactive) {
    const speed = Number(controls.speed ?? 1);
    el.speedSlider.value = speed;
    el.speedOutput.value = `${Math.round(speed*100)}%`;
    el.pauseToggle.textContent = speed <= 0 ? "▶" : "Ⅱ";
    if (speed > 0) state.lastNonzeroSpeed = speed;
  } else {
    const scale = Number(controls.step_scale ?? 1);
    el.stepSlider.min = controls.step_scale_min ?? .1;
    el.stepSlider.max = controls.step_scale_max ?? 3;
    el.stepSlider.value = scale;
    el.stepOutput.value = `${scale.toFixed(2)}×`;
  }
}

function renderActionState() {
  const request = state.actionRequest;
  const interactive = state.frame?.controls?.mode === "interactive";
  const active = Boolean(request && interactive && state.role === "controller");
  const headings = new Set(request?.headings || []);
  for (const button of $$(".direction")) {
    const enabled = active && headings.has(button.dataset.action);
    button.disabled = !enabled;
    button.classList.toggle("enabled", enabled);
  }
  const specials = new Set(request?.specials || []);
  const hasJump = active && specials.has("jump");
  el.jumpButton.classList.toggle("hidden", !hasJump);
  el.jumpButton.disabled = !hasJump;
  el.endButton.disabled = !(active && specials.has("End episode"));
  el.choiceNote.textContent = active
    ? "Waiting for your action"
    : (request && state.role === "observer" ? "Observer · another tab is controlling" : "Automatic policy step");
  el.choiceNote.classList.toggle("active", active);
}

function chooseAction(name) {
  const request = state.actionRequest;
  if (!request) return;
  const valid = new Set([...(request.headings || []), ...(request.specials || [])]);
  if (!valid.has(name)) return;
  send({ type: "action", request_id: request.request_id, name });
}

el.lmSelect.addEventListener("change", () => send({ type: "select_lm", id: el.lmSelect.value }));
el.channelSelect.addEventListener("change", () => send({ type: "select_channel", id: el.channelSelect.value }));
$$('.direction').forEach((button) => button.addEventListener("click", () => chooseAction(button.dataset.action)));
el.jumpButton.addEventListener("click", () => chooseAction("jump"));
el.endButton.addEventListener("click", () => chooseAction("End episode"));
el.resetCamera.addEventListener("click", () => el.montyViz._rendererHandle?.reset?.());

el.speedSlider.addEventListener("input", () => {
  const value = Number(el.speedSlider.value);
  el.speedOutput.value = `${Math.round(value*100)}%`;
  if (value > 0) state.lastNonzeroSpeed = value;
  el.pauseToggle.textContent = value <= 0 ? "▶" : "Ⅱ";
  send({ type: "set_speed", value });
});
el.pauseToggle.addEventListener("click", () => {
  const current = Number(el.speedSlider.value);
  const value = current > 0 ? 0 : Math.max(state.lastNonzeroSpeed, .25);
  el.speedSlider.value = value;
  el.speedSlider.dispatchEvent(new Event("input"));
});
el.stepSlider.addEventListener("input", () => {
  const value = Number(el.stepSlider.value);
  el.stepOutput.value = `${value.toFixed(2)}×`;
  send({ type: "set_step_scale", value });
});

document.addEventListener("keydown", (event) => {
  if (isEditing(event.target)) return;

  // Prevent browser scrolling while keyboard-controlling Monty,
  // including the brief gap between action requests.
  if (
    state.frame?.controls?.mode === "interactive"
    && state.role === "controller"
    && ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", " "].includes(event.key)
  ) {
    event.preventDefault();
  }
  
  if (!state.actionRequest || isEditing(event.target)) return;
  const map = {
    w: "up", W: "up", ArrowUp: "up",
    s: "down", S: "down", ArrowDown: "down",
    a: "left", A: "left", ArrowLeft: "left",
    d: "right", D: "right", ArrowRight: "right",
    " ": "jump", Delete: "End episode",
  };
  const action = map[event.key];
  if (!action) return;
  const valid = new Set([...(state.actionRequest.headings || []), ...(state.actionRequest.specials || [])]);
  if (!valid.has(action)) return;
  chooseAction(action);
});

function isEditing(target) {
  return target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement || target?.isContentEditable;
}

function setConnection(online, label) {
  el.connection.classList.toggle("online", online);
  el.connection.querySelector("span").textContent = label;
}

function canvasContext(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width*dpr));
  const height = Math.max(1, Math.floor(rect.height*dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
  }
  return { ctx: canvas.getContext("2d"), width, height, dpr };
}

function inferFrame(points) {
  if (!points.length) return { center:[0,0,0], half:.025 };
  const dims = Math.min(points[0]?.length || 2, 3);
  const low = Array(dims).fill(Infinity), high = Array(dims).fill(-Infinity);
  for (const point of points) for (let i=0; i<dims; i+=1) {
    low[i] = Math.min(low[i], point[i]); high[i] = Math.max(high[i], point[i]);
  }
  const center = low.map((value, i) => (value+high[i])/2);
  const span = Math.max(...high.map((value, i) => value-low[i]));
  let size = .05;
  if (span > size) size += .05*Math.ceil((span-.05)/.05);
  return { center, half:size/2 };
}

function flatPoints(groups) { return groups.flatMap((group) => group.points || []); }
function pad3(point) { return [Number(point?.[0] || 0), Number(point?.[1] || 0), Number(point?.[2] || 0)]; }
function clamp(value, low, high) { return Math.max(low, Math.min(high, value)); }
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function seriesColor(index) { return css(`--series-${(Number(index)%6)+1}`) || css("--accent"); }
function drawCross(ctx, x, y, size, color, width) { ctx.save(); ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath(); ctx.moveTo(x-size,y-size); ctx.lineTo(x+size,y+size); ctx.moveTo(x+size,y-size); ctx.lineTo(x-size,y+size); ctx.stroke(); ctx.restore(); }
function formatNumber(value) { return Math.abs(value) >= 1000 ? value.toExponential(1) : Number(value.toFixed(2)).toString(); }
function titleForKind(kind) { return ({ point_cloud:"Spatial representation", planar_cloud:"Planar representation", placeholder:"Representation" })[kind] || "Representation"; }
function escapeHtml(value) { const node=document.createElement("span"); node.textContent=String(value); return node.innerHTML; }
function svgNode(tag, attrs={}) { const node=document.createElementNS(SVG_NS, tag); for (const [key,value] of Object.entries(attrs)) node.setAttribute(key, value); return node; }
function toast(message) { el.toast.textContent=message; el.toast.classList.add("show"); clearTimeout(toast.timer); toast.timer=setTimeout(()=>el.toast.classList.remove("show"),1800); }

connect();
