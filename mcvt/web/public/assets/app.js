/* MCVT Vision dashboard: camera grid / focus stream, cross-camera follow, vehicle gallery, journey requests. */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const S = {
    csrf: null, ws: null,
    cams: new Map(),          // id -> {info, stats, bitmap, header, cell, canvas, rect, busy, pending}
    vehicles: new Map(),      // live vehicles by gid
    follow: null, focus: null, target: null,
    mode: "grid", mainCam: null, manualCam: null, autoSince: 0,
    autoSwitch: true,
    fade: null, mainRect: null, dirty: true,
    tab: "live", history: [], thumbV: "",
  };
  const COLORS = { black: "#111", white: "#f4f4f4", silver: "#c0c4c8", grey: "#7c8288", red: "#d83b3b", orange: "#ef8a2f",
    yellow: "#e8cf3a", green: "#3fa865", cyan: "#3fc4c9", blue: "#3b6fd8", purple: "#8e5ad8" };

  // ----------------------------------------------------------------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString([], { hour12: false }) : "–";
  /** shared camera clock (seconds of the day, cameras.clock_start) -> "08:01:23" */
  const fmtClock = (s) => {
    if (s === null || s === undefined) return "–";
    s = Math.floor(s) % 86400;
    const p = (n) => String(n).padStart(2, "0");
    return `${p(Math.floor(s / 3600))}:${p(Math.floor(s % 3600 / 60))}:${p(s % 60)}`;
  };
  const fmtDate = (ts) => ts ? new Date(ts * 1000).toLocaleString([], { hour12: false }) : "–";
  const ago = (ts) => {
    if (!ts) return "";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return `${Math.round(s)}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    return fmtDate(ts);
  };
  const hue = (gid) => { let h = 0; for (const c of String(gid)) h = (h * 31 + c.charCodeAt(0)) % 360; return h; };
  const gidColor = (gid, a = 1) => gid ? `hsla(${hue(gid)}, 85%, 60%, ${a})` : `rgba(148,163,184,${a})`;
  const TARGET = "#10b981";
  const swatch = (c) => c ? `<span class="swatch" style="background:${COLORS[c] || c}"></span>${esc(c)}` : "–";
  const camLabel = (c) => (c.info.name && c.info.name !== c.info.id ? `${c.info.id} · ${c.info.name}` : c.info.id);

  async function api(path, opts = {}) {
    const res = await fetch(path, { credentials: "same-origin", ...opts,
      headers: { "Content-Type": "application/json", "X-CSRF-Token": S.csrf || "", ...(opts.headers || {}) } });
    if (res.status === 401) { location.href = "login.php"; throw new Error("signed out"); }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  function toast(msg, kind = "") {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = msg;
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 6000);
  }

  function setStatus(id, text, kind) {
    const el = $(id);
    el.querySelector(".txt").textContent = text;
    el.querySelector(".status-dot").className = `status-dot ${kind}`;
  }

  // --------------------------------------------------------------- websocket
  async function connect() {
    setStatus("hubStatus", "Connecting…", "tracking");
    let session;
    try { session = await api("api/session.php"); } catch (e) { setTimeout(connect, 3000); return; }
    S.csrf = session.csrf;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const base = session.ws.url || `${proto}://${location.hostname}:${session.ws.port}/`;
    const ws = new WebSocket(`${base}${base.includes("?") ? "&" : "?"}token=${encodeURIComponent(session.ws.token)}`);
    ws.binaryType = "arraybuffer";
    S.ws = ws;
    ws.onopen = () => {
      setStatus("hubStatus", "Live link", "online");
      if (S.follow) send({ type: "focus", gid: S.follow });
      sendView();
    };
    ws.onclose = () => {
      setStatus("hubStatus", "Link down – retrying", "offline");
      setStatus("engineStatus", "AI Engine Unknown", "tracking");
      setTimeout(connect, 2000);
    };
    ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onFrame(ev.data));
  }
  const send = (obj) => { if (S.ws && S.ws.readyState === 1) S.ws.send(JSON.stringify(obj)); };

  /** Tell the engine what we watch: the focus camera at full rate, or every grid camera. */
  function sendView() {
    send({ type: "view", cam: S.mode === "focus" ? S.mainCam : null, grid: S.mode === "grid" ? [...S.cams.keys()] : [] });
  }

  function onEvent(m) {
    switch (m.type) {
      case "engine": setStatus("engineStatus", m.status === "online" ? "AI Engine Online" : "AI Engine Offline", m.status === "online" ? "online" : "offline"); break;
      case "cameras": m.items.forEach(upsertCamera); sendView(); break;
      case "stats": onStats(m); break;
      case "vehicles": onVehicles(m.items); break;
      case "focus":
        if (m.gid === S.follow) { S.focus = m; $("followReason").textContent = m.reason || ""; updateCells(); chooseMain(); }
        break;
      case "switch": if (m.gid === S.follow) onSwitch(m); break;
      case "merge": onMerge(m); break;
      case "handoff": if (m.gid === S.follow) onHandoff(m); break;
      case "job": onJob(m); break;
      case "log": addLog(m); break;
    }
  }

  // ------------------------------------------------------------------ frames
  function onFrame(buf) {
    const dv = new DataView(buf);
    const hlen = dv.getUint32(0);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
    const cam = S.cams.get(header.cam) || upsertCamera({ id: header.cam, name: header.cam, status: "online" });
    const jpeg = new Blob([new Uint8Array(buf, 4 + hlen)], { type: "image/jpeg" });
    if (header.delay != null) S.delay = header.delay;
    if (cam.busy) { cam.pending = { header, jpeg }; return; }
    decode(cam, header, jpeg);
  }

  async function decode(cam, header, jpeg) {
    cam.busy = true;
    try {
      const bmp = await createImageBitmap(jpeg);
      const old = cam.bitmap;
      cam.bitmap = bmp; cam.header = header;
      if (old && (!S.fade || S.fade.bitmap !== old)) old.close();
      drawCell(cam);
      $("gridEmpty").classList.add("hidden");
      if (!S.mainCam) chooseMain();
      if (cam.info.id === S.mainCam) S.dirty = true;
    } catch (e) { /* skip corrupt frame */ }
    cam.busy = false;
    if (cam.pending) { const p = cam.pending; cam.pending = null; decode(cam, p.header, p.jpeg); }
  }

  function upsertCamera(info) {
    let cam = S.cams.get(info.id);
    if (!cam) {
      const cell = document.createElement("button");
      cell.className = "camera-cell";
      cell.innerHTML = `<canvas></canvas><div class="cell-wait hidden"></div><span class="cell-live">LIVE</span><span class="cell-target hidden"></span>
        <div class="cell-overlay"><div><div class="cell-name"></div><div class="cell-sub"></div></div><span class="cell-badge">0 vehicles</span></div>
        <span class="cell-score" style="width:0"></span>`;
      cam = { info, stats: {}, cell, canvas: cell.querySelector("canvas") };
      cell.addEventListener("click", (ev) => onCellClick(cam, ev));
      S.cams.set(info.id, cam);
      const ids = [...S.cams.keys()].sort();
      ids.forEach((id) => $("camGrid").appendChild(S.cams.get(id).cell));
      const n = ids.length;
      $("camGrid").style.setProperty("--cols", n <= 1 ? 1 : n <= 4 ? 2 : n <= 9 ? 3 : 4);
      $("camCount").textContent = n;
    }
    cam.info = { ...cam.info, ...info };
    cam.cell.querySelector(".cell-name").textContent = cam.info.id;
    cam.cell.title = `Open ${camLabel(cam)} in the focus stream`;
    cam.cell.classList.toggle("offline", cam.info.status === "offline");
    updateCellText(cam);
    return cam;
  }

  function updateCellText(cam) {
    const i = cam.info, st = cam.stats || {};
    const parts = [];
    if (i.name && i.name !== i.id) parts.push(i.name);
    if (i.width) parts.push(`${i.width}×${i.height}`);
    if (st.decode_fps != null) parts.push(`${st.decode_fps} fps · AI ${st.process_fps}/s`);
    cam.cell.querySelector(".cell-sub").textContent = parts.join(" · ");
    const n = cam.header ? cam.header.tracks.filter((t) => t.d).length : (st.tracks ?? 0);
    cam.cell.querySelector(".cell-badge").textContent = `${n} vehicle${n === 1 ? "" : "s"}`;
  }

  function fitRect(cw, ch, iw, ih) {
    const s = Math.min(cw / iw, ch / ih);
    return { x: (cw - iw * s) / 2, y: (ch - ih * s) / 2, w: iw * s, h: ih * s };
  }

  function drawTracks(ctx, header, rect, small) {
    const scale = small ? 0.8 : 1;
    for (const t of header.tracks || []) {
      const [x1, y1, x2, y2] = t.b;
      const x = rect.x + x1 * rect.w, y = rect.y + y1 * rect.h, w = (x2 - x1) * rect.w, h = (y2 - y1) * rect.h;
      const target = S.follow && t.g === S.follow;
      const color = target ? TARGET : gidColor(t.g, t.g ? 0.95 : 0.55);
      ctx.save();
      ctx.lineWidth = (target ? 3.5 : 2) * scale;
      ctx.strokeStyle = color;
      if (!t.d) ctx.setLineDash([6, 4]);
      ctx.strokeRect(x, y, w, h);
      if (target) { ctx.shadowColor = TARGET; ctx.shadowBlur = 12; ctx.strokeRect(x, y, w, h); ctx.shadowBlur = 0; }
      // global id + the features stored with it (colour, type); no label until the id is known
      const label = !t.g ? "" : small ? t.g : `${t.g} · ${[t.o, t.c].filter(Boolean).join(" ")}`;
      if (label) {
        ctx.font = `600 ${small ? 11 : 13}px Inter, Segoe UI, sans-serif`;
        const tw = ctx.measureText(label).width + 10, th = small ? 16 : 20;
        ctx.fillStyle = color;
        ctx.fillRect(x, Math.max(0, y - th), tw, th);
        ctx.fillStyle = "#0a0e17";
        ctx.fillText(label, x + 5, Math.max(th - 5, y - 5));
      }
      ctx.restore();
    }
  }

  function sizeCanvas(c) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(c.clientWidth * dpr)), h = Math.max(1, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return { w, h, dpr };
  }

  function drawCell(cam) {
    if (!cam.bitmap) return;
    const c = cam.canvas, { w, h, dpr } = sizeCanvas(c), ctx = c.getContext("2d");
    ctx.fillStyle = "#05070c"; ctx.fillRect(0, 0, w, h);
    const rect = fitRect(w, h, cam.bitmap.width, cam.bitmap.height);
    ctx.drawImage(cam.bitmap, rect.x, rect.y, rect.w, rect.h);
    drawTracks(ctx, cam.header, rect, S.mode === "focus" || w < 500 * dpr);
    cam.rect = { ...rect, dpr };
    updateCellText(cam);
  }

  function render(now) {
    const canvas = $("mainCanvas");
    if (S.mode === "focus") {
      const { w: cw, h: ch, dpr } = sizeCanvas(canvas);
      const cam = S.cams.get(S.mainCam);
      if (cam && cam.bitmap && (S.dirty || S.fade || canvas.dataset.size !== `${cw}x${ch}`)) {
        canvas.dataset.size = `${cw}x${ch}`;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cw, ch);
        const rect = fitRect(cw, ch, cam.bitmap.width, cam.bitmap.height);
        ctx.drawImage(cam.bitmap, rect.x, rect.y, rect.w, rect.h);
        drawTracks(ctx, cam.header, rect, false);
        if (S.fade) { // smooth cross-fade on camera switches (NFR-U-02)
          const a = 1 - (now - S.fade.t0) / 450;
          if (a <= 0 || !S.fade.bitmap) { if (S.fade.bitmap && S.fade.bitmap !== cam.bitmap) try { S.fade.bitmap.close(); } catch (e) {} S.fade = null; }
          else { const r2 = fitRect(cw, ch, S.fade.bitmap.width, S.fade.bitmap.height); ctx.globalAlpha = a; ctx.drawImage(S.fade.bitmap, r2.x, r2.y, r2.w, r2.h); ctx.globalAlpha = 1; }
        }
        S.mainRect = { ...rect, dpr };
        S.dirty = false;
      }
    }
    requestAnimationFrame(render);
  }

  // hit-test a click against the boxes drawn on a canvas
  function hitVehicle(header, rect, canvas, ev) {
    if (!header || !rect) return null;
    const r = canvas.getBoundingClientRect();
    const px = ((ev.clientX - r.left) * rect.dpr - rect.x) / rect.w;
    const py = ((ev.clientY - r.top) * rect.dpr - rect.y) / rect.h;
    const hits = (header.tracks || []).filter((t) => t.g && px >= t.b[0] && px <= t.b[2] && py >= t.b[1] && py <= t.b[3]);
    hits.sort((a, b) => (a.b[2] - a.b[0]) * (a.b[3] - a.b[1]) - (b.b[2] - b.b[0]) * (b.b[3] - b.b[1]));
    return hits[0] ? hits[0].g : null;
  }

  function onCellClick(cam, ev) {
    const gid = S.mode === "grid" ? hitVehicle(cam.header, cam.rect, cam.canvas, ev) : null;
    if (gid) { track(gid); return; }
    S.manualCam = cam.info.id;
    setMode("focus");
    chooseMain(true);
  }

  $("mainCanvas").addEventListener("click", (ev) => {
    const cam = S.cams.get(S.mainCam);
    const gid = cam && hitVehicle(cam.header, S.mainRect, $("mainCanvas"), ev);
    if (gid) track(gid);
  });

  // ------------------------------------------------------- view mode + main
  function setMode(mode) {
    S.mode = mode;
    $("cameraPanel").className = `camera-panel ${mode === "grid" ? "grid-view" : "single-view"}`;
    $("btnGrid").classList.toggle("btn-primary", mode === "grid");
    $("btnFocus").classList.toggle("btn-primary", mode === "focus");
    S.dirty = true;
    requestAnimationFrame(() => S.cams.forEach(drawCell));
    sendView();
  }
  $("btnGrid").onclick = () => { S.manualCam = null; setMode("grid"); };
  $("btnFocus").onclick = () => { setMode("focus"); chooseMain(true); };
  $("btnBackGrid").onclick = () => { S.manualCam = null; setMode("grid"); };
  $("autoSwitch").onchange = (e) => {
    S.autoSwitch = e.target.checked;
    $("autoState").textContent = S.autoSwitch ? "ENGAGED" : "OFF";
    $("autoState").style.color = S.autoSwitch ? "" : "var(--text-muted)";
    if (S.autoSwitch && S.follow) { S.manualCam = null; setMode("focus"); chooseMain(true); }
  };

  function chooseMain(userAction = false) {
    if (S.replay) return; // the journey replay owns the main view until it is closed
    let target = S.mainCam, mode = "[Auto]";
    const followActive = S.follow && S.autoSwitch && !S.manualCam;
    if (followActive) {
      const v = S.vehicles.get(S.follow);
      target = (S.focus && S.focus.gid === S.follow && S.focus.active) || (v && v.primary_camera) || target;
      mode = "[Cross-camera follow · best view chosen by the switching engine]";
    } else if (S.manualCam) {
      target = S.manualCam; mode = "[Manual]";
    } else {
      let best = target, bestN = -1;
      for (const [id, c] of S.cams) {
        const n = c.header ? c.header.tracks.length : 0;
        if (n > bestN) { best = id; bestN = n; }
      }
      const cur = S.cams.get(target);
      const curN = cur && cur.header ? cur.header.tracks.length : -1;
      if (!target || userAction || (best !== target && bestN > curN && Date.now() - S.autoSince > 5000)) target = best;
      mode = "[Auto · busiest camera]";
    }
    $("modeChip").textContent = mode;
    if (target && target !== S.mainCam) {
      const old = S.cams.get(S.mainCam);
      if (old && old.bitmap && S.mode === "focus") S.fade = { bitmap: old.bitmap, t0: performance.now() };
      S.mainCam = target; S.autoSince = Date.now(); S.dirty = true;
      const c = S.cams.get(target);
      $("mainLabel").textContent = c ? `CAM ${camLabel(c)}` : target;
      sendView();
    }
    updateCells();
  }
  setInterval(() => { if (!S.follow && !S.manualCam) chooseMain(); }, 1000);

  function updateCells() {
    const scores = (S.focus && S.focus.gid === S.follow && S.focus.scores) || {};
    const max = Math.max(0.0001, ...Object.values(scores));
    const live = S.follow ? S.vehicles.get(S.follow) : null;
    const seenNow = new Set(live ? Object.keys(live.cameras_now || {}) : []);
    for (const [id, c] of S.cams) {
      c.cell.classList.toggle("active", id === S.mainCam && S.mode === "focus");
      const has = !!S.follow && (scores[id] || seenNow.has(id));
      c.cell.classList.toggle("has-target", has);
      const tag = c.cell.querySelector(".cell-target");
      tag.classList.toggle("hidden", !has);
      tag.textContent = S.follow || "";
      c.cell.querySelector(".cell-score").style.width = has && scores[id] ? `${Math.round(100 * scores[id] / max)}%` : "0";
    }
    $("mainCamera").classList.toggle("active-tracking", !!S.follow);
  }

  function onSwitch(m) {
    const b = $("switchBanner");
    b.textContent = m.from ? `Camera switch: ${m.from} → ${m.to}` : `Now showing ${m.to}`;
    b.classList.remove("hidden");
    clearTimeout(S.bannerTimer);
    S.bannerTimer = setTimeout(() => b.classList.add("hidden"), 3500);
    const f = $("switchFlash");
    f.classList.remove("active"); void f.offsetWidth; f.classList.add("active");
    S.focus = { ...(S.focus || {}), gid: m.gid, active: m.to };
    if (S.autoSwitch) { S.manualCam = null; if (S.mode !== "focus") setMode("focus"); }
    chooseMain();
    refreshTarget();
  }

  /** The followed vehicle left every camera: show where it is expected next; announce when found. */
  function onHandoff(m) {
    S.handoff = m;
    const exp = (m.expected || []).map((e) => e.camera);
    if (m.status === "searching") {
      const list = exp.length ? `${exp.join(", ")} (learned)${m.others && m.others.length ? ` · then ${m.others.join(", ")}` : ""}`
        : (m.others || []).join(", ");
      const txt = `Left ${m.from} ${Math.round(m.elapsed)} s ago · searching next camera: ${list}`;
      $("followReason").textContent = txt;
      $("tLabel").textContent = `◌ Searching next camera…`;
      $("tNow").textContent = `left ${m.from}`;
    } else if (m.status === "lost") {
      $("followReason").textContent = `Not seen since leaving ${m.from} (${Math.round(m.elapsed)} s) – still watching all cameras`;
      $("tLabel").textContent = "○ Lost – waiting to reappear";
    } else if (m.status === "found") {
      const b = $("switchBanner");
      b.textContent = `${m.gid} found on ${m.to} (left ${m.from} ${Math.round(m.after)} s earlier)`;
      b.classList.remove("hidden");
      clearTimeout(S.bannerTimer);
      S.bannerTimer = setTimeout(() => b.classList.add("hidden"), 4000);
      toast(`Hand-off: ${m.gid} ${m.from} → ${m.to}`, "ok");
      if (S.autoSwitch) { S.manualCam = null; S.focus = { ...(S.focus || {}), gid: m.gid, active: m.to }; setMode("focus"); chooseMain(); }
      S.handoff = null;
      refreshTarget(true);
    }
    const expected = new Set(m.status === "searching" ? [...exp, ...(m.others || [])] : []);
    for (const [id, c] of S.cams) {
      c.cell.classList.toggle("expected", expected.has(id));
      c.cell.classList.toggle("expected-top", exp[0] === id && m.status === "searching");
    }
  }

  // ---------------------------------------------------------------- tracking
  /** Clicking a vehicle locks onto it and follows it across all cameras. */
  function track(gid) {
    if (S.follow !== gid) {
      stopReplay();
      S.follow = gid; S.focus = null; S.manualCam = null; S.target = null; S.journey = null;
      $("tJourney").innerHTML = "";
      $("btnReplay").disabled = true;
      send({ type: "focus", gid });
      addLog({ ts: Date.now() / 1000, level: "info", source: "ui", message: `Cross-camera lock on ${gid}.` });
    }
    $("followGid").textContent = gid;
    $("followReason").textContent = "Looking for the best camera…";
    $("followChip").classList.remove("hidden");
    if (S.autoSwitch) setMode("focus");
    chooseMain(true);
    renderGallery();
    refreshTarget(true);
  }

  function untrack() {
    if (S.follow) addLog({ ts: Date.now() / 1000, level: "info", source: "ui", message: `Released lock on ${S.follow}.` });
    stopReplay();
    S.follow = null; S.focus = null; S.target = null; S.handoff = null; S.journey = null;
    S.cams.forEach((c) => c.cell.classList.remove("expected", "expected-top"));
    send({ type: "focus", gid: null });
    $("followChip").classList.add("hidden");
    $("target").classList.add("hidden");
    $("noTarget").classList.remove("hidden");
    chooseMain(true); renderGallery();
  }
  $("unfollow").onclick = untrack;
  $("btnStop").onclick = untrack;

  async function refreshTarget(full = false) {
    const gid = S.follow;
    if (!gid) return;
    $("noTarget").classList.add("hidden");
    $("target").classList.remove("hidden");
    $("tGid").textContent = gid;
    const live = S.vehicles.get(gid);
    if (live) fillLive(live);
    if (full || !S.journey || Date.now() - S.journey.at > 3000) loadJourney(gid);
    if (full || !S.features || S.features.gid !== gid || Date.now() - S.features.at > 30000) loadFeatures(gid);
    if (!full && S.target && S.target.vehicle.gid === gid && Date.now() - S.target.at < 5000) return;
    try {
      const d = await api(`api/vehicle.php?gid=${encodeURIComponent(gid)}`);
      if (S.follow !== gid) return;
      S.target = { ...d, at: Date.now() };
      const v = d.vehicle;
      if (!live) {
        $("tClass").textContent = v.vclass || "–";
        $("tColor").innerHTML = swatch(v.color);
        $("tMotion").textContent = v.motion || "–";
        $("tNow").textContent = "not in view";
        $("tLabel").textContent = "○ Locked · not in view";
        $("target").classList.add("idle");
        if (v.thumb) $("tThumb").src = `${v.thumb}&t=${Date.now()}`;
      }
      $("tFirst").textContent = fmtDate(v.first_seen);
      $("tLast").textContent = fmtDate(v.last_seen);
      renderPath(d.sightings, live);
      renderTimeline(d.sightings);
      $("tSwitches").innerHTML = d.switches.length
        ? d.switches.map((s) => `<li><span class="mono muted">${fmtTime(s.ts)}</span> ${esc(s.from_cam || "–")} → <strong>${esc(s.to_cam)}</strong></li>`).join("")
        : `<li class="muted">No camera switches yet.</li>`;
      renderJobs(d.jobs, $("tJobs"));
    } catch (e) { toast(e.message, "err"); }
  }

  function fillLive(v) {
    $("tClass").textContent = v.class || "–";
    $("tColor").innerHTML = swatch(v.color);
    $("tMotion").textContent = v.motion || "–";
    const now = Object.keys(v.cameras_now || {});
    $("tNow").innerHTML = now.length ? now.map((c) => c === v.primary_camera ? `<strong>${esc(c)}</strong>` : esc(c)).join(", ") : "–";
    $("tLast").textContent = fmtDate(v.last_seen);
    if (v.live || !S.handoff) $("tLabel").textContent = v.live ? "● Tracking live"
      : v.searching ? `◌ Left ${v.search_from || "view"} · searching ${(v.expected || []).join(", ") || "all cameras"} (${Math.floor(v.search_left / 60)}:${String(v.search_left % 60).padStart(2, "0")} left)`
      : "○ Locked · not in view";
    $("target").classList.toggle("idle", !v.live);
    if (v.thumb_version && S.thumbV !== `${v.gid}:${v.thumb_version}`) {
      S.thumbV = `${v.gid}:${v.thumb_version}`;
      $("tThumb").src = `media.php?thumb=${encodeURIComponent(v.gid)}&v=${v.thumb_version}`;
    }
    if (S.target) renderPath(S.target.sightings, v);
  }

  function renderPath(rows, live) {
    const order = [];
    for (const r of rows || []) if (order[order.length - 1] !== r.camera_id) order.push(r.camera_id);
    const now = new Set(live ? Object.keys(live.cameras_now || {}) : []);
    for (const c of now) if (order[order.length - 1] !== c) order.push(c);
    $("tPath").innerHTML = order.map((c, i) => `${i ? '<span class="camera-path-arrow">→</span>' : ""}<span class="camera-path-item ${now.has(c) && i === order.length - 1 ? "now" : ""}">${esc(c)}</span>`).join("");
  }

  function renderTimeline(rows) {
    const el = $("tTimeline");
    if (!rows.length) { el.innerHTML = `<p class="muted small">No clips saved yet – they are written while the vehicle is tracked.</p>`; return; }
    const t0 = Math.min(...rows.map((r) => r.start_ts)), t1 = Math.max(...rows.map((r) => r.end_ts));
    const span = Math.max(1, t1 - t0);
    const byCam = {};
    rows.forEach((r) => (byCam[r.camera_id] = byCam[r.camera_id] || []).push(r));
    el.innerHTML = Object.keys(byCam).sort().map((cam) => `<div class="tl-row"><span class="mono">${esc(cam)}</span><div class="tl-track">${
      byCam[cam].map((r) => `<span class="tl-seg" title="${r.start_clock != null ? `${fmtClock(r.start_clock)} – ${fmtClock(r.end_clock)}` : `${fmtTime(r.start_ts)} – ${fmtTime(r.end_ts)}`}" style="left:${(100 * (r.start_ts - t0) / span).toFixed(2)}%;width:${Math.max(0.8, 100 * (r.end_ts - r.start_ts) / span).toFixed(2)}%"></span>`).join("")
    }</div></div>`).join("") + `<div class="tl-axis"><span>${fmtTime(t0)}</span><span>${Math.round(span)} s</span><span>${fmtTime(t1)}</span></div>`;
  }

  // ---------------------------------------------------------- journey + replay
  /** The followed vehicle's camera visits (from the sightings in the database), oldest first. */
  async function loadJourney(gid) {
    try {
      const d = await api(`api/journey_steps.php?gid=${encodeURIComponent(gid)}`);
      if (S.follow !== gid && d.gid !== S.follow) return;
      S.journey = { gid: d.gid, steps: d.steps, at: Date.now() };
      renderJourney();
    } catch (e) { /* the target panel shows the error of the main request */ }
  }

  // ------------------------------------------------------- extracted features
  /** What ReID extracted for the followed vehicle: colour/type, the 2048-number embedding,
   *  its stored samples and the most similar other vehicles (api/features.php). */
  async function loadFeatures(gid) {
    S.features = { gid, at: Date.now(), data: S.features && S.features.gid === gid ? S.features.data : null };
    try {
      const d = await api(`api/features.php?gid=${encodeURIComponent(gid)}`);
      if (S.follow !== gid && d.gid !== S.follow) return;
      S.features = { gid, at: Date.now(), data: d };
      renderFeatures(d);
    } catch (e) { $("fAttrs").innerHTML = `<span class="muted small">${esc(e.message)}</span>`; }
  }

  /** value in [-1, 1] -> blue (negative) / dark (0) / orange (positive) */
  function divColor(x) {
    const t = Math.max(-1, Math.min(1, x));
    const base = [17, 24, 39], neg = [59, 111, 216], pos = [239, 138, 47];
    const c = t < 0 ? neg : pos, a = Math.abs(t);
    return base.map((b, i) => Math.round(b + (c[i] - b) * a));
  }

  function renderFeatures(d) {
    const m = d.model || {};
    $("fModel").textContent = m.name ? `${m.name.split(" (")[0]} · ${(m.input || []).join("×")} input` : "";
    const a = d.attributes || {};
    $("fAttrs").innerHTML = [
      `<span class="feat-attr">Type <b>${esc(a.type || "unknown")}</b></span>`,
      `<span class="feat-attr">Colour <b>${a.color ? swatch(a.color) : "unknown"}</b></span>`,
      `<span class="feat-attr">Embedding <b>${m.dim || 0}-d</b></span>`,
      `<span class="feat-attr">Samples <b>${d.samples.length}</b></span>`,
    ].join("");

    // the mean embedding as a 64×32 grid; values scaled by their largest magnitude
    const emb = d.embedding || [];
    const cv = $("fEmb"), ctx = cv.getContext("2d");
    const cols = emb.length > 512 ? 64 : 32, rows = Math.max(1, Math.ceil(emb.length / cols));
    cv.width = cols; cv.height = rows;
    cv.style.aspectRatio = `${cols} / ${rows}`;
    const img = ctx.createImageData(cols, rows);
    const peak = emb.reduce((p, x) => Math.max(p, Math.abs(x)), 1e-6);
    // square-root scale so the many small values stay visible next to the few large ones
    emb.forEach((x, i) => { const [r, g, b] = divColor(Math.sign(x) * Math.sqrt(Math.abs(x) / peak)); img.data.set([r, g, b, 255], i * 4); });
    ctx.putImageData(img, 0, 0);
    if (emb.length) {
      const nz = emb.filter((x) => Math.abs(x) > peak * 0.25).length;
      $("fEmbInfo").textContent = `· mean of ${d.samples.length} crop${d.samples.length === 1 ? "" : "s"}, length 1, ${nz} strong values`;
    } else $("fEmbInfo").textContent = "· no embedding stored yet";

    // pairwise similarity between stored samples
    const P = d.pairwise || [], n = P.length, pc = $("fPair"), pctx = pc.getContext("2d");
    pc.width = Math.max(1, n); pc.height = Math.max(1, n);
    if (n) {
      const pi = pctx.createImageData(n, n);
      // 0.3 (unrelated) .. 1.0 (identical) mapped onto the orange scale
      P.forEach((row, i) => row.forEach((s, j) => {
        const [r, g, b] = divColor(Math.max(0, (s - 0.3) / 0.7));
        pi.data.set([r, g, b, 255], (i * n + j) * 4);
      }));
      pctx.putImageData(pi, 0, 0);
      let sum = 0, cnt = 0;
      P.forEach((row, i) => row.forEach((s, j) => { if (i < j) { sum += s; cnt++; } }));
      $("fSampInfo").textContent = cnt ? `· samples agree ${(sum / cnt).toFixed(2)} on average` : "· one sample";
    } else { pctx.clearRect(0, 0, 1, 1); $("fSampInfo").textContent = ""; }
    $("fSamples").innerHTML = d.samples.slice().reverse().map((s) =>
      `<li><span><b>${esc(s.camera)}</b> ${fmtTime(s.ts)}</span><span title="Similarity to the mean">${s.sim_to_mean.toFixed(2)}</span></li>`).join("");

    // nearest other vehicles with the thresholds the matcher uses
    const thr = m.threshold || 0.75, ho = m.handoff_threshold || thr;
    $("fNearInfo").textContent = `· ID kept above ${ho.toFixed(2)} (hand-off) / ${thr.toFixed(2)}`;
    const pct = (s) => `${Math.max(0, Math.min(100, s * 100)).toFixed(1)}%`;
    $("fNear").innerHTML = (d.nearest || []).map((v) => `<li><button data-gid="${esc(v.gid)}" title="Follow ${esc(v.gid)}">
        <img ${v.thumb ? `src="${esc(v.thumb)}"` : ""} alt="">
        <span class="feat-near-txt"><span class="feat-near-row"><span><span class="gid" style="color:${gidColor(v.gid)}">${esc(v.gid)}</span>
          <span class="muted">${esc([v.color, v.type].filter(Boolean).join(" "))}${v.camera ? ` · ${esc(v.camera)}` : ""}</span></span>
          <span class="mono">${v.similarity.toFixed(2)}</span></span>
          <span class="feat-bar"><span class="${v.similarity >= ho ? "over" : ""}" style="width:${pct(v.similarity)}"></span>
            <i style="left:${pct(ho)}"></i><i style="left:${pct(thr)}"></i></span></span>
      </button></li>`).join("");
  }
  $("fNear").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-gid]");
    if (b) track(b.dataset.gid);
  });

  function fmtDur(s) {
    s = Math.max(0, Math.round(s));
    return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`;
  }

  function renderJourney() {
    const steps = (S.journey && S.journey.steps) || [];
    const live = S.follow ? S.vehicles.get(S.follow) : null;
    const now = new Set(live && live.live ? Object.keys(live.cameras_now || {}) : []);
    const el = $("tJourney");
    el.innerHTML = steps.map((s, i) => {
      const isLive = i === steps.length - 1 && now.has(s.camera);
      const gap = i && s.gap_s !== null
        ? `<li class="journey-gap">${s.gap_s < 0 ? "same moment, another camera" : `↓ ${fmtDur(s.gap_s)} later`}</li>` : "";
      return `${gap}<li><button class="journey-step ${isLive ? "live" : ""}" data-i="${i}" ${s.clip ? "" : "disabled"}
          title="${s.clip ? "Replay the journey from here" : "No recording for this visit"}">
        <span class="n">${i + 1}</span>
        <span><span class="cam">${esc(s.camera)}</span> <span class="muted">${esc(s.camera_name !== s.camera ? s.camera_name : "")}</span>
          <span class="when">${s.start_clock !== null && s.start_clock !== undefined
            ? `${fmtClock(s.start_clock)} – ${isLive ? "now" : fmtClock(s.end_clock)}`
            : `${fmtTime(s.start_ts)} – ${isLive ? "now" : fmtTime(s.end_ts)}`}</span></span>
        <span class="dur">${isLive ? "● LIVE" : fmtDur(s.end_ts - s.start_ts)}</span></button></li>`;
    }).join("");
    el.querySelectorAll(".journey-step").forEach((b) => (b.onclick = () => startReplay(+b.dataset.i)));
    $("btnReplay").disabled = !steps.some((s) => s.clip);
  }
  $("btnReplay").onclick = () => startReplay(0);
  setInterval(() => { if (S.follow && !S.replay) loadJourney(S.follow); }, 5000);

  // boxes are recorded at the AI rate (every 1-3 s per camera on a CPU): bridge gaps up to 4 s
  const MAX_BOX_GAP = 4.0, SPEEDS = [1, 2, 4];
  const rVideo = $("rVideo"), rCanvas = $("rCanvas");

  /** Play the recorded video of every camera the vehicle visited, in order, switching camera
   *  along its path and drawing its box. */
  function startReplay(from = 0) {
    if (!S.journey || !S.journey.steps.length) return;
    S.replay = { gid: S.journey.gid, steps: S.journey.steps, i: -1, speed: S.replay ? S.replay.speed : 1 };
    $("rGid").textContent = S.replay.gid;
    $("rSpeed").textContent = `${S.replay.speed}×`;
    $("replay").classList.remove("hidden");
    setMode("focus");
    $("modeChip").textContent = "[Journey replay · recorded video]";
    renderReplaySteps();
    playStep(from);
    requestAnimationFrame(drawReplay);
  }

  function stopReplay() {
    if (!S.replay) return;
    S.replay = null;
    rVideo.pause();
    rVideo.removeAttribute("src");
    rVideo.load();
    $("replay").classList.add("hidden");
    chooseMain(true);
  }

  function renderReplaySteps() {
    const R = S.replay;
    $("rSteps").innerHTML = R.steps.map((s, i) =>
      `${i ? `<span class="replay-gap">${s.gap_s === null ? "→" : s.gap_s < 0 ? "‖" : `→ ${fmtDur(s.gap_s)} →`}</span>` : ""}` +
      `<button class="replay-step ${i < R.i ? "done" : ""} ${i === R.i ? "current" : ""}" data-i="${i}" ${s.clip ? "" : "disabled"}>${esc(s.camera)}</button>`
    ).join("");
    $("rSteps").querySelectorAll(".replay-step").forEach((b) => (b.onclick = () => playStep(+b.dataset.i)));
  }

  function playStep(i) {
    const R = S.replay;
    if (!R) return;
    while (i < R.steps.length && !R.steps[i].clip) i++; // visits without a recording are skipped
    if (i < 0 || i >= R.steps.length) { finishReplay(); return; }
    const prev = R.steps[R.i], step = R.steps[i];
    R.i = i;
    // each visit is its own short clip (cut on the server): play it from the start
    if (rVideo.getAttribute("src") !== step.clip) rVideo.src = step.clip;
    else rVideo.currentTime = 0;
    // once this clip is loaded, have the server cut the next camera's clip so the switch is quick
    const next = R.steps.slice(i + 1).find((s) => s.clip);
    if (next) rVideo.addEventListener("canplaythrough", () => {
      if (S.replay === R) fetch(next.clip, { headers: { Range: "bytes=0-0" } }).catch(() => {});
    }, { once: true });
    rVideo.defaultPlaybackRate = rVideo.playbackRate = R.speed; // a new src resets the rate to the default
    rVideo.play().catch(() => {});
    $("rPlay").textContent = "⏸";
    $("mainLabel").textContent = `CAM ${step.camera} (recorded)`;
    renderReplaySteps();
    const b = $("switchBanner");
    b.textContent = prev && prev.camera !== step.camera
      ? `Camera switch: ${prev.camera} → ${step.camera}${step.gap_s > 0 ? ` (${fmtDur(step.gap_s)} later)` : ""}`
      : `Journey step ${i + 1}/${R.steps.length}: ${step.camera}`;
    b.classList.remove("hidden");
    clearTimeout(S.bannerTimer);
    S.bannerTimer = setTimeout(() => b.classList.add("hidden"), 3000);
  }

  function finishReplay() {
    const R = S.replay;
    if (!R) return;
    toast(`Journey of ${R.gid}: ${R.steps.map((s) => s.camera).join(" → ")} – back to live.`, "ok");
    stopReplay();
  }

  /** The vehicle's box at video time t, interpolated between recorded detections. */
  function boxAt(boxes, t) {
    if (!boxes || !boxes.length) return null;
    let lo = 0, hi = boxes.length - 1;
    if (t < boxes[0][0] - 0.3 || t > boxes[hi][0] + 0.3) return null;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (boxes[mid][0] <= t) lo = mid; else hi = mid; }
    const a = boxes[lo], b = boxes[hi];
    if (t <= a[0]) return a.slice(1);
    if (t >= b[0]) return b.slice(1);
    if (b[0] - a[0] > MAX_BOX_GAP) return Math.abs(t - a[0]) < 0.4 ? a.slice(1) : Math.abs(t - b[0]) < 0.4 ? b.slice(1) : null;
    const f = (t - a[0]) / (b[0] - a[0]);
    return [1, 2, 3, 4].map((k) => a[k] + (b[k] - a[k]) * f);
  }

  function drawReplay() {
    const R = S.replay;
    if (!R) return;
    const step = R.steps[R.i];
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(rCanvas.clientWidth * dpr), ch = Math.round(rCanvas.clientHeight * dpr);
    if (rCanvas.width !== cw || rCanvas.height !== ch) { rCanvas.width = cw; rCanvas.height = ch; }
    const ctx = rCanvas.getContext("2d");
    ctx.clearRect(0, 0, cw, ch);
    const t = step ? step.clip_from + rVideo.currentTime : 0; // position in the camera recording
    if (step && rVideo.videoWidth) {
      const sc = Math.min(cw / rVideo.videoWidth, ch / rVideo.videoHeight);
      const w = rVideo.videoWidth * sc, h = rVideo.videoHeight * sc, x0 = (cw - w) / 2, y0 = (ch - h) / 2;
      const box = boxAt(step.boxes, t);
      if (box) {
        const [x1, y1, x2, y2] = box;
        const col = gidColor(R.gid);
        ctx.lineWidth = 3 * dpr; ctx.strokeStyle = col;
        ctx.strokeRect(x0 + x1 * w, y0 + y1 * h, (x2 - x1) * w, (y2 - y1) * h);
        ctx.font = `700 ${13 * dpr}px ${getComputedStyle(document.body).fontFamily}`;
        const label = `${R.gid} · ${step.camera}`, tw = ctx.measureText(label).width + 10 * dpr;
        ctx.fillStyle = col; ctx.fillRect(x0 + x1 * w, y0 + y1 * h - 20 * dpr, tw, 20 * dpr);
        ctx.fillStyle = "#0a0e17"; ctx.fillText(label, x0 + x1 * w + 5 * dpr, y0 + y1 * h - 6 * dpr);
      }
      const into = t - step.start_offset;
      const hasClock = step.start_clock !== null && step.start_clock !== undefined;
      const clockText = hasClock ? fmtClock(Math.max(0, step.start_clock + into)) : fmtTime(step.start_ts + Math.max(0, into));
      $("rClock").textContent = `${step.camera} · ${clockText} · step ${R.i + 1}/${R.steps.length}`;
      if (hasClock && step.mask) {
        // the camera's own timestamp is covered in the clip; the shared clock is written in its place
        const [mx1, my1, mx2, my2] = step.mask;
        const bx = x0 + mx1 * w, by = y0 + my1 * h, bw = (mx2 - mx1) * w, bh = (my2 - my1) * h;
        ctx.fillStyle = "#121212"; ctx.fillRect(bx, by, bw, bh);
        ctx.font = `700 ${Math.max(10, bh * 0.72)}px ui-monospace, Consolas, monospace`;
        ctx.fillStyle = "#ffffff"; ctx.textBaseline = "middle";
        ctx.fillText(clockText, bx + bh * 0.3, by + bh / 2);
        ctx.textBaseline = "alphabetic";
      }
    }
    requestAnimationFrame(drawReplay);
  }

  // next camera once this visit's clip has played
  rVideo.addEventListener("ended", () => { if (S.replay) playStep(S.replay.i + 1); });
  rVideo.addEventListener("error", () => {
    if (!S.replay) return;
    toast(`Recording of ${S.replay.steps[S.replay.i].camera} could not be played – skipping.`, "err");
    playStep(S.replay.i + 1);
  });
  $("rPlay").onclick = () => {
    if (rVideo.paused) { rVideo.play().catch(() => {}); $("rPlay").textContent = "⏸"; }
    else { rVideo.pause(); $("rPlay").textContent = "▶"; }
  };
  $("rPrev").onclick = () => S.replay && playStep(Math.max(0, S.replay.i - 1));
  $("rNext").onclick = () => S.replay && playStep(S.replay.i + 1);
  $("rSpeed").onclick = () => {
    if (!S.replay) return;
    S.replay.speed = SPEEDS[(SPEEDS.indexOf(S.replay.speed) + 1) % SPEEDS.length];
    rVideo.defaultPlaybackRate = rVideo.playbackRate = S.replay.speed;
    $("rSpeed").textContent = `${S.replay.speed}×`;
  };
  $("rClose").onclick = stopReplay;

  function onMerge(m) {
    if (S.follow === m.from) { S.follow = null; track(m.to); }
    toast(`${m.from} was recognised as ${m.to} – records merged.`);
  }

  // ----------------------------------------------------------------- gallery
  function onVehicles(items) {
    S.vehicles = new Map(items.map((v) => [v.gid, v]));
    if (S.tab === "live") renderGallery();
    if (S.follow) { const v = S.vehicles.get(S.follow); if (v) fillLive(v); if (!S.focus) chooseMain(); updateCells(); }
  }

  function vehicleCard(v, live) {
    const cams = live ? (v.live ? Object.keys(v.cameras_now || {}) : (v.search_from ? [v.search_from] : [])) : (v.cameras || []);
    const thumb = live ? (v.thumb_version ? `media.php?thumb=${encodeURIComponent(v.gid)}&v=${v.thumb_version}` : "") : (v.thumb || "");
    const el = document.createElement("button");
    el.className = "vehicle-card" + (v.gid === S.follow ? " tracked" : "");
    el.style.setProperty("--sw", gidColor(v.gid));
    el.innerHTML = `
      ${thumb ? `<img class="vthumb" loading="lazy" src="${esc(thumb)}" alt="">` : `<span class="vthumb"></span>`}
      <span><span class="vehicle-id" style="color:${gidColor(v.gid)}">${esc(v.gid)}</span>
        <span class="vehicle-meta" style="display:block">${esc(v.class || v.vclass || "vehicle")} · ${swatch(v.color)} · ${esc(v.motion || "")}</span>
        <span class="vehicle-cameras">${cams.map((c) => `<span class="cam-badge ${live ? "active" : ""}">${esc(c)}</span>`).join("")}</span></span>
      <span class="ago ${live && v.live ? "live" : live && v.searching ? "searching" : ""}">${live && v.live ? "● LIVE"
        : live && v.searching ? `◌ Searching${(v.expected || []).length ? ` ${esc(v.expected.join(", "))}` : ""} · ${Math.floor(v.search_left / 60)}:${String(v.search_left % 60).padStart(2, "0")}`
        : ago(v.last_seen)}</span>`;
    el.onclick = () => track(v.gid);
    return el;
  }

  function matches(v, f) {
    return !f || [v.gid, v.class, v.vclass, v.color, v.motion, ...(v.cameras || []), ...Object.keys(v.cameras_now || {})].join(" ").toLowerCase().includes(f);
  }

  function renderGallery() {
    const f = $("search").value.trim().toLowerCase();
    const list = $("vehicleList");
    if (S.tab === "requests") return;
    const live = S.tab === "live";
    // live tab: vehicles in view, then those that left and are still being looked for on other cameras
    const items = live ? [...S.vehicles.values()].filter((v) => v.live || v.searching).sort((a, b) => (b.live - a.live)) : S.history;
    const shown = items.filter((v) => matches(v, f));
    if (live) $("liveCount").textContent = items.length;
    if (!shown.length) {
      list.innerHTML = `<div class="empty-list">${live ? "Scanning video frames for vehicles…" : "No vehicles match."}</div>`;
      return;
    }
    list.replaceChildren(...shown.map((v) => vehicleCard(v, live)));
  }

  let searchTimer;
  $("search").oninput = () => {
    if (S.tab === "history") { clearTimeout(searchTimer); searchTimer = setTimeout(loadHistory, 250); } else renderGallery();
  };
  async function loadHistory() {
    try { S.history = (await api(`api/vehicles.php?q=${encodeURIComponent($("search").value.trim())}`)).items; renderGallery(); }
    catch (e) { toast(e.message, "err"); }
  }

  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => {
    S.tab = t.dataset.tab;
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $("vehicleList").classList.toggle("hidden", S.tab === "requests");
    $("search").classList.toggle("hidden", S.tab === "requests");
    $("jobList").classList.toggle("hidden", S.tab !== "requests");
    if (S.tab === "history") loadHistory();
    else if (S.tab === "requests") loadJobs();
    else renderGallery();
  }));

  // ---------------------------------------------------------------- journeys
  $("btnJourney").onclick = () => {
    if (!S.follow) return;
    $("jGid").textContent = S.follow;
    try { $("jEmail").value = localStorage.getItem("mcvt.email") || ""; } catch (e) {}
    $("jError").classList.add("hidden");
    $("journeyDialog").showModal();
    $("jEmail").focus();
  };
  $("journeyForm").addEventListener("submit", async (ev) => {
    if (ev.submitter && ev.submitter.value !== "send") return;
    ev.preventDefault();
    const email = $("jEmail").value.trim(), gid = S.follow;
    $("jSend").disabled = true;
    try {
      const r = await api("api/journey.php", { method: "POST", body: JSON.stringify({ gid, email }) });
      try { if (email) localStorage.setItem("mcvt.email", email); } catch (e) {}
      $("journeyDialog").close();
      toast(`Request #${r.id} queued: journey video of ${gid}${email ? ` for ${email}` : ""}.`, "ok");
      refreshTarget(true); loadJobs();
    } catch (e) {
      $("jError").textContent = e.message; $("jError").classList.remove("hidden");
    } finally { $("jSend").disabled = false; }
  });

  function renderJobs(jobs, el) {
    el.innerHTML = jobs.length ? jobs.map((j) => `<li>
      <span class="status ${esc(j.status)}">${esc(j.status.replace("_", " "))}</span> · #${j.id}${j.gid ? ` · <span class="mono">${esc(j.gid)}</span>` : ""}
      <div class="muted small">${esc(j.message || "")}</div>
      ${j.video ? `<a href="${esc(j.video)}" target="_blank" rel="noopener">Watch</a> · <a href="${esc(j.video)}&download">Download</a>` : ""}
    </li>`).join("") : `<li class="muted">No requests yet.</li>`;
  }
  async function loadJobs() { try { renderJobs((await api("api/jobs.php")).items, $("jobList")); } catch (e) {} }
  function onJob(m) {
    const kind = ["sent", "done"].includes(m.status) ? "ok" : ["failed", "email_failed"].includes(m.status) ? "err" : "";
    if (kind) toast(`Request #${m.id} (${m.gid}): ${m.message}`, kind);
    if (S.tab === "requests") loadJobs();
    if (S.follow === m.gid) refreshTarget(true);
  }

  // ---------------------------------------------------- stats + event log
  function onStats(m) {
    m.cameras.forEach((c) => {
      const cam = S.cams.get(c.id) || upsertCamera({ id: c.id, name: c.id, status: c.status });
      cam.stats = c; cam.info.status = c.status;
      cam.cell.classList.toggle("offline", c.status === "offline");
      const waiting = c.status === "waiting";
      cam.cell.classList.toggle("waiting", waiting);
      const w = cam.cell.querySelector(".cell-wait");
      w.classList.toggle("hidden", !waiting);
      if (waiting) {
        const s = Math.max(0, c.starts_in || 0);
        w.innerHTML = `<b>Recording starts at ${esc(c.start_clock || "a later time")}</b><span>in ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}</span>`;
      }
      updateCellText(cam);
    });
    const e = m.engine || {};
    $("statKnown").textContent = e.vehicles_known ?? 0;
    $("statLive").textContent = e.in_view ?? 0;
    $("statCross").textContent = e.cross_camera_matches ?? 0;
    $("statNew").textContent = e.new_signatures ?? 0;
    $("perf").textContent = `YOLOv8 ${e.det_ms ?? "–"} ms/frame · ReID ${e.reid_ms ?? "–"} ms/crop · ${e.live_tracks ?? 0} tracks`
      + (S.delay ? ` · view delayed ${S.delay}s for exact boxes` : "");
  }

  function addLog(m) {
    const msg = String(m.message || "");
    let kind = m.level === "error" ? "error" : m.level === "warning" ? "warning" : "";
    if (!kind) kind = /switch|falling back/i.test(msg) ? "switch" : /^New vehicle/.test(msg) ? "new" : /re-identified|same as/i.test(msg) ? "match" : "";
    const el = document.createElement("div");
    el.className = `event-item ${kind}`;
    el.innerHTML = `<span class="event-time">${fmtTime(m.ts)}</span><span class="event-text">${esc(msg)}</span>`;
    const log = $("log");
    log.prepend(el);
    while (log.children.length > 200) log.lastChild.remove();
  }

  window.addEventListener("resize", () => { S.dirty = true; S.cams.forEach(drawCell); });
  setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString([], { hour12: false }); }, 1000);
  api("api/logs.php?limit=60").then((d) => d.items.forEach(addLog)).catch(() => {});
  renderGallery();
  connect();
  requestAnimationFrame(render);
})();
