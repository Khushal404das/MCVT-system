"""Real-time multi-camera pipeline (detection -> tracking -> ReID -> switching).

Runs one processing loop over the latest frame of every camera, keeps the
vehicle database up to date and streams frames + events to the WebSocket hub.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from . import logbus
from .attributes import AttributeGate, ColorNamer, VoteBox
from .bridge import Bridge
from .camera import CameraStream, FramePacket, PlaybackClock, assign_resolution_weights
from .config import Config, format_clock
from .db import Database
from .detector import VehicleDetector, resolve_device
from .flow import FlowBoxes
from .jobs import JourneyWorker
from .matcher import CrossCameraMatcher, TrackIdentity
from .reid.extractor import FeatureExtractor, color_histogram
from .reid.sequence import SequenceAggregator
from .switching import CameraSwitchingEngine, OverlapDetection, camera_score
from .tracker import CameraTracker, TrackState

log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- state
@dataclass
class TrackRecord:
    """Everything the pipeline knows about one local track."""

    camera_id: str
    track_id: str
    session_id: str
    identity: TrackIdentity
    first_ts: float
    first_offset: float
    last_ts: float = 0.0
    last_offset: float = 0.0
    frames: int = 0
    best_score: float = 0.0
    last_sample_cycle: int = -10_000
    pending_reid: bool = False
    first_center: tuple[float, float] | None = None
    max_shift: float = 0.0
    moving: bool = False  # is this vehicle moving (vs. parked / stopped)?
    last_motion_ts: float = 0.0
    prev_center: tuple[float, float] | None = None
    labels: VoteBox = field(default_factory=VoteBox)
    last_box: np.ndarray | None = None  # normalised box at the last AI update (for overlay extrapolation)
    last_box_time: float = 0.0
    velocity: np.ndarray | None = None  # normalised box change per second
    ended_at: float = 0.0
    last_hist: np.ndarray | None = None  # appearance at the last detection (for re-linking a new track)
    chain: str = ""  # stable key of this vehicle's record while its tracker tracks are re-linked
    path: list = field(default_factory=list)  # [video offset s, x1, y1, x2, y2 (0..1)] per detection (journey replay)

    @property
    def gid(self) -> str | None:
        return self.identity.gid


@dataclass
class VehicleLive:
    """Aggregated live view of one global vehicle id."""

    gid: str
    first_seen: float
    last_seen: float = 0.0
    labels: VoteBox = field(default_factory=VoteBox)
    colors: VoteBox = field(default_factory=VoteBox)
    cameras_seen: dict[str, float] = field(default_factory=dict)
    now: dict[str, dict] = field(default_factory=dict)  # camera -> {score, box, tid}
    thumb_score: float = 0.0
    thumb_saved_at: float = 0.0
    thumb_version: int = 0
    moving: bool = False
    primary: str | None = None
    dirty: bool = True
    # left every camera: kept "searching" for reid.search_seconds while the other cameras are checked
    search_since: float = 0.0       # wall time it left the last camera (0 = not searching)
    search_from: str | None = None  # camera it left
    search_clock: str | None = None  # shared clock when it left
    search_logged: bool = False
    # current visit per camera: wall time and shared clock when it came into that camera's view
    visit_start: dict[str, float] = field(default_factory=dict)
    visit_clock: dict[str, str] = field(default_factory=dict)


@dataclass
class ReIDRequest:
    camera_id: str
    track_id: str
    crop: np.ndarray
    ts: float


class ReIDWorker(threading.Thread):
    """Runs the ResNet-IBN extractor off the main loop (CPU-friendly)."""

    def __init__(self, extractor: FeatureExtractor, batch: int, threads: int = 0):
        super().__init__(name="reid", daemon=True)
        self.extractor = extractor
        self.batch = max(1, batch)
        self.threads = int(threads)  # CPU threads for ReID; the rest stay free for YOLOv8
        # Priority queue: moving vehicles without an id first, parked-car pre-samples last.
        self.inbox: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._results: list[tuple[ReIDRequest, np.ndarray]] = []
        self._lock = threading.Lock()
        self.ms_per_crop = 0.0

    def run(self) -> None:
        if self.threads > 0:
            torch.set_num_threads(self.threads)  # per-thread setting: does not change the detector's
        while True:
            reqs = [self.inbox.get()[2]]
            while len(reqs) < self.batch:
                try:
                    reqs.append(self.inbox.get_nowait()[2])
                except queue.Empty:
                    break
            t = time.perf_counter()
            try:
                feats = self.extractor.extractFeatures([r.crop for r in reqs], [r.camera_id for r in reqs])
            except Exception:
                log.exception("ReID extraction failed")
                continue
            self.ms_per_crop = (time.perf_counter() - t) * 1000 / len(reqs)
            with self._lock:
                self._results.extend(zip(reqs, feats))

    def submit(self, priority: int, req: ReIDRequest) -> None:
        self._seq += 1
        self.inbox.put((priority, self._seq, req))

    def drain(self) -> list[tuple[ReIDRequest, np.ndarray]]:
        with self._lock:
            out, self._results = self._results, []
        return out


# ------------------------------------------------------------------------ pipeline
class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.root
        self.storage = cfg.path("paths.storage")
        self.thumb_dir = self.storage / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.db = Database(cfg.path("paths.database"), self.root / "schema.sql")
        try:  # the web app cuts journey-replay clips with the same ffmpeg (web/public/media.php)
            from .stitcher import find_ffmpeg
            (cfg.path("paths.storage") / "ffmpeg_path.txt").write_text(find_ffmpeg(cfg.get("stitching.ffmpeg", "auto")))
        except Exception as e:  # noqa: BLE001 - replay clips are optional
            log.warning("ffmpeg not found (%s); journey replay clips will not be available.", e)
        logbus.setup(cfg.path("paths.logs"), self.db)
        self.engine_session = uuid.uuid4().hex[:8]
        self.stop_event = threading.Event()

        # ---- cameras
        realtime = bool(cfg.get("cameras.realtime", True))
        loop = bool(cfg.get("cameras.loop_files", True))
        reconnect = float(cfg.get("cameras.reconnect_seconds", 5))
        self.cameras: dict[str, CameraStream] = {}
        self.playback_clock = PlaybackClock() if cfg.get("cameras.sync_playback", True) else None
        for spec in cfg.camera_specs():
            cam = CameraStream(spec, realtime=realtime, loop=loop, reconnect_s=reconnect, clock=self.playback_clock,
                               motion_width=int(cfg.get("detection.motion_mask_width", 320))
                               if cfg.get("detection.moving_only", True) else 0,
                               motion_history_s=float(cfg.get("detection.motion_history_seconds", 20)),
                               motion_fps=float(cfg.get("detection.motion_mask_fps", 6)))
            if not cam.probe():
                logbus.warning("camera", f"Camera {spec.id} ({spec.source}) cannot be opened; it will be retried.")
            self.cameras[cam.id] = cam
        if not self.cameras:
            raise SystemExit("No cameras configured or discovered - check 'cameras' in config.yaml")
        self._apply_start_clocks()
        assign_resolution_weights(list(self.cameras.values()), cfg.get("cameras.resolution_weight_mode", "sqrt_pixels"))

        # ---- models
        device = resolve_device(cfg.get("detection.device", "auto"))
        self.detector = VehicleDetector(
            cfg.path("detection.model"), cfg.get("detection.confidence", 0.6),
            cfg.get("detection.classes", ["car", "motorcycle", "bus", "truck"]),
            cfg.get("detection.imgsz", 640), device, cfg.get("detection.min_box_area_ratio", 0.0),
            bool(cfg.get("detection.agnostic_nms", True)), float(cfg.get("detection.nms_iou", 0.7)),
        )
        adapter = cfg.path("reid.adapter") if cfg.get("reid.adapter") else None
        self.extractor = FeatureExtractor(cfg.path("reid.weights"), cfg.get("reid.input_size", [256, 256]), device, adapter)
        if adapter and self.extractor.adapter is None:
            logbus.warning("reid", f"ReID adapter {adapter} not found - using the plain ResNet-IBN features.")
        appearance = cfg.get("tracking.appearance", "auto")
        self.reid_for_tracking = appearance == "reid" or (appearance == "auto" and device.startswith("cuda"))
        self.reid_worker = None if self.reid_for_tracking else ReIDWorker(
            self.extractor, int(cfg.get("reid.worker_batch", 1)), int(cfg.get("reid.cpu_threads", 0) or 0))
        logbus.info("engine", f"Models loaded on {device.upper()}: YOLOv8 detector, ResNet-IBN ReID. "
                              f"DeepSORT appearance = {'ResNet-IBN' if self.reid_for_tracking else 'colour histogram'}.")

        # ---- trackers / matcher / switching
        self.trackers = {cid: CameraTracker(cid, cfg.get("tracking.max_age", 30), cfg.get("tracking.n_init", 3),
                                            cfg.get("tracking.max_iou_distance", 0.7),
                                            cfg.get("tracking.max_cosine_distance", 0.3))
                         for cid in self.cameras}
        # Thresholds are set for the plain ResNet-IBN similarity scale. With a camera adapter the
        # scale changes, so each is converted to accept the same share of different-vehicle pairs.
        ad = self.extractor.adapter
        thr = {k: float(cfg.get(f"reid.{k}", d)) for k, d in (("similarity_threshold", 0.75), ("handoff_threshold", 0.6),
                                                             ("follow_threshold", 0.55), ("follow_min_similarity", 1.0))}
        margins = {k: float(cfg.get(f"reid.{k}", d)) for k, d in (("merge_margin", 0.05), ("follow_margin", 0.08))}
        # Colour (camera-cast corrected) and type of each vehicle, stored with its id and used to
        # rule out ReID matches with a clearly different colour/type (reid.attributes).
        attr_cfg = cfg.get("reid.attributes") or {}
        self.color_namer = ColorNamer(attr_cfg.get("color_naming") or {},
                                      cfg.path("paths.storage") / "color_cast.json")
        self.attr_gate = AttributeGate(attr_cfg)
        margins["steal_margin"] = float(cfg.get("reid.steal_margin", 0.05))
        if ad is not None:
            thr = {k: (ad.threshold(v) if v < 1.0 else v) for k, v in thr.items()}
            # A threshold measured directly in the adapter's space (engine/training) replaces the
            # converted ones: the conversion only matches the share of different-car pairs and
            # gave ~0.2, which let far too many look-alikes through.
            at = cfg.get("reid.adapter_threshold")
            if at:
                for k in ("similarity_threshold", "handoff_threshold", "follow_threshold"):
                    thr[k] = float(at)
            margins = {k: ad.margin(v) for k, v in margins.items()}
            thr.update(margins)
            logbus.info("reid", f"ReID camera adapter loaded ({ad.info.get('method')}, {ad.dim}-d); thresholds: "
                                + ", ".join(f"{k} {v:.2f}" for k, v in thr.items()))
        # Sequence ReID (GRU over a track's crop features). Its signatures have their own similarity
        # scale: every threshold set for the plain average is converted to accept the same share of
        # different-vehicle pairs (measured on our cameras' test tracks when it was trained), unless
        # reid.sequence_threshold gives the id threshold directly.
        self.seq_model = None
        sim_keys = {"overlap_min_similarity": 0.10, "route_min_similarity": 0.10, "route_feature_min": 0.35}
        sims_cfg = {k: float(cfg.get(f"reid.{k}", d)) for k, d in sim_keys.items()}
        route_margins = {k: float(cfg.get(f"reid.{k}", d)) for k, d in (("route_margin", 0.08), ("route_override", 0.15))}
        if cfg.get("reid.sequence_model"):
            sp = cfg.path("reid.sequence_model")
            if sp.exists():
                self.seq_model = SequenceAggregator(sp)
                sq = self.seq_model
                thr = {k: sq.convert(v) for k, v in thr.items()}
                st = cfg.get("reid.sequence_threshold")
                if st:
                    for k in ("similarity_threshold", "handoff_threshold", "follow_threshold"):
                        thr[k] = float(st)
                margins = {k: sq.convert_margin(v) for k, v in margins.items()}
                sims_cfg = {k: sq.convert(v) for k, v in sims_cfg.items()}
                route_margins = {k: sq.convert_margin(v) for k, v in route_margins.items()}
                logbus.info("reid", f"Sequence ReID (GRU over each track's crops, {sq.dim}-d) loaded; thresholds: "
                                    + ", ".join(f"{k} {v:.2f}" for k, v in {**thr, **margins, **sims_cfg,
                                                                           **route_margins}.items()))
            else:
                logbus.warning("reid", f"Sequence ReID model {sp} not found - using the average of the crop features.")
        self.matcher = CrossCameraMatcher(thr["similarity_threshold"], cfg.get("reid.gallery_size", 30),
                                          margins["merge_margin"], self.db,
                                          handoff_threshold=thr["handoff_threshold"],
                                          follow_threshold=thr["follow_threshold"],
                                          handoff_window=cfg.get("reid.handoff_window_seconds", 90),
                                          follow_min=thr["follow_min_similarity"],
                                          follow_margin=margins["follow_margin"],
                                          overlaps=cfg.get("reid.overlapping_cameras") or [],
                                          gate=self.attr_gate if self.attr_gate.enabled else None,
                                          live_gap=cfg.get("reid.live_gap_seconds", 2.0),
                                          steal_margin=margins["steal_margin"],
                                          overlap_link=cfg.get("reid.overlap_link_seconds", 0.0),
                                          overlap_min_sim=sims_cfg["overlap_min_similarity"],
                                          overlap_arrival=float(cfg.get("reid.overlap_arrival_seconds", 6.0)),
                                          route=cfg.get("reid.route") or [],
                                          route_max_seconds=float(cfg.get("reid.route_max_seconds", 60)),
                                          route_back_seconds=float(cfg.get("reid.route_back_seconds", 20)),
                                          route_skip=int(cfg.get("reid.route_skip", 1)),
                                          route_min_sim=sims_cfg["route_min_similarity"],
                                          route_margin=route_margins["route_margin"],
                                          route_override=route_margins["route_override"],
                                          route_fifo=bool(cfg.get("reid.route_fifo", True)),
                                          route_typical_seconds=float(cfg.get("reid.route_typical_seconds", 15)),
                                          route_learn_min=int(cfg.get("reid.route_learn_min", 5)),
                                          route_feature_min=sims_cfg["route_feature_min"],
                                          seq=self.seq_model,
                                          route_bidirectional=bool(cfg.get("reid.route_bidirectional", True)))
        self.handoff: dict[str, dict] = {}  # followed gid -> hand-off search state
        # a vehicle that left every camera stays listed / searched for this long (reid.search_seconds)
        self.search_seconds = float(cfg.get("reid.search_seconds", 300))
        if cfg.get("reid.load_gallery_on_start", True):
            max_age_h = float(cfg.get("reid.gallery_max_age_hours", 72))
            since = time.time() - max_age_h * 3600
            # Features stored before the current ReID weights/adapter were installed come from another
            # model (same size, different meaning): they cannot be compared, so they are not loaded.
            for key in ("reid.weights", "reid.adapter"):
                p = cfg.get(key) or ""
                if p and cfg.path(key).exists():
                    since = max(since, cfg.path(key).stat().st_mtime)
            stored = self.db.load_gallery(since, cfg.get("reid.gallery_size", 30))
            dim = self.extractor.dim  # features of another ReID setup (with/without adapter) cannot be compared
            stored = {g: [v for v in vs if v.size == dim] for g, vs in stored.items()}
            self.matcher.load({g: vs for g, vs in stored.items() if vs})
            self.matcher.load_attributes({g: a for g, a in self.db.load_vehicle_attributes(since).items()
                                          if g in self.matcher.gallery})
            if self.matcher.gallery:
                logbus.info("reid", f"Loaded {len(self.matcher.gallery)} known vehicles from the feature database.")
        else:
            logbus.info("reid", "Starting with an empty ReID gallery (reid.load_gallery_on_start is off).")
        self.switchers: dict[str, CameraSwitchingEngine] = {}
        self.focus_gids: set[str] = set()
        self._focus_lock = threading.Lock()

        self.records: dict[tuple[str, str], TrackRecord] = {}
        self.vehicles: dict[str, VehicleLive] = {}
        self.cam_tracks: dict[str, list[TrackState]] = {cid: [] for cid in self.cameras}
        self.cam_frame_shape: dict[str, tuple[int, int]] = {}
        self.cam_proc_count: dict[str, deque] = {cid: deque(maxlen=60) for cid in self.cameras}
        self.cycle = 0
        self.det_ms = 0.0
        self.cam_visible: dict[str, set[str]] = {}
        self.cam_moving: dict[str, list[TrackState]] = {}
        self.ended_pending: dict[tuple[str, str], TrackRecord] = {}  # ended tracks awaiting a ReID result
        self.track_alias: dict[tuple[str, str], tuple[str, str]] = {}  # replaced track -> re-linked track
        self.dup_hits: dict[str, dict[tuple[str, str], int]] = {}  # camera -> overlapping track pair -> checks
        self.moving_only = bool(cfg.get("detection.moving_only", True))
        self.journey_max_points = int(cfg.get("pipeline.journey_max_points", 400))  # boxes kept per camera visit
        self.overlays: dict[str, tuple] = {}
        self.overlay_hist: dict[str, deque] = {}  # camera -> recent AI results (frame ts, {track: item})
        self._overlay_lock = threading.Lock()
        self.view_cams: set[str] = set()  # cameras some operator has in the main (focus) view
        self.grid_cams: set[str] = set()  # cameras some operator watches in the grid
        self.counters = {"new_signatures": 0, "reid_matches": 0, "cross_camera_matches": 0}
        self._motion_ref: dict[str, np.ndarray] = {}
        self._last_processed: dict[str, float] = {}

        # ---- bridge to PHP hub
        ws_host = cfg.get("server.ws_host", "127.0.0.1")
        ws_host = "127.0.0.1" if ws_host in ("0.0.0.0", "::") else ws_host
        self.bridge = Bridge(f"ws://{ws_host}:{cfg.get('server.ws_port', 8090)}/",
                             os.environ.get("ENGINE_TOKEN", ""), self._on_command, self._on_bridge_connect)
        logbus.add_listener(lambda level, src, msg: self.bridge.send_event(
            {"type": "log", "level": level, "source": src, "message": msg, "ts": time.time()}))
        self.jobs = JourneyWorker(cfg, self.db, self.bridge, self.stop_event)

    # ------------------------------------------------------------------ bridge
    def _on_bridge_connect(self) -> None:
        self.bridge.send_event({"type": "cameras", "items": [c.info() for c in self.cameras.values()]})
        self.bridge.send_event({"type": "engine", "status": "online", "session": self.engine_session})

    def _on_command(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "focus_set":
            gids = {str(g) for g in msg.get("gids", []) if g}
            with self._focus_lock:
                for g in gids - self.focus_gids:
                    logbus.info("switching", f"Operator started following vehicle {g} across all cameras.")
                self.focus_gids = gids
                self.matcher.followed = set(gids)
        elif kind == "view_set":
            self.view_cams = {str(c) for c in msg.get("cams", []) if c}
            self.grid_cams = {str(c) for c in msg.get("grid", []) if c}
        elif kind == "cameras_request":
            self._on_bridge_connect()

    # -------------------------------------------------------------------- run
    def run(self) -> None:
        for cam in self.cameras.values():
            self.db.upsert_camera(cam.info())
        if self.playback_clock is not None:
            self.playback_clock.restart()  # frame 0 of every video = now
            logbus.info("camera", "All camera videos start from the beginning together.")
        for cam in self.cameras.values():
            cam.start()
        if self.reid_worker:
            self.reid_worker.start()
        self.bridge.start()
        self.jobs.start()
        FrameStreamer(self).start()
        logbus.info("engine", f"Engine started with {len(self.cameras)} camera(s): {', '.join(self.cameras)}.")

        det_threads = int(self.cfg.get("detection.cpu_threads", 0) or 0)
        if det_threads > 0:
            torch.set_num_threads(det_threads)  # this (detection) thread only
        max_fps = float(self.cfg.get("pipeline.max_process_fps", 10))
        min_period = 1.0 / max_fps if max_fps > 0 else 0
        t_flush = t_vehicles = t_stats = time.time()
        last_status = {c: c_.status for c, c_ in self.cameras.items()}
        try:
            while not self.stop_event.is_set():
                t0 = time.time()
                fresh = [p for p in (c.latest() for c in self.cameras.values()) if p is not None]
                packets = self._schedule([p for p in fresh if self._should_process(p)])
                if packets:
                    self._process(packets)
                self._apply_reid_results()
                self._update_switching()
                now = time.time()
                if now - t_flush >= float(self.cfg.get("pipeline.db_flush_interval", 1.0)):
                    self._flush()
                    t_flush = now
                if now - t_vehicles >= float(self.cfg.get("pipeline.vehicle_broadcast_interval", 1.0)):
                    self._broadcast_vehicles()
                    t_vehicles = now
                if now - t_stats >= 2.0:
                    self._broadcast_stats()
                    for cid, cam in self.cameras.items():
                        if cam.status != last_status[cid]:
                            self.db.set_camera_status(cid, cam.status)
                            last_status[cid] = cam.status
                            self.bridge.send_event({"type": "cameras", "items": [c.info() for c in self.cameras.values()]})
                    t_stats = now
                spent = time.time() - t0
                if not packets:
                    time.sleep(0.005)
                elif spent < min_period:
                    time.sleep(min_period - spent)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.stop_event.set()
        for cam in self.cameras.values():
            cam.stop()
        try:
            self._flush()
        except Exception:
            log.exception("final flush failed")
        self.bridge.send_event({"type": "engine", "status": "offline"})
        time.sleep(0.3)
        self.bridge.stop()
        logbus.info("engine", "Engine stopped.")

    # --------------------------------------------------------------- process
    def _should_process(self, pkt: FramePacket) -> bool:
        """Motion-gated scheduling: busy cameras every cycle, still ones periodically."""
        if not self.cfg.get("pipeline.motion_gating", True):
            return True
        now = time.time()
        if pkt.motion is not None:
            # Background model: is there a moving blob big enough to be a vehicle? (Swaying trees and
            # noise give only small scattered blobs.) Measured: busy gate cameras ~50% of the time,
            # the others 15-25%, so the AI goes to the cameras where something is driving.
            n, _, stats, _ = cv2.connectedComponentsWithStats(pkt.motion, connectivity=8)
            big = float(stats[1:, cv2.CC_STAT_AREA].max()) / pkt.motion.size if n > 1 else 0.0
            moving = big >= float(self.cfg.get("pipeline.motion_blob_fraction", 0.002))
        else:
            small = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(pkt.frame, cv2.COLOR_BGR2GRAY), (160, 90)), (5, 5), 0)
            prev = self._motion_ref.get(pkt.camera_id)
            self._motion_ref[pkt.camera_id] = small
            moving = prev is None or float((cv2.absdiff(small, prev) > 18).mean()) > float(self.cfg.get("pipeline.motion_threshold", 0.006))
        # Keep analysing while a moving vehicle is on this camera (its box must not drop out).
        # Lost parked cars do not count: they kept every camera busy all the time.
        tracking = bool(self.cam_moving.get(pkt.camera_id)) if self.moving_only else \
            any(not t.detected for t in self.cam_tracks.get(pkt.camera_id, []))
        idle_due = now - self._last_processed.get(pkt.camera_id, 0) >= float(self.cfg.get("pipeline.idle_camera_interval", 2.0))
        return bool(moving or tracking or idle_due)

    @staticmethod
    def _motion_fraction(mask: np.ndarray | None, box, w: int, h: int) -> float:
        """Share of moving (foreground) pixels inside a box, from the camera's background model."""
        if mask is None:
            return 0.0
        mh, mw = mask.shape[:2]
        x1, y1, x2, y2 = box
        a, b = int(x1 / w * mw), int(y1 / h * mh)
        c, d = max(a + 1, int(x2 / w * mw)), max(b + 1, int(y2 / h * mh))
        roi = mask[max(0, b):d, max(0, a):c]
        return float((roi > 0).mean()) if roi.size else 0.0

    def _priority_cameras(self) -> set[str]:
        """Cameras an operator is looking at, plus any that sees (or should soon see) a followed vehicle."""
        cams = set(self.view_cams)
        with self._focus_lock:
            focus = set(self.focus_gids)
        for gid in focus:
            v = self.vehicles.get(gid)
            if v is not None:
                cams.update(v.now)
            h = self.handoff.get(gid)
            if h and h.get("status") == "searching":
                cams.update(e["camera"] for e in self.matcher.expected_next(h.get("from") or ""))
        return cams

    def _schedule(self, packets: list[FramePacket]) -> list[FramePacket]:
        """Pick the cameras analysed this cycle: priority cameras every cycle, the others in turn
        (longest-waiting first). Fewer frames per cycle = more AI updates for the camera on screen."""
        limit = int(self.cfg.get("pipeline.max_cameras_per_cycle", 3))
        prio = self._priority_cameras()
        first = [p for p in packets if p.camera_id in prio]
        rest = sorted((p for p in packets if p.camera_id not in prio),
                      key=lambda p: self._last_processed.get(p.camera_id, 0))
        chosen = first + rest[: max(1 if rest else 0, limit - len(first))] if limit > 0 else packets
        now = time.time()
        for p in chosen:
            self._last_processed[p.camera_id] = now
        return chosen

    def _process(self, packets: list[FramePacket]) -> None:
        self.cycle += 1
        t = time.perf_counter()
        all_dets = self.detector.detect_batch([p.frame for p in packets])
        self.det_ms = (time.perf_counter() - t) * 1000 / max(1, len(packets))
        min_crop = int(self.cfg.get("reid.min_crop_size", 28))
        interval = int(self.cfg.get("reid.sample_interval", 4))
        budget = int(self.cfg.get("reid.max_crops_per_cycle", 8))
        moving_only = self.moving_only
        move_min = float(self.cfg.get("detection.moving_min_shift", 0.02))
        move_frac = float(self.cfg.get("detection.moving_box_fraction", 0.3))
        presamples = int(self.cfg.get("reid.parked_samples", 1))  # ReID samples taken of a parked car
        verify_samples = int(self.cfg.get("reid.verify_samples", 3))  # samples to confirm a new vehicle
        visit_samples = int(self.cfg.get("reid.samples_per_visit", 4))  # max samples per camera visit
        follow_samples = int(self.cfg.get("reid.follow_samples", 10))  # samples per visit of a followed vehicle
        edge_px = 3
        id_wait = float(self.cfg.get("reid.id_wait_seconds", 1.5))  # longest a shown vehicle waits for its id
        with self._focus_lock:
            focus = set(self.focus_gids)
        motion_frac = float(self.cfg.get("detection.motion_box_fraction", 0.2))
        stop_after = float(self.cfg.get("detection.stopped_after_seconds", 8))
        candidates: list[tuple[float, TrackRecord, np.ndarray, float]] = []

        for pkt, dets in zip(packets, all_dets):
            cam = self.cameras[pkt.camera_id]
            frame = pkt.frame
            h, w = frame.shape[:2]
            self.cam_frame_shape[cam.id] = (h, w)
            crops = [frame[int(d.y1):int(d.y2), int(d.x1):int(d.x2)] for d in dets]
            if self.reid_for_tracking:
                embeds = list(self.extractor.extractFeatures(crops, [cam.id] * len(crops))) if crops else []
            else:
                embeds = [color_histogram(c) for c in crops]
            tracks, removed = self.trackers[cam.id].update(dets, embeds, frame.shape)
            self._link_tracks(cam.id, tracks, embeds, pkt.ts, w, h)
            tracks = self._drop_duplicate_tracks(cam.id, tracks, embeds, pkt.ts)
            for tid in removed:
                self._end_track(cam.id, tid)
            self.cam_tracks[cam.id] = tracks
            self.cam_proc_count[cam.id].append(time.time())
            frame_area = float(w * h)

            for tr in tracks:
                key = (cam.id, tr.track_id)
                rec = self.records.get(key)
                if rec is None:
                    rec = TrackRecord(cam.id, tr.track_id, f"{self.engine_session}-{int(cam.session_start)}",
                                      TrackIdentity(cam.id, tr.track_id), pkt.ts, pkt.offset)
                    rec.chain = f"{tr.track_id}"
                    rec.identity.first_ts = pkt.ts
                    self.records[key] = rec
                if tr.detected and tr.det_index is not None and tr.det_index < len(embeds):
                    rec.last_hist = embeds[tr.det_index]
                x1, y1, x2, y2 = tr.box
                score = camera_score((x2 - x1) * (y2 - y1), frame_area, cam.resolution_weight)
                rec.last_ts, rec.last_offset = pkt.ts, pkt.offset
                if tr.detected:
                    rec.path.append([round(float(pkt.offset), 2), round(x1 / w, 4), round(y1 / h, 4),
                                     round(x2 / w, 4), round(y2 / h, 4)])
                    if len(rec.path) > self.journey_max_points:
                        rec.path = rec.path[::2]  # long visit (parked car): keep every other point
                rec.frames += 1
                rec.best_score = max(rec.best_score, score)
                center = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)
                if rec.first_center is None:
                    rec.first_center = center
                rec.max_shift = max(rec.max_shift, float(np.hypot(center[0] - rec.first_center[0],
                                                                  center[1] - rec.first_center[1])))
                if rec.gid:
                    # A track the tracker lost (prediction only) must not block its id for a new track
                    # of the same car - that produced duplicate ids at low AI frame rates.
                    if tr.detected:
                        self.matcher.active.setdefault(rec.gid, set()).add(key)
                    else:
                        self.matcher.release(rec.gid, cam.id, tr.track_id)
                if tr.detected:
                    # Moving = enough changing pixels inside the box (background model, works from the
                    # first detection) OR the box centre travelled further than detector jitter.
                    size = float(np.hypot((x2 - x1) / w, (y2 - y1) / h))
                    fg = self._motion_fraction(pkt.motion, tr.box, w, h)
                    step = (float(np.hypot(center[0] - rec.prev_center[0], center[1] - rec.prev_center[1]))
                            if rec.prev_center else 0.0)
                    rec.prev_center = center
                    was_moving = rec.moving
                    if fg >= motion_frac or step >= max(move_min, move_frac * size) * 0.5:
                        rec.last_motion_ts = pkt.ts
                    if not rec.moving:
                        rec.moving = fg >= motion_frac or rec.max_shift >= max(move_min, move_frac * size)
                        if rec.moving:
                            rec.last_motion_ts = pkt.ts
                    elif pkt.ts - rec.last_motion_ts > stop_after and not (rec.gid and rec.gid in focus):
                        # (a vehicle the operator follows stays shown when it parks)
                        rec.moving = False  # stopped / parked: hide again (keeps its id)
                        rec.first_center = center
                        rec.max_shift = 0.0
                        logbus.info("tracking", f"{rec.gid or 'Vehicle'} stopped on camera {cam.id}; hidden until it moves.")
                    if rec.moving and not was_moving and moving_only:
                        logbus.info("tracking", f"Moving {tr.label} detected on camera {cam.id} (track {tr.track_id}).")
                        if rec.gid is None and rec.identity.embeddings:
                            # Samples were collected while it stood still (e.g. at the barrier): id it now.
                            self._add_sample(rec, None, pkt.ts, force=True)
                if tr.detected:
                    rec.labels.add(tr.label, tr.confidence)
                    rec.identity.label = rec.labels.value()
                    rec.identity.types = rec.labels.counter  # type votes used by the attribute check
                if (rec.gid is None and tr.detected and (rec.moving or not moving_only)
                        and pkt.ts - rec.first_ts >= id_wait):
                    # Every shown vehicle carries an id: one still without one after id_wait seconds
                    # (crop too small for ReID, or its sample still queued) is identified with the
                    # samples it has, or gets a new id that its first samples can still merge.
                    if rec.identity.embeddings:
                        self._add_sample(rec, None, pkt.ts, force=True)
                    else:
                        self._assign_without_features(rec, pkt.ts)
                waiting = moving_only and not rec.moving  # stationary: hidden, no global id yet
                if waiting and len(rec.identity.embeddings) >= presamples:
                    continue  # enough appearance samples stored for when it starts to move
                if not tr.detected or tr.det_index is None:
                    continue
                crop = crops[tr.det_index]
                if crop.shape[0] < min_crop or crop.shape[1] < min_crop:
                    # too small for ReID, but its colour can still be named (cheap) so the box shows it
                    if (not waiting and crop.shape[0] >= 8 and crop.shape[1] >= 12
                            and sum(rec.identity.colors.values()) < 3 and self.cycle - rec.last_sample_cycle >= 2):
                        self._add_color(rec, self.color_namer.observe(cam.id, crop))
                        rec.last_sample_cycle = self.cycle
                    continue
                # Moving vehicles pass through a view in a few analysed frames: sample them every frame
                # until they have an id. Stationary ones get a few slow, low-priority samples.
                followed = rec.gid is not None and rec.gid in focus and not waiting
                if followed:
                    # The vehicle the operator follows: its gallery is what the next camera is matched
                    # against, so take a sample on every AI check, only of the whole vehicle (a box cut by
                    # the frame edge is a poor sample), up to reid.follow_samples per camera visit.
                    x1, y1, x2, y2 = tr.box
                    cut = x1 <= edge_px or y1 <= edge_px or x2 >= w - edge_px or y2 >= h - edge_px
                    if cut or len(rec.identity.embeddings) >= follow_samples:
                        continue
                    wait = 1
                elif waiting:
                    wait = interval * 2
                elif rec.gid is None:
                    wait = 1  # moving and still without an id: highest priority
                elif rec.identity.created_new and not rec.identity.verified:
                    if len(rec.identity.embeddings) >= verify_samples:
                        continue  # checked enough times for a hand-off match
                    wait = 2
                else:
                    if len(rec.identity.embeddings) >= visit_samples:
                        continue  # enough appearance samples for this visit
                    wait = interval * 6
                if rec.pending_reid or self.cycle - rec.last_sample_cycle < wait:
                    continue
                if self.reid_for_tracking:
                    self._add_color(rec, self.color_namer.observe(cam.id, crop))
                    self._add_sample(rec, embeds[tr.det_index], pkt.ts)
                    rec.last_sample_cycle = self.cycle
                else:
                    # -1 followed vehicle, 0 moving without id (incl. hand-off candidates), 1 extra, 2 parked
                    prio = (-1 if followed else 2 if waiting else 0 if rec.gid is None else 1, -score)
                    candidates.append((prio, rec, crop.copy(), pkt.ts, score))

            if moving_only:  # only moving vehicles are shown, listed and counted
                tracks = [t for t in tracks if (r := self.records.get((cam.id, t.track_id))) is not None and r.moving]
                self.cam_moving[cam.id] = tracks
            visible: set[str] = set()
            for tr in tracks:
                rec = self.records.get((cam.id, tr.track_id))
                if rec and rec.gid:
                    crop = crops[tr.det_index] if tr.detected and tr.det_index is not None else None
                    self._update_vehicle_live(rec, tr, crop, frame_area, cam, pkt.ts)
                    visible.add(rec.gid)
            for gid in self.cam_visible.get(cam.id, set()) - visible:  # vehicle left this camera's view
                v = self.vehicles.get(gid)
                if v is not None and cam.id in v.now and len(v.now) == 1:
                    # out of every camera now: keep it and look for it on the others
                    v.search_since, v.search_from, v.search_logged = time.time(), cam.id, False
                    clock = cam.clock_at(pkt.offset)
                    v.search_clock = format_clock(clock) if clock is not None else None
                if v is not None and v.now.pop(cam.id, None) is not None:
                    v.dirty = True
            self.cam_visible[cam.id] = visible
            self._publish_overlay(cam, frame, tracks, pkt.ts)

        if self.reid_worker is not None and candidates:
            backlog = self.reid_worker.inbox.qsize()
            room = max(0, budget * 2 - backlog)
            candidates.sort(key=lambda c: c[0])
            for prio, rec, crop, ts, _score in candidates[: min(budget, room)]:
                if prio[0] == 2 and backlog >= max(1, budget // 2):
                    break  # stationary pre-samples only when the ReID worker is idle enough
                backlog += 1
                rec.pending_reid = True
                rec.last_sample_cycle = self.cycle
                self.reid_worker.submit(prio[0], ReIDRequest(rec.camera_id, rec.track_id, crop, ts))

    def _link_tracks(self, cam_id: str, tracks: list[TrackState], embeds: list, ts: float, w: int, h: int) -> None:
        """Re-link a vehicle whose tracker track was replaced by a new one.

        At 1-2 AI checks per second a car driving towards the camera moves and grows so much
        between checks that the tracker's box overlap fails and it starts a new track: the car
        then lost its box, its ReID samples and its id every few seconds. A new track that
        appears where a moving vehicle's lost track should be now (its last box moved along
        its velocity), with a similar size and colour, continues that vehicle's record."""
        window = float(self.cfg.get("tracking.link_seconds", 4.0))
        if window <= 0:
            return
        present = {t.track_id for t in tracks if t.detected}
        new = [t for t in tracks if t.detected and (cam_id, t.track_id) not in self.records]
        if not new:
            return
        lost = [r for (c, tid), r in self.records.items()
                if c == cam_id and tid not in present and r.moving and r.last_box is not None
                and 0 < ts - r.last_box_time <= window]
        if not lost:
            return
        max_speed = float(self.cfg.get("tracking.link_max_speed", 0.35))  # frame diagonals per second
        min_sim = float(self.cfg.get("tracking.link_min_similarity", 0.5))
        # IoU match: the new box overlaps where the lost vehicle's box should be now (its last box moved
        # along its velocity) by at least link_iou -> same vehicle, with a looser colour check.
        link_iou = float(self.cfg.get("tracking.link_iou", 0.3))
        iou_min_sim = float(self.cfg.get("tracking.link_iou_min_similarity", 0.3))
        pairs = []
        for t in new:
            x1, y1, x2, y2 = t.box
            nb = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)
            n_area = max(1e-6, (nb[2] - nb[0]) * (nb[3] - nb[1]))
            n_diag = float(np.hypot(nb[2] - nb[0], nb[3] - nb[1]))
            for r in lost:
                dt = ts - r.last_box_time
                pred = r.last_box + (r.velocity * dt if r.velocity is not None else 0)
                o_area = max(1e-6, (r.last_box[2] - r.last_box[0]) * (r.last_box[3] - r.last_box[1]))
                if not 0.15 <= n_area / o_area <= 6.0:
                    continue  # far too different in size
                dist = float(np.hypot((nb[0] + nb[2] - pred[0] - pred[2]) / 2, (nb[1] + nb[3] - pred[1] - pred[3]) / 2))
                allowed = max_speed * dt + n_diag
                ov = _iou(nb, pred)
                by_iou = link_iou > 0 and ov >= link_iou
                if dist > allowed and not by_iou:
                    continue
                sim = 1.0
                if r.last_hist is not None and t.det_index is not None and t.det_index < len(embeds):
                    a, b = np.asarray(r.last_hist, np.float32), np.asarray(embeds[t.det_index], np.float32)
                    sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                if sim < (iou_min_sim if by_iou else min_sim):
                    continue
                # ov is logged only when the IoU match alone linked them (outside the distance limit)
                pairs.append((min(1.0, dist / allowed) - 0.5 * sim - ov, t, r, ov if dist > allowed else 0.0))
        used_t, used_r = set(), set()
        for _cost, t, r, ov in sorted(pairs, key=lambda p: p[0]):
            if t.track_id in used_t or id(r) in used_r:
                continue
            used_t.add(t.track_id)
            used_r.add(id(r))
            if ov >= link_iou > 0:
                logbus.info("tracking", f"{r.gid or 'Vehicle'} on {cam_id}: new track {t.track_id} matched by IoU "
                                         f"{ov:.2f} to track {r.track_id}; same vehicle.")
            old = (cam_id, r.track_id)
            self.records.pop(old, None)
            if r.gid:
                self.matcher.release(r.gid, cam_id, r.track_id)
            self.trackers[cam_id].delete(r.track_id)
            self.track_alias[old] = (cam_id, t.track_id)
            r.track_id = t.track_id
            r.identity.track_id = t.track_id
            self.records[(cam_id, t.track_id)] = r
            v = self.vehicles.get(r.gid) if r.gid else None
            if v is not None and cam_id in v.now:
                v.now[cam_id]["tid"] = t.track_id
        if len(self.track_alias) > 5000:
            for k in list(self.track_alias)[:2500]:
                del self.track_alias[k]

    def _drop_duplicate_tracks(self, cam_id: str, tracks: list[TrackState], embeds: list, ts: float) -> list[TrackState]:
        """One vehicle followed as two tracks (boxed twice by the detector) is one vehicle: see
        tracking.duplicate_iou. The younger track is removed; if it already got a new id, that id
        is merged into the older track's id (a matched, established id is never merged)."""
        thr = float(self.cfg.get("tracking.duplicate_iou", 0.6))
        need = int(self.cfg.get("tracking.duplicate_checks", 2))
        young_s = float(self.cfg.get("tracking.duplicate_young_seconds", 6.0))
        hits_prev = self.dup_hits.get(cam_id, {})
        hits: dict[tuple[str, str], int] = {}
        if thr <= 0 or thr > 1:
            return tracks
        det = [t for t in tracks if t.detected and (cam_id, t.track_id) in self.records]
        drop: set[str] = set()
        for i, a in enumerate(det):
            for b in det[i + 1:]:
                if a.track_id in drop or b.track_id in drop or _iou(a.box, b.box) < thr:
                    continue
                pair = tuple(sorted((a.track_id, b.track_id)))
                hits[pair] = hits_prev.get(pair, 0) + 1
                if hits[pair] < need:
                    continue
                ra, rb = self.records[(cam_id, a.track_id)], self.records[(cam_id, b.track_id)]
                if not (ra.moving or rb.moving):
                    continue  # parked cars side by side in a car park overlap a lot: leave them
                keep, lose = (ra, rb) if ra.first_ts <= rb.first_ts else (rb, ra)
                if keep.gid is None and lose.gid is not None:
                    keep, lose = lose, keep  # keep the track that already carries the id
                if ts - lose.first_ts > young_s:
                    continue  # both long-lived: two vehicles, one passing behind the other
                sim = 0.0
                if keep.last_hist is not None and lose.last_hist is not None:
                    x, y = np.asarray(keep.last_hist, np.float32), np.asarray(lose.last_hist, np.float32)
                    sim = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
                drop.add(lose.track_id)
                lose_gid = lose.gid
                key = (cam_id, lose.track_id)
                self.records.pop(key, None)
                self.ended_pending.pop(key, None)
                if lose_gid:
                    self.matcher.release(lose_gid, cam_id, lose.track_id)
                    v = self.vehicles.get(lose_gid)
                    if v is not None and v.now.get(cam_id, {}).get("tid") == lose.track_id:
                        v.now.pop(cam_id, None)
                        v.dirty = True
                self.trackers[cam_id].delete(lose.track_id)
                self.track_alias[key] = (cam_id, keep.track_id)
                for vec in lose.identity.embeddings:
                    keep.identity.add(vec)
                keep.identity.colors.update(lose.identity.colors)
                if lose_gid and keep.gid and lose_gid != keep.gid and lose.identity.created_new:
                    self._merge(lose_gid, keep.gid, sim)
                logbus.info("tracking", f"Camera {cam_id}: track {lose.track_id} ({lose_gid or 'no id'}) is the same "
                                        f"vehicle as track {keep.track_id} ({keep.gid or 'no id'}) - one car boxed "
                                        "twice; the duplicate was removed.")
        self.dup_hits[cam_id] = hits
        return [t for t in tracks if t.track_id not in drop] if drop else tracks

    def _log_follow_check(self, rec: TrackRecord, ts: float) -> None:
        """While a followed vehicle is being searched for, say how close each new vehicle came."""
        with self._focus_lock:
            focus = [g for g in self.focus_gids if self.handoff.get(g, {}).get("status") in ("searching", "lost")]
        if not focus or not rec.identity.embeddings:
            return
        sims = self.matcher.similarities(self.matcher.query_of(rec.identity))
        for g in focus:
            if g in sims and g != rec.gid:
                thr = self.matcher.threshold_for(g, rec.camera_id, ts, rec.first_ts)
                logbus.info("reid", f"Checked {rec.gid} on {rec.camera_id} against followed {g}: similarity "
                                    f"{sims[g]:.2f} (needs {thr:.2f}, {len(rec.identity.embeddings)} sample(s)).")

    def _apply_reid_results(self) -> None:
        if self.reid_worker is None:
            return
        for req, vec in self.reid_worker.drain():
            key = (req.camera_id, req.track_id)
            for _ in range(10):  # follow re-linked tracks to the record that holds this vehicle now
                if key in self.records or key not in self.track_alias:
                    break
                key = self.track_alias[key]
            rec = self.records.get(key)
            color = self.color_namer.observe(req.camera_id, req.crop)
            if rec is None:
                ended = self.ended_pending.pop(key, None)
                if ended is not None:
                    # The vehicle left the view while its sample was being processed: identify it now.
                    ended.pending_reid = False
                    ended.identity.add(vec)
                    if color:
                        ended.identity.colors[color] += 1
                    self.records[key] = ended
                    self._end_track(*key)
                continue
            rec.pending_reid = False
            self._add_color(rec, color)
            self._add_sample(rec, vec, req.ts)

    def _add_color(self, rec: TrackRecord, color: str | None) -> None:
        """One colour reading of a track: votes for the track (used when it is matched) and, once it
        has an id, for the stored vehicle."""
        if not color:
            return
        rec.identity.colors[color] += 1
        if rec.gid is not None:
            self.matcher.add_attributes(rec.gid, color)
            v = self.vehicles.get(rec.gid)
            if v is not None:
                v.colors.add(color)
                v.dirty = True

    def _add_sample(self, rec: TrackRecord, vec: np.ndarray | None, ts: float, force: bool = False) -> None:
        """Add one ReID embedding to a track; assign its global id once enough samples exist
        (``force`` = assign now with whatever samples the track has)."""
        if vec is not None:
            rec.identity.add(vec)
        min_samples = int(self.cfg.get("reid.min_samples", 2))
        if rec.gid is None:
            if self.moving_only and not rec.moving:
                return  # keep the samples; the id is given once the vehicle moves
            if self.moving_only and rec.moving:
                min_samples = 1  # passing cars: id at once, later samples can still merge it
            if len(rec.identity.embeddings) < min_samples and not force:
                return
            self._assign(rec, ts)
            return
        if vec is None:
            return
        self.matcher.add_embedding(rec.gid, vec, rec.camera_id, ts, owner=(rec.identity.camera_id, rec.identity.track_id))
        if rec.identity.timing_link and not rec.identity.created_new:
            self._check_timing_link(rec, ts)
            return
        if rec.identity.created_new and not rec.identity.verified:
            # A fresh id may really be a vehicle that just left another camera: keep checking
            # with every new sample (hand-off), and stop after a few samples.
            if len(rec.identity.embeddings) >= int(self.cfg.get("reid.verify_samples", 3)):
                rec.identity.verified = True
            self._log_follow_check(rec, ts)
            found = self.matcher.find_merge(rec.identity)
            if not found and rec.identity.provisional:
                # given a new id before any feature: the route may say which vehicle it is
                found = self.matcher.route_candidate(rec.identity, ts)
                if found:
                    rec.identity.provisional = False
                    logbus.info("reid", f"{rec.gid} on {rec.camera_id} is {found[0]} by the route (just on "
                                        f"camera {self.matcher.last_seen[found[0]][1]}); ids merged.")
            if found:
                self._merge(rec.gid, found[0], found[1])

    def _check_timing_link(self, rec: TrackRecord, ts: float) -> None:
        """The track got its id from TIMING (overlap / route) before it had features. Now that it has
        features, compare them with every vehicle timing allows (reid.link_check_samples samples): if
        another of them clearly matches better (by route_margin, at least route_feature_min) than the id
        it got, it is that vehicle - the id is corrected (fixes look-alike / same-moment collisions)."""
        ident = rec.identity
        ident.timing_link -= 1
        old = ident.gid
        owner = (ident.camera_id, ident.track_id)
        q = self.matcher.query_of(ident)
        cur = self.matcher.similarity_without(q, old, owner)
        cands = {g: s for g, s in self.matcher.link_candidates(ident, ts).items() if g != old and s >= 0}
        if not cands:
            return
        best, best_s = max(cands.items(), key=lambda kv: kv[1])
        need = (cur + self.matcher.route_margin) if cur is not None else self.matcher.threshold
        if best_s < max(need, self.matcher.route_feature_min):
            return
        cam = self.cameras[rec.camera_id]
        self.matcher.displace(old, ident.camera_id, ident.track_id, since_ts=rec.first_ts, track=ident)
        self.matcher.link_overlap(ident, best, ts)
        ident.timing_link = 0
        self.matcher.touch(best, rec.camera_id, ts)
        vo = self.vehicles.get(old)
        if vo is not None:
            vo.now.pop(rec.camera_id, None)
            vo.dirty = True
        vn = self.vehicles.get(best)
        if vn is None:
            vn = self.vehicles[best] = VehicleLive(best, rec.first_ts)
        vn.labels.counter.update(rec.labels.counter)
        vn.colors.counter.update(ident.colors)
        vis = self.cam_visible.get(cam.id)
        if vis is not None:
            vis.discard(old)
        logbus.info("reid", f"Feature check on camera {rec.camera_id}{self._at(rec)}: the car given {old} by timing "
                            f"looks like {best} (similarity {best_s:.2f} vs "
                            f"{'%.2f' % cur if cur is not None else 'no features'} for {old}); id corrected to {best}.")

    def _assign(self, rec: TrackRecord, ts: float, exclude: frozenset = frozenset(), depth: int = 0) -> None:
        """Give ``rec`` its global id from its features (highest similarity above the threshold,
        else a new id). If it takes an id that another camera's track holds with a lower
        similarity, that track is re-identified without it."""
        res = self.matcher.matchAcrossCameras(rec.identity, ts, exclude, force_new=depth >= 3)
        if res.via.startswith(("route", "overlap")):
            rec.identity.timing_link = int(self.cfg.get("reid.link_check_samples", 2))  # check by features later
        rejections = list(self.matcher.last_rejections)
        if len(res.candidates) > 1:
            logbus.info("reid", f"Track on {rec.camera_id}: {len(res.candidates)} stored vehicles passed the "
                                "threshold (" + ", ".join(f"{g} {s:.2f}" for g, s in res.candidates[:4])
                                + f"); the highest, {res.gid}, gives the id.")
        if res.displaced is not None:
            self._displace(res.gid, res.displaced, rec, res.similarity, ts, depth)
        self.matcher.touch(res.gid, rec.camera_id, max(ts, rec.last_ts))
        v = self.vehicles.get(res.gid)
        if v is None:
            v = self.vehicles[res.gid] = VehicleLive(res.gid, rec.first_ts)
        v.first_seen = min(v.first_seen, rec.first_ts)
        v.labels.counter.update(rec.labels.counter)
        v.colors.counter.update(rec.identity.colors)
        for gid, sim, reason in rejections:
            logbus.info("reid", f"Track on {rec.camera_id} looked like {gid} (similarity {sim:.2f}) "
                                f"but is not it: {reason}.")
        color, vtype = self.attr_gate.describe(rec.identity.colors, rec.identity.types,
                                               self.attr_gate.track_min_votes)
        desc = " ".join(x for x in (color, vtype or rec.identity.label) if x) or "vehicle"
        if res.is_new:
            self.counters["new_signatures"] += 1
            logbus.info("reid", f"New vehicle {res.gid} ({desc}) detected on camera {rec.camera_id}{self._at(rec)}; "
                                f"features stored in the database.")
            self._log_follow_check(rec, ts)
        else:
            self.counters["reid_matches"] += 1
            if any(c != rec.camera_id for c in v.cameras_seen):
                self.counters["cross_camera_matches"] += 1
            seen = ", ".join(sorted(c for c in v.cameras_seen if c != rec.camera_id)) or rec.camera_id
            if res.via.startswith("route"):
                logbus.info("reid", f"Vehicle {res.gid} ({desc}) re-identified on camera {rec.camera_id}{self._at(rec)} by the route: "
                                    f"it was just on camera {res.via[6:]} on the route "
                                    f"(similarity {res.similarity:.2f}; front and rear views differ).")
            elif res.via.startswith("overlap"):
                logbus.info("reid", f"Vehicle {res.gid} ({desc}) re-identified on camera {rec.camera_id}{self._at(rec)}: the only "
                                    f"vehicle seen at this moment on overlapping camera {res.via[8:]} (similarity "
                                    f"{res.similarity:.2f}; the two views differ too much in size/angle).")
            else:
                logbus.info("reid", f"Vehicle {res.gid} ({desc}) re-identified on camera {rec.camera_id}{self._at(rec)} "
                                    f"(similarity {res.similarity:.2f}; previously seen on {seen}).")
        self.matcher.touch(res.gid, rec.camera_id, ts)

    def _at(self, rec: TrackRecord) -> str:
        """" at 08:01:23" - the shared camera clock when the track was first seen (or "")."""
        cam = self.cameras.get(rec.camera_id)
        clock = cam.clock_at(rec.first_offset) if cam is not None else None
        return f" at {format_clock(clock)}" if clock is not None else ""

    def _assign_without_features(self, rec: TrackRecord, ts: float) -> None:
        """New id for a shown vehicle that has no ReID sample yet (see reid.id_wait_seconds) - or
        the id of the one vehicle seen at this moment on an overlapping camera."""
        og = self.matcher.overlap_candidate(rec.identity, ts)
        how = "the only vehicle seen at this moment on overlapping camera"
        if og is None:
            rc = self.matcher.route_candidate(rec.identity, ts)
            if rc is not None:
                og, how = rc[0], "by the route, it was just on camera"
        if og is not None:
            partner = self.matcher.last_seen[og][1]
            self.matcher.link_overlap(rec.identity, og, ts)
            rec.identity.timing_link = int(self.cfg.get("reid.link_check_samples", 2))  # check by features later
            self.matcher.touch(og, rec.camera_id, ts)
            v = self.vehicles.get(og)
            if v is None:
                v = self.vehicles[og] = VehicleLive(og, rec.first_ts)
            v.labels.counter.update(rec.labels.counter)
            v.colors.counter.update(rec.identity.colors)
            logbus.info("reid", f"Vehicle {og} re-identified on camera {rec.camera_id}{self._at(rec)}: {how} {partner}.")
            return
        res = self.matcher.new_vehicle(rec.identity, ts)
        v = self.vehicles.get(res.gid)
        if v is None:
            v = self.vehicles[res.gid] = VehicleLive(res.gid, rec.first_ts)
        v.labels.counter.update(rec.labels.counter)
        v.colors.counter.update(rec.identity.colors)
        self.counters["new_signatures"] += 1
        self.matcher.touch(res.gid, rec.camera_id, ts)
        logbus.info("reid", f"New vehicle {res.gid} ({rec.identity.label or 'vehicle'}) on camera {rec.camera_id}{self._at(rec)}: "
                            "no ReID features yet (far away or still queued); they are compared with the "
                            "database as soon as a usable crop is extracted.")

    def _displace(self, gid: str, loser_key: tuple[str, str], winner: TrackRecord, sim: float, ts: float,
                  depth: int) -> None:
        """``winner`` matched ``gid`` better than the track that held it on another camera: that
        track gives the id up (its samples, sighting and live state are removed from ``gid``) and
        is re-identified without it - the next best stored vehicle, or a new id."""
        loser = next((r for r in self.records.values()
                      if (r.identity.camera_id, r.identity.track_id) == loser_key and r.gid == gid), None)
        if loser is None:
            self.matcher.displace(gid, *loser_key, since_ts=0.0)
            return
        self.matcher.displace(gid, *loser_key, since_ts=loser.first_ts, track=loser.identity)
        self.matcher.holder[gid] = (winner.identity.camera_id, winner.identity.track_id, sim)
        self.matcher.last_seen[gid] = (ts, winner.camera_id)
        self.db.execute("DELETE FROM sightings WHERE gid=? AND session_id=? AND camera_id=? AND local_track_id=?",
                        (gid, loser.session_id, loser.camera_id, loser.track_id))
        v = self.vehicles.get(gid)
        if v is not None:
            now = v.now.get(loser.camera_id)
            if now and now.get("tid") == loser.track_id:
                v.now.pop(loser.camera_id, None)
            v.dirty = True
        logbus.info("reid", f"{gid}: the track on {winner.camera_id} matches it better ({sim:.2f}) than the track "
                            f"on {loser.camera_id} that had it; that track is re-identified without {gid}.")
        loser.identity.gid = None
        loser.identity.created_new = False
        loser.identity.verified = False
        # The loser may in turn take an id from a weaker holder; after 3 such steps it gets a new id.
        self._assign(loser, ts, exclude=frozenset({gid}), depth=depth + 1)

    def _merge(self, src: str, dst: str, sim: float) -> None:
        self.matcher.merge(src, dst)
        for r in self.records.values():
            if r.identity.gid == src:
                r.identity.gid = dst
                r.identity.created_new = False
        vs = self.vehicles.pop(src, None)
        vd = self.vehicles.get(dst)
        if vd is None and vs is not None:
            vs.gid = dst
            self.vehicles[dst] = vs
        elif vd is not None and vs is not None:
            vd.first_seen = min(vd.first_seen, vs.first_seen)
            vd.cameras_seen.update(vs.cameras_seen)
            vd.labels.counter.update(vs.labels.counter)
            vd.colors.counter.update(vs.colors.counter)
            vd.dirty = True
        with self._focus_lock:
            if src in self.switchers:
                self.switchers.setdefault(dst, self.switchers.pop(src))
            if src in self.focus_gids:
                # The operator follows this vehicle: keep following it under its merged id.
                self.focus_gids = (self.focus_gids - {src}) | {dst}
                self.matcher.followed = set(self.focus_gids)
                if src in self.handoff and dst not in self.handoff:
                    self.handoff[dst] = self.handoff.pop(src)
        self.bridge.send_event({"type": "merge", "from": src, "to": dst})
        logbus.info("reid", f"Vehicle {src} is the same as {dst} (similarity {sim:.2f}); the two records were merged.")

    def _update_vehicle_live(self, rec: TrackRecord, tr: TrackState, crop: np.ndarray | None, frame_area: float,
                             cam: CameraStream, ts: float) -> None:
        v = self.vehicles.get(rec.gid)
        if v is None:
            v = self.vehicles[rec.gid] = VehicleLive(rec.gid, rec.first_ts)
        x1, y1, x2, y2 = tr.box
        score = camera_score((x2 - x1) * (y2 - y1), frame_area, cam.resolution_weight)
        h, w = self.cam_frame_shape[cam.id]
        # If two tracks of one camera carry the same id (briefly, before a merge), keep the bigger.
        prev = v.now.get(cam.id)
        if prev and prev["tid"] != tr.track_id and prev["ts"] >= time.time() - 0.5 and prev["score"] > score:
            return
        if v.search_since and not v.now:
            if cam.id != v.search_from and v.search_logged:
                logbus.info("tracking", f"{rec.gid} found again on camera {cam.id}{self._at(rec)}, "
                                        f"{time.time() - v.search_since:.0f} s after it left {v.search_from}.")
            v.search_since, v.search_from = 0.0, None
        if cam.id not in v.now:  # a new visit of this camera starts
            v.visit_start[cam.id] = time.time()
            clock = cam.clock_at(rec.last_offset)
            v.visit_clock[cam.id] = format_clock(clock) if clock is not None else ""
        v.now[cam.id] = {"score": score, "tid": tr.track_id, "ts": time.time(), "detected": tr.detected,
                         "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]}
        v.moving = rec.max_shift > float(self.cfg.get("pipeline.parked_motion_ratio", 0.03)) or v.moving
        v.dirty = True
        if crop is None:
            return  # Kalman prediction only - nothing new to learn from this frame
        v.last_seen = max(v.last_seen, ts)
        v.cameras_seen[cam.id] = ts
        self.matcher.touch(rec.gid, cam.id, ts)
        v.labels.add(tr.label, tr.confidence)
        now = time.time()
        if score > v.thumb_score * 1.15 and now - v.thumb_saved_at > 2.0 and crop.size:
            pad_crop = crop
            cv2.imwrite(str(self.thumb_dir / f"{rec.gid}.jpg"), pad_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            v.thumb_score, v.thumb_saved_at = score, now
            v.thumb_version += 1

    def _end_track(self, camera_id: str, track_id: str) -> None:
        rec = self.records.get((camera_id, track_id))
        if rec is None:
            return
        if rec.gid is None and rec.pending_reid and (rec.moving or not self.moving_only):
            # Wait for the queued ReID sample instead of losing this vehicle (see _apply_reid_results).
            self.ended_pending[(camera_id, track_id)] = self.records.pop((camera_id, track_id))
            rec.ended_at = time.time()
            for k in [k for k, r in self.ended_pending.items() if time.time() - r.ended_at > 30]:
                del self.ended_pending[k]
            return
        if rec.gid is None and rec.identity.embeddings and (rec.moving or not self.moving_only):
            # Short visit (e.g. a car driving past): identify it from the samples we have.
            self._add_sample(rec, None, rec.last_ts, force=True)
        self.records.pop((camera_id, track_id), None)
        if rec.gid:
            self._write_sighting(rec)
            self.matcher.release(rec.gid, camera_id, track_id)
            v = self.vehicles.get(rec.gid)
            if v is not None and v.now.get(camera_id, {}).get("tid") == track_id:
                v.now.pop(camera_id, None)
                v.dirty = True

    # ------------------------------------------------------------- switching
    def _scores_for(self, gid: str) -> dict[str, float]:
        """FR-CS-01 scores of every camera whose latest processed frame detected the vehicle."""
        v = self.vehicles.get(gid)
        if v is None:
            return {}
        return {c: d["score"] for c, d in v.now.items() if d["detected"]}

    def _update_switching(self) -> None:
        with self._focus_lock:
            focus = set(self.focus_gids)
            for g in list(self.switchers):
                if g not in focus:
                    del self.switchers[g]
            for g in focus:
                if g not in self.switchers:
                    self.switchers[g] = CameraSwitchingEngine(
                        self.cfg.get("switching.hysteresis", 0.15), self.cfg.get("switching.buffer_frames", 5),
                        self.cfg.get("switching.fallback_frames", 3), self.cfg.get("switching.dbn_stay_probability", 0.85),
                        self.cfg.get("switching.dbn_evidence_gain", 8.0))
            switchers = dict(self.switchers)
        if not switchers or not self._new_cycle_for_switching():
            return
        for gid, eng in switchers.items():
            scores = self._scores_for(gid)
            d = eng.update(scores)
            if d.switched:
                ts = time.time()
                self.db.execute("INSERT INTO switch_events(gid,from_cam,to_cam,reason,ts) VALUES(?,?,?,?,?)",
                                (gid, d.previous, d.active, d.reason, ts))
                self.bridge.send_event({"type": "switch", "gid": gid, "from": d.previous, "to": d.active,
                                        "reason": d.reason, "ts": ts})
                logbus.info("switching", f"Vehicle {gid}: {d.reason}.")
            self.bridge.send_event({"type": "focus", "gid": gid, "active": d.active, "scores": d.scores,
                                    "belief": {k: round(v, 3) for k, v in d.belief.items()}, "reason": d.reason})
            self._update_handoff(gid, scores)
        for g in list(self.handoff):
            if g not in switchers:
                del self.handoff[g]

    def _update_handoff(self, gid: str, scores: dict[str, float]) -> None:
        """Follow mode: when the vehicle leaves every camera, search the others for it and report
        which camera it is expected in next; when it reappears elsewhere, announce the hand-off."""
        state = self.handoff.get(gid, {"status": "visible", "sent": 0.0})
        now = time.time()
        v = self.vehicles.get(gid)
        visible_now = bool(v and v.now)
        seen = self.matcher.last_seen.get(gid)
        if visible_now:
            cams = sorted(scores, key=scores.get, reverse=True) or list(v.now)
            if state["status"] in ("searching", "lost") and cams and cams[0] != state.get("from"):
                waited = now - state.get("since", now)
                msg = f"Hand-off: {gid} left {state.get('from')} and was found on {cams[0]} after {waited:.0f} s."
                logbus.info("switching", msg)
                self.bridge.send_event({"type": "handoff", "gid": gid, "status": "found", "from": state.get("from"),
                                        "to": cams[0], "after": round(waited, 1), "ts": now})
            self.handoff[gid] = {"status": "visible", "camera": cams[0] if cams else None, "sent": now}
            return
        window = self.matcher.handoff_window
        if seen is None:
            status, payload = "waiting", {"from": None, "elapsed": None, "expected": []}
        else:
            searching = state["status"] in ("searching", "lost")
            # Keep the camera it originally left from for the whole search (a re-identification
            # elsewhere updates last_seen one cycle before the vehicle shows up as visible).
            origin = state.get("from") if searching and state.get("from") else seen[1]
            since = state.get("since", seen[0]) if searching else seen[0]
            elapsed = max(0.0, now - since)
            status = "searching" if elapsed <= window else "lost"
            expected = self.matcher.expected_next(origin)
            payload = {"from": origin, "elapsed": round(elapsed, 1), "window": window, "expected": expected,
                       "others": [c for c in self.cameras if c != origin and c not in {e["camera"] for e in expected}]}
            if not searching and status == "searching":
                exp = ", ".join(e["camera"] for e in expected) or "all other cameras"
                logbus.info("switching", f"{gid} left camera {origin}; searching {exp} for it.")
                state["since"] = since
            if state["status"] == "searching" and status == "lost":
                logbus.warning("switching", f"{gid} was not found in any camera within {window:.0f} s of leaving {seen[1]}.")
        if status != state["status"] or now - state.get("sent", 0) >= 1.0:
            self.bridge.send_event({"type": "handoff", "gid": gid, "status": status, "ts": now, **payload})
            state["sent"] = now
        state.update({"status": status, "from": payload["from"]})
        self.handoff[gid] = state

    _last_switch_cycle = -1

    def _new_cycle_for_switching(self) -> bool:
        """The switching engine is stepped once per processing cycle (one 'frame')."""
        if self.cycle == self._last_switch_cycle:
            return False
        self._last_switch_cycle = self.cycle
        return True

    # ---------------------------------------------------------------- output
    def _publish_overlay(self, cam: CameraStream, frame: np.ndarray, tracks: list[TrackState], frame_ts: float) -> None:
        """Store the AI result of this camera (normalised boxes + velocity) for the video streamer.

        Only boxes the detector actually found in this frame are shown: Kalman-only predictions
        drift when the AI runs at ~1 frame/s and showed up as ghost / duplicate boxes."""
        h, w = frame.shape[:2]
        items = []
        for tr in tracks:
            if not tr.detected:
                continue
            rec = self.records.get((cam.id, tr.track_id))
            x1, y1, x2, y2 = tr.box
            box = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)
            vel = np.zeros(4, dtype=np.float32)
            if rec is not None:
                dt = frame_ts - rec.last_box_time  # video time between two detections of this vehicle
                if rec.last_box is not None and 0.05 < dt <= 2.5:
                    inst = (box - rec.last_box) / dt
                    rec.velocity = 0.5 * rec.velocity + 0.5 * inst if rec.velocity is not None else inst
                    vel = rec.velocity
                else:
                    rec.velocity = None
                rec.last_box, rec.last_box_time = box, frame_ts
            # the vehicle's features shown on its box: colour and type, as stored with its id
            v = self.vehicles.get(rec.gid) if rec is not None and rec.gid else None
            color = (v.colors.value() if v is not None else None) or (
                rec.identity.colors.most_common(1)[0][0] if rec is not None and rec.identity.colors else None)
            vtype = (v.labels.value() if v is not None else None) or tr.label
            items.append({"g": rec.gid if rec else None, "l": tr.track_id, "k": rec.chain if rec else tr.track_id,
                          "c": vtype, "o": color,
                          "p": round(tr.confidence, 2), "d": tr.detected, "box": box, "vel": vel})
        with self._overlay_lock:
            self.overlays[cam.id] = (frame_ts, items, frame)  # frame: for optical-flow box following
            hist = self.overlay_hist.setdefault(cam.id, deque())
            hist.append((frame_ts, {it["l"]: it for it in items}))
            while hist and frame_ts - hist[0][0] > 15:
                hist.popleft()

    def _vehicle_payload(self, v: VehicleLive, now: float) -> dict:
        current = dict(v.now)
        primary, suppressed = None, []
        if current:
            dets = [OverlapDetection(c, d["score"], 1.0, 1.0) for c, d in current.items()]
            primary, suppressed = CameraSwitchingEngine.handleOverlapZone(dets)
        v.primary = primary
        return {
            "gid": v.gid, "class": v.labels.value(), "color": v.colors.value(),
            "motion": "moving" if v.moving else "parked",
            "first_seen": v.first_seen, "last_seen": v.last_seen, "live": bool(current),
            "primary_camera": primary, "overlap_suppressed": suppressed,
            "cameras_now": {c: {"score": round(d["score"], 4), "box": d["box"]} for c, d in current.items()},
            "cameras_seen": sorted(v.cameras_seen), "thumb_version": v.thumb_version,
            # left every camera and still being looked for on the others
            "searching": bool(v.search_since and not current),
            "search_left": round(max(0.0, self.search_seconds - (now - v.search_since))) if v.search_since else 0,
            "search_from": v.search_from,
            "expected": self._expected_cameras(v.search_from) if v.search_since and v.search_from else [],
        }

    def _merge_covisible(self, now: float) -> None:
        """Overlapping cameras by timestamp (reid.overlap_merge): cameras that watch the same place
        see one car AT THE SAME TIME. Two different ids in view on two overlapping cameras that came
        into view within overlap_merge_arrival_seconds of each other and have stayed in view together
        for overlap_merge_min_seconds are one vehicle - if each is the other's only such partner and
        colour/type do not conflict. The younger id is merged into the older one."""
        if not self.cfg.get("reid.overlap_merge", True):
            return
        arrival = float(self.cfg.get("reid.overlap_merge_arrival_seconds", 6.0))
        min_together = float(self.cfg.get("reid.overlap_merge_min_seconds", 4.0))
        in_view = defaultdict(list)  # camera -> [(gid, visit start)]
        for v in self.vehicles.values():
            for c, d in v.now.items():
                if now - d["ts"] <= 2.0 and c in v.visit_start:
                    in_view[c].append((v.gid, v.visit_start[c]))
        pairs: dict[str, list[tuple[str, float, str, str]]] = defaultdict(list)
        for p in self.matcher.overlaps:
            x, y = tuple(p)
            for ga, sa in in_view.get(x, []):
                for gb, sb in in_view.get(y, []):
                    if ga == gb:
                        continue
                    va, vb = self.vehicles.get(ga), self.vehicles.get(gb)
                    if va is None or vb is None or y in va.now or x in vb.now:
                        continue  # one of them is on both cameras already
                    if abs(sa - sb) > arrival or now - max(sa, sb) < min_together:
                        continue  # did not come into view together / not long enough together yet
                    pairs[ga].append((gb, abs(sa - sb), x, y))
                    pairs[gb].append((ga, abs(sa - sb), y, x))
        # An id may have one partner per OTHER camera (C41 + C51 + C61 can all see one car); two
        # candidates on the same camera = ambiguous. Linked ids form a group, merged into its oldest.
        parent: dict[str, str] = {}

        def root(g):
            while parent.get(g, g) != g:
                g = parent[g]
            return g

        for ga, lst in pairs.items():
            per_cam = defaultdict(int)
            for _gb, _dt, _ca, cb in lst:
                per_cam[cb] += 1
            for gb, dt, ca, cb in lst:
                back = sum(1 for x in pairs.get(gb, []) if x[3] == ca)
                if per_cam[cb] != 1 or back != 1 or self.matcher.gids_conflict(ga, gb):
                    continue
                parent[root(ga)] = root(gb)
        groups = defaultdict(list)
        for g in list(parent):
            groups[root(g)].append(g)
        for g, members in list(groups.items()):
            if g not in members:
                members.append(g)
            members = [m for m in members if m in self.vehicles]
            cams = [c for m in members for c in self.vehicles[m].now]
            if len(members) < 2 or len(cams) != len(set(cams)):
                continue  # two ids of the group on one camera: two cars, not one
            members.sort(key=lambda m: self.vehicles[m].first_seen)
            keep = members[0]
            kv = self.vehicles[keep]
            kc = next(iter(kv.now))
            for lose in members[1:]:
                lv = self.vehicles.get(lose)
                if lv is None:
                    continue
                lc = next(iter(lv.now))
                dt = abs(kv.visit_start.get(kc, now) - lv.visit_start.get(lc, now))
                logbus.info("reid", f"{lose} on camera {lc} is {keep} on camera {kc}: the two cameras watch the same "
                                    f"place; both came into view at {kv.visit_clock.get(kc) or '?'} / "
                                    f"{lv.visit_clock.get(lc) or '?'} ({dt:.1f} s apart) and stayed in view together "
                                    f"for {now - max(kv.visit_start.get(kc, now), lv.visit_start.get(lc, now)):.0f} s; "
                                    "ids merged.")
                vis, clk = dict(lv.visit_start), dict(lv.visit_clock)
                sim = (self.matcher.similarities(self.matcher.gallery_centre(lose)).get(keep, 0.0)
                       if self.matcher.gallery.get(lose) else 0.0)
                self._merge(lose, keep, max(0.0, sim))
                kv = self.vehicles.get(keep) or kv
                for c, t in vis.items():
                    kv.visit_start.setdefault(c, t)
                    kv.visit_clock.setdefault(c, clk.get(c, ""))

    def _expected_cameras(self, cam_id: str | None) -> list[str]:
        """Where a vehicle that left ``cam_id`` should appear: the next route camera(s) and the
        cameras that watch the same place (reid.route, reid.overlapping_cameras)."""
        out = []
        route = self.matcher.route
        if cam_id in route:
            i = route.index(cam_id)
            out += route[i + 1:i + 2 + self.matcher.route_skip]
            if self.matcher.route_bidirectional and i > 0:
                out.append(route[i - 1])  # or back the way it came (driving out)
        for p in self.matcher.overlaps:
            if cam_id in p:
                out += [c for c in p if c != cam_id]
        return list(dict.fromkeys(c for c in out if c != cam_id))

    def _broadcast_vehicles(self) -> None:
        now = time.time()
        stale = float(self.cfg.get("pipeline.vehicle_live_timeout", 15.0))
        for v in self.vehicles.values():
            for c in [c for c, d in v.now.items() if now - d["ts"] > stale]:  # camera stalled / offline
                del v.now[c]
        for v in self.vehicles.values():
            if v.search_since and not v.now and not v.search_logged and now - v.search_since >= 3.0:
                v.search_logged = True  # gone for 3 s, not just one missed detection
                nxt = self._expected_cameras(v.search_from)
                logbus.info("tracking", f"{v.gid} left camera {v.search_from}"
                                        f"{' at ' + v.search_clock if v.search_clock else ''}; kept for "
                                        f"{self.search_seconds / 60:.0f} min while the other cameras are checked"
                                        f"{' (expected next on ' + ', '.join(nxt) + ')' if nxt else ''}.")
            if v.search_since and not v.now and now - v.search_since > self.search_seconds:
                logbus.info("tracking", f"{v.gid} was not seen on another camera within {self.search_seconds / 60:.0f} "
                                        f"min of leaving {v.search_from}; search ended (its features stay stored).")
                v.search_since, v.search_from = 0.0, None
                v.dirty = True
        self._merge_covisible(now)
        recent = [v for v in self.vehicles.values() if v.now or v.search_since or now - v.last_seen <= 120]
        items = [self._vehicle_payload(v, now) for v in sorted(recent, key=lambda v: v.gid)]
        self.bridge.send_event({"type": "vehicles", "ts": now, "items": items})

    def _apply_start_clocks(self) -> None:
        """Place each recording on one timeline from its start_clock (cameras.overrides).

        The earliest start plays at once; a camera whose recording begins later waits
        for the difference, so every camera shows the same real time.
        """
        def seconds(text: str) -> float | None:
            try:
                parts = [float(p) for p in str(text).strip().split(":")]
            except ValueError:
                return None
            while len(parts) < 3:
                parts.append(0.0)
            return parts[0] * 3600 + parts[1] * 60 + parts[2]

        clocks = {cid: seconds(cam.spec.extra["start_clock"]) for cid, cam in self.cameras.items()
                  if cam.spec.is_file and cam.spec.extra.get("start_clock")}
        clocks = {cid: s for cid, s in clocks.items() if s is not None}
        if not clocks or self.playback_clock is None:
            return
        origin = min(clocks.values())
        for cid, s in clocks.items():
            cam = self.cameras[cid]
            cam.start_clock = str(cam.spec.extra["start_clock"])
            cam.start_delay = s - origin
            if cam.start_delay > 0:
                logbus.info("camera", f"Camera {cid} recording starts at {cam.start_clock}: "
                                      f"it begins {cam.start_delay:.0f} s after the others.")

    def _broadcast_stats(self) -> None:
        now = time.time()
        cams = []
        for cid, cam in self.cameras.items():
            stamps = [t for t in self.cam_proc_count[cid] if now - t <= 5]
            cams.append({"id": cid, "status": cam.status, "decode_fps": round(cam.decode_fps, 1),
                         "starts_in": round(cam.starts_in()), "start_clock": cam.start_clock,
                         "process_fps": round(len(stamps) / 5, 1),
                         "tracks": len(self.cam_moving.get(cid, [])) if self.moving_only else len(self.cam_tracks.get(cid, [])),
                         "resolution_weight": round(cam.resolution_weight, 3)})
        if now - getattr(self, "_perf_logged", 0) >= float(self.cfg.get("pipeline.perf_log_interval", 60)):
            self._perf_logged = now
            logbus.info("engine", f"Speed: YOLOv8 {self.det_ms:.0f} ms/frame, ReID "
                                  f"{self.reid_worker.ms_per_crop if self.reid_worker else 0:.0f} ms/crop "
                                  f"(queue {self.reid_worker.inbox.qsize() if self.reid_worker else 0}), AI checks/s: "
                                  + ", ".join(f"{c['id']} {c['process_fps']}" for c in cams))
        self.bridge.send_event({"type": "stats", "ts": now, "cameras": cams,
                                "engine": {"det_ms": round(self.det_ms, 1),
                                           "reid_ms": round(self.reid_worker.ms_per_crop, 1) if self.reid_worker else None,
                                           "reid_queue": self.reid_worker.inbox.qsize() if self.reid_worker else 0,
                                           "vehicles_known": len(self.vehicles) if self.moving_only else len(self.matcher.gallery),
                                           "live_tracks": len(self.records),
                                           "in_view": sum(1 for v in self.vehicles.values() if v.now),
                                           **self.counters}})

    # ------------------------------------------------------------------- db
    def _write_sighting(self, rec: TrackRecord) -> None:
        cam = self.cameras[rec.camera_id]
        self.db.execute(
            """INSERT INTO sightings(gid,camera_id,session_id,local_track_id,source,start_ts,end_ts,start_offset,end_offset,
                                     best_score,frames,width,height,media,boxes,start_clock,end_clock)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(session_id,camera_id,local_track_id,gid) DO UPDATE SET
                 end_ts=excluded.end_ts, end_offset=excluded.end_offset, end_clock=excluded.end_clock,
                 best_score=MAX(best_score, excluded.best_score), frames=excluded.frames,
                 media=excluded.media, boxes=excluded.boxes""",
            (rec.gid, rec.camera_id, rec.session_id, rec.track_id, cam.spec.source, rec.first_ts, rec.last_ts,
             rec.first_offset, rec.last_offset, rec.best_score, rec.frames, cam.width, cam.height,
             # the file actually analysed (offsets are on its timeline) and the vehicle's boxes in it,
             # so the dashboard can replay the vehicle's journey camera by camera
             cam.read_source if cam.spec.is_file else None, json.dumps(rec.path, separators=(",", ":")),
             cam.clock_at(rec.first_offset), cam.clock_at(rec.last_offset)),
        )

    def _flush(self) -> None:
        self.matcher.flush()
        self.color_namer.save()
        for rec in list(self.records.values()):
            if rec.gid:
                self._write_sighting(rec)
        rows = []
        for v in self.vehicles.values():
            if not v.dirty:
                continue
            v.dirty = False
            last_cam = max(v.cameras_seen, key=v.cameras_seen.get) if v.cameras_seen else None
            thumb = f"thumbs/{v.gid}.jpg" if v.thumb_version else None
            rows.append((v.labels.value(), v.colors.value(), v.last_seen or None, last_cam,
                         "moving" if v.moving else "parked", thumb, v.thumb_score, v.gid))
        self.db.executemany(
            """UPDATE vehicles SET vclass=COALESCE(?,vclass), color=COALESCE(?,color),
                 last_seen=MAX(COALESCE(last_seen,0), COALESCE(?,0)), last_camera=COALESCE(?,last_camera), motion=?,
                 thumb_path=COALESCE(?,thumb_path), thumb_score=MAX(COALESCE(thumb_score,0),?) WHERE gid=?""",
            rows,
        )


class FrameStreamer(threading.Thread):
    """Sends camera video to the hub at the camera's own frame rate, independent of the AI.

    The camera shown in an operator's main view is sent at full rate and size;
    other cameras go out as small, low-rate tiles. Boxes from the last AI result
    are moved forward along each track's velocity so they keep up with the video.
    """

    def __init__(self, pipe: "Pipeline"):
        super().__init__(name="streamer", daemon=True)
        self.p = pipe
        cfg = pipe.cfg
        self.main_fps = float(cfg.get("pipeline.stream_fps_main", 30))
        self.tile_fps = float(cfg.get("pipeline.stream_fps_tile", 4))
        self.grid_fps = float(cfg.get("pipeline.stream_fps_grid", 12))
        self.main_w = int(cfg.get("pipeline.stream_width", 960))
        self.tile_w = int(cfg.get("pipeline.tile_width", 480))
        self.quality = int(cfg.get("pipeline.jpeg_quality", 70))
        self.max_extrapolate = float(cfg.get("pipeline.overlay_extrapolate_seconds", 0.6))
        self.damping = float(cfg.get("pipeline.overlay_extrapolate_damping", 0.6))
        self.stale_after = float(cfg.get("pipeline.overlay_stale_seconds", 2.5))
        self.use_flow = bool(cfg.get("pipeline.overlay_optical_flow", True))
        self.flow_width = int(cfg.get("pipeline.overlay_flow_width", 480))
        self.flows: dict[str, FlowBoxes] = {}
        self.sent_seq: dict[str, int] = {}
        self.sent_at: dict[str, float] = {}
        # Display delay: frames are held this long so every shown frame lies BETWEEN two AI results
        # and its boxes can be interpolated exactly. 0 = no delay (boxes follow by optical flow).
        self.delay = float(cfg.get("pipeline.display_delay_seconds", 2.5))
        self.hold = float(cfg.get("pipeline.overlay_hold_seconds", 0.4))
        # A box is kept across missed detections up to this gap (seconds of video) ...
        self.max_gap = float(cfg.get("pipeline.overlay_max_gap_seconds", 4.0))
        # ... and extended this long before the first / after the last detection along the car's motion.
        self.edge = max(self.hold, float(cfg.get("pipeline.overlay_edge_seconds", 0.8)))
        self.buffers: dict[str, deque] = {}
        self.latest_ts: dict[str, float] = {}

    def run(self) -> None:
        while not self.p.stop_event.is_set():
            if not self.p.bridge.connected:
                time.sleep(0.2)
                continue
            views = self.p.view_cams
            now = time.time()
            busy = False
            for cid, cam in self.p.cameras.items():
                seq, pkt = cam.peek()
                if pkt is None or seq == self.sent_seq.get(cid):
                    continue
                main = cid in views
                if main:
                    rate = min(self.main_fps, cam.fps or self.main_fps)
                elif cid in self.p.grid_cams:  # grid view: every camera moving, smaller frames
                    rate = min(self.grid_fps, cam.fps or self.grid_fps)
                else:
                    rate = self.tile_fps
                if now - self.sent_at.get(cid, 0) < 1.0 / max(rate, 0.1) * 0.95:
                    continue
                self.sent_seq[cid], self.sent_at[cid] = seq, now
                if self.delay > 0:
                    self._buffer(cam, pkt, self.main_w if main else self.tile_w)
                else:
                    self._send(cam, pkt, self.main_w if main else self.tile_w, now)
                busy = True
            if self.delay > 0:
                busy = self._emit_due() or busy
            if not busy:
                time.sleep(0.004)

    # ------------------------------------------------------- delayed display
    def _buffer(self, cam: CameraStream, pkt: FramePacket, width: int) -> None:
        frame = pkt.frame
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        frame = stamp_clock(frame, cam, pkt.offset)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        buf = self.buffers.setdefault(cam.id, deque())
        if buf and pkt.ts < buf[-1][0]:
            buf.clear()  # the video restarted (looped file)
        buf.append((pkt.ts, pkt.offset, frame.shape[1], frame.shape[0], jpg.tobytes()))
        self.latest_ts[cam.id] = pkt.ts
        while len(buf) > 400:
            buf.popleft()

    def _emit_due(self) -> bool:
        sent = False
        for cid, buf in self.buffers.items():
            cutoff = self.latest_ts.get(cid, 0) - self.delay
            while buf and buf[0][0] <= cutoff:
                ts, off, w, h, jpg = buf.popleft()
                if buf and buf[0][0] <= cutoff:
                    continue  # behind schedule: skip to the newest due frame
                clock = self.p.cameras[cid].clock_at(off)
                self.p.bridge.send_frame(cid, {"cam": cid, "ts": ts, "off": round(off, 2), "w": w, "h": h,
                                               "clock": format_clock(clock) if clock is not None else None,
                                               "delay": self.delay, "tracks": self._boxes_at(cid, ts)}, jpg)
                sent = True
        return sent

    def _edge_box(self, near: tuple[float, dict], far: tuple[float, dict] | None, t: float):
        """Box at time ``t`` just outside a vehicle's detections: moved along the velocity of its two
        nearest detections (held still if only one exists, but then only for overlay_hold_seconds)."""
        tn, it = near
        if far is None or abs(far[0] - tn) < 0.05:
            return (it["box"], it) if abs(t - tn) <= self.hold else (None, it)
        vel = (it["box"] - far[1]["box"]) / (tn - far[0])
        return it["box"] + vel * (t - tn), it

    def _boxes_at(self, cid: str, t: float) -> list[dict]:
        """Boxes for video time ``t``: interpolated between the AI results just before and after it."""
        with self.p._overlay_lock:
            hist = list(self.p.overlay_hist.get(cid, ()))
        # Observations per vehicle record (its chain key stays the same when the tracker's track for
        # the car is replaced and re-linked), so the car keeps one continuous box.
        obs: dict[str, list[tuple[float, dict]]] = {}
        for ts, items in hist:
            if abs(ts - t) > self.max_gap + self.edge:
                continue
            for tid, it in items.items():
                obs.setdefault(str(it.get("k", tid)), []).append((ts, it))
        cands = []
        for key, seq in obs.items():
            seq.sort(key=lambda o: o[0])
            # the vehicle's current id: from its record as it is now (ids are assigned after ReID)
            gid = None
            for _ts, it in reversed(seq):
                rec = self.p.records.get((cid, it["l"]))
                gid = (rec.gid if rec else None) or it["g"]
                if gid:
                    break
            # ... and its current colour/type (they firm up as more crops are named)
            v = self.p.vehicles.get(gid) if gid else None
            vcol = v.colors.value() if v is not None else None
            vtype = v.labels.value() if v is not None else None
            seq = [(ts_, {**it_, "g": gid, "o": vcol or it_.get("o"), "c": vtype or it_["c"]}) for ts_, it_ in seq]
            before = [o for o in seq if o[0] <= t]
            after = [o for o in seq if o[0] > t]
            box = None
            if before and after and after[0][0] - before[-1][0] <= self.max_gap:
                # between two detections (missed detections in between are bridged): interpolate
                (t0, a), (t1, b) = before[-1], after[0]
                k = (t - t0) / max(1e-6, t1 - t0)
                box, it = a["box"] * (1 - k) + b["box"] * k, b
            elif before and t - before[-1][0] <= self.edge:
                # after the last detection (car leaving / next result not in yet): continue its motion
                box, it = self._edge_box(before[-1], before[-2] if len(before) > 1 else None, t)
            elif after and after[0][0] - t <= self.edge:
                # before the first detection (car just entered): run its motion backwards
                box, it = self._edge_box(after[0], after[1] if len(after) > 1 else None, t)
            if box is None:
                continue
            b = np.clip(box, 0.0, 1.0)
            if b[2] - b[0] < 0.003 or b[3] - b[1] < 0.003:
                continue
            cands.append({"g": it["g"], "l": it["l"], "c": it["c"], "o": it.get("o"), "p": it["p"], "d": True, "_b": b})
        # One box per vehicle: drop boxes that mostly overlap a better one (has an id, higher confidence).
        cands.sort(key=lambda c: (c["g"] is not None, c["p"]), reverse=True)
        tracks = []
        for c in cands:
            if any(_overlap(c["_b"], k["_b"]) > 0.5 for k in tracks):
                continue
            tracks.append(c)
        for c in tracks:
            c["b"] = [round(float(v), 4) for v in c.pop("_b")]
        return tracks

    def _send(self, cam: CameraStream, pkt: FramePacket, width: int, now: float) -> None:
        frame = pkt.frame
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        frame = stamp_clock(frame, cam, pkt.offset)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self.p._overlay_lock:
            at, items, analysed = self.p.overlays.get(cam.id, (pkt.ts, [], None))
        age = pkt.ts - at  # video seconds between the analysed frame and the frame shown
        flow = self.flows.setdefault(cam.id, FlowBoxes(self.flow_width))
        if age > self.stale_after:
            items = []  # no fresh AI result for this camera: better no box than a wrong one
        elif self.use_flow and analysed is not None:
            if flow.src_ts != at:
                flow.reset(analysed, items, at)  # new AI result: start from its exact boxes
            items = flow.step(pkt.frame)  # follow the pixels to the frame being shown
        tracks = []
        for it in items:
            if self.use_flow and analysed is not None:
                b = it["box"]
            else:
                dt = min(max(0.0, age), self.max_extrapolate) * self.damping
                b = np.clip(it["box"] + it["vel"] * dt, 0.0, 1.0)
            if b[2] - b[0] < 0.003 or b[3] - b[1] < 0.003:
                continue
            tracks.append({"g": it["g"], "l": it["l"], "c": it["c"], "o": it.get("o"), "p": it["p"], "d": it["d"],
                           "b": [round(float(v), 4) for v in b]})
        self.p.bridge.send_frame(cam.id, {"cam": cam.id, "ts": pkt.ts, "off": round(pkt.offset, 2),
                                          "w": frame.shape[1], "h": frame.shape[0], "tracks": tracks},
                                 jpg.tobytes())


def stamp_clock(frame: np.ndarray, cam: CameraStream, offset: float) -> np.ndarray:
    """Cover the camera's own burned-in timestamp (cameras.overrides.<id>.timestamp_mask) and write
    the shared clock (cameras.clock_start + video time) in its place. Returns a new image."""
    clock = cam.clock_at(offset)
    mask = cam.timestamp_mask
    if clock is None and not mask:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    if mask:
        x1, y1, x2, y2 = (int(round(v * s)) for v, s in zip(mask, (w, h, w, h)))
    else:
        x1, y1, x2, y2 = int(0.01 * w), int(0.02 * h), int(0.2 * w), int(0.08 * h)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return out
    if clock is None:
        out[y1:y2, x1:x2] = (18, 18, 18)
        return out
    text = format_clock(clock)
    font = cv2.FONT_HERSHEY_SIMPLEX
    box_h = y2 - y1
    scale = max(0.3, box_h * 0.7 / 22.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if not mask:
        x2 = min(w, x1 + tw + int(box_h * 0.6))
    out[y1:y2, x1:x2] = (18, 18, 18)
    cv2.putText(out, text, (x1 + int(box_h * 0.3), y1 + (box_h + th) // 2), font, scale, (255, 255, 255), thick,
                cv2.LINE_AA)
    return out


def _iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller box (catches a small box inside a big one, too)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return float(ix * iy / small) if small > 0 else 0.0
