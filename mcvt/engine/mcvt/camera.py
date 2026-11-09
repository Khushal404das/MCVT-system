"""Camera capture: one background thread per camera (RTSP stream or video file).

Each reader keeps only the most recent frame so the detector always works on
"live" data. Files are paced at their native FPS when ``realtime`` is on, so a
recorded video behaves like an RTSP camera. A camera that fails is retried
without affecting the others (NFR-R-02).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import logbus
from .config import CameraSpec


@dataclass
class FramePacket:
    camera_id: str
    frame: np.ndarray
    frame_index: int
    offset: float  # seconds from the start of the source (used for clips)
    ts: float  # wall-clock timestamp of the frame
    motion: np.ndarray | None = None  # small foreground mask (255 = moving pixel) for this frame


class MotionMask:
    """Per-camera background model on every decoded frame (cheap, at small size).

    Parked cars become background within ~history frames; anything moving shows up
    as foreground. Used to decide whether a detected vehicle is moving from its very
    first detection, independent of how often the AI runs.
    """

    def __init__(self, width: int = 320, history_s: float = 20.0, fps: float = 15.0):
        self.width = width
        self.sub = cv2.createBackgroundSubtractorMOG2(history=max(50, int(history_s * fps)), varThreshold=32,
                                                      detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, max(1, int(h * self.width / w))), interpolation=cv2.INTER_AREA)
        fg = self.sub.apply(small)
        fg = np.where(fg == 255, 255, 0).astype(np.uint8)  # drop shadows (127)
        return cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)


class PlaybackClock:
    """Shared start time for recorded videos so every camera plays from the beginning together.

    When the first video ends, the clock is reset and *all* video cameras jump back to
    their first frame, keeping the recordings in step with each other.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.epoch = time.time()
        self.generation = 0

    def restart(self, seen_generation: int | None = None) -> None:
        with self._lock:
            if seen_generation is not None and seen_generation != self.generation:
                return  # another camera already restarted this cycle
            self.generation += 1
            self.epoch = time.time()


class CameraStream:
    """Background reader for one camera."""

    def __init__(self, spec: CameraSpec, realtime: bool = True, loop: bool = True, reconnect_s: float = 5.0,
                 motion_width: int = 0, motion_history_s: float = 20.0, clock: PlaybackClock | None = None,
                 motion_fps: float = 0.0):
        self.clock = clock
        self.motion_fps = float(motion_fps)  # 0 = background model on every frame
        self._mask: np.ndarray | None = None
        self._mask_ts = -1e9
        self.motion_width = motion_width
        self.motion_history_s = motion_history_s
        self.motion: MotionMask | None = None
        self.spec = spec
        self.id = spec.id
        self.realtime = realtime
        self.loop = loop
        self.reconnect_s = reconnect_s
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self.status = "starting"
        self.resolution_weight = 1.0
        self.session_start = time.time()
        self._latest: FramePacket | None = None
        self._latest_seq = 0
        self._consumed_seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cam-{self.id}", daemon=True)
        self.decode_fps = 0.0
        self._restart_reason = "eof"
        self._generation: int | None = None
        self.start_delay = 0.0  # seconds after the shared timeline start (set by the pipeline)
        self.start_clock = ""   # recording start time shown to the operator, e.g. "08:07:28"
        # offset (seconds on the decoded file's timeline) of the first played frame; the shown
        # clock is cameras.clock_start there and runs with the video (restarts when it loops)
        self.offset0 = 0.0
        self.clock_start = self.spec.extra.get("clock_start")  # seconds of the day or None
        self.timestamp_mask = self.spec.extra.get("timestamp_mask")  # [x1, y1, x2, y2] 0..1 or None

    def clock_at(self, offset: float) -> float | None:
        """Clock time (seconds of the day) of the frame at ``offset``; None for live cameras / no clock."""
        if self.clock_start is None or not self.spec.is_file:
            return None
        return float(self.clock_start) + max(0.0, offset - self.offset0)

    def starts_in(self) -> float:
        """Seconds until a waiting camera's recording begins (0 when playing)."""
        if self.status != "waiting" or self.clock is None:
            return 0.0
        return max(0.0, self.clock.epoch + self.start_delay - time.time())

    # ---------------------------------------------------------------- control
    @property
    def read_source(self) -> str:
        """Stream actually decoded: the analysis sub-stream if configured, else the main one."""
        return self.spec.analysis_source or self.spec.source

    @staticmethod
    def _props(source: str) -> tuple[bool, int, int, float, int]:
        cap = cv2.VideoCapture(source)
        ok = cap.isOpened()
        w = h = n = 0
        fps = 25.0
        if ok:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = cap.get(cv2.CAP_PROP_FPS)
            fps = float(f) if f and f > 0 and not math.isnan(f) else 25.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if w <= 0 or h <= 0:
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
        cap.release()
        return ok, w, h, fps, n

    def probe(self) -> bool:
        """Learn the camera's real resolution (main stream) and the decoded stream's FPS."""
        ok, self.width, self.height, self.fps, self.frame_count = self._props(self.spec.source)
        if self.spec.analysis_source:
            a_ok, _, _, a_fps, a_n = self._props(self.spec.analysis_source)
            if a_ok:
                self.fps, self.frame_count = a_fps, a_n
                ok = True
            else:
                self.spec.analysis_source = None
        self.status = "ready" if ok else "offline"
        return ok

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def peek(self) -> tuple[int, FramePacket | None]:
        """Newest frame + its sequence number, without marking it consumed (for live streaming)."""
        with self._lock:
            return self._latest_seq, self._latest

    def latest(self) -> FramePacket | None:
        """Return the newest frame if it was not consumed yet, else None."""
        with self._lock:
            if self._latest is None or self._latest_seq == self._consumed_seq:
                return None
            self._consumed_seq = self._latest_seq
            return self._latest

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.read_source)
            if not cap.isOpened():
                self._set_status("offline", f"Camera {self.id} could not be opened. Retrying in {self.reconnect_s:.0f} s.")
                self._stop.wait(self.reconnect_s)
                continue
            via = " via analysis stream" if self.spec.analysis_source else ""
            self._set_status("online", f"Camera {self.id} is online ({self.width}x{self.height}, analysed at {self.fps:.1f} FPS{via}).")
            self._read_loop(cap)
            cap.release()
            if self._stop.is_set():
                break
            if self.spec.is_file and not self.loop:
                self._set_status("finished", f"Camera {self.id} reached the end of its video.")
                break
            if self.spec.is_file:
                if self._restart_reason == "eof":
                    if self.clock is not None:
                        logbus.info("camera", f"Camera {self.id} video finished - all cameras restart from the beginning.")
                        self.clock.restart(self._generation)
                    else:
                        logbus.info("camera", f"Camera {self.id} video finished, restarting from the beginning.")
                self.session_start = time.time()
            else:
                self._set_status("offline", f"Camera {self.id} stream was lost. Reconnecting in {self.reconnect_s:.0f} s.")
                self._stop.wait(self.reconnect_s)

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        index = 0
        start = time.time()
        self._restart_reason = "eof"
        self._generation = None
        if self.spec.is_file:
            if self.clock is not None:
                self._generation = self.clock.generation
                # frame 0 plays when the shared timeline reaches this camera's recording start
                start = self.clock.epoch + self.start_delay
                if time.time() < start:
                    self._set_status("waiting", f"Camera {self.id} starts in {start - time.time():.0f} s "
                                                f"(its recording begins at {self.start_clock or 'a later time'}).")
                    with self._lock:
                        self._latest = None
                    while time.time() < start:
                        if self._stop.is_set():
                            return
                        if self.clock.generation != self._generation:
                            self._restart_reason = "sync"
                            return
                        time.sleep(min(0.5, max(0.0, start - time.time())))
                    self._set_status("online", f"Camera {self.id} recording started ({self.start_clock}).")
            self.session_start = start
            self.motion = None  # new playback: relearn the background
        # Per-camera alignment (config overrides): start_seconds skips the start of a recording
        # so all cameras show the same clock time; clip_offset_seconds maps our position onto
        # FFmpeg's timeline of the same file when the two decoders disagree on where it starts.
        # start_seconds / clip_offset_seconds apply to the original file; the FFmpeg-made
        # analysis copy is already on FFmpeg's timeline and uses analysis_start_seconds.
        original = self.spec.is_file and not self.spec.analysis_source
        if original:
            skip = float(self.spec.extra.get("start_seconds", 0.0))
            clip_shift = float(self.spec.extra.get("clip_offset_seconds", 0.0))
        else:
            skip = float(self.spec.extra.get("analysis_start_seconds", 0.0)) if self.spec.is_file else 0.0
            clip_shift = 0.0
        self.offset0 = skip + clip_shift
        if skip > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, skip * 1000.0)
        frame_period = 1.0 / self.fps if self.fps else 0.04
        failures = 0
        t_stat, n_stat = time.time(), 0
        while not self._stop.is_set():
            if self._generation is not None and self.clock.generation != self._generation:
                self._restart_reason = "sync"  # another camera's video ended: restart together
                return
            if self.spec.is_file and self.realtime:
                # Skip (grab without decode-to-BGR) frames we are behind on, sleep if ahead.
                due = int((time.time() - start) / frame_period)
                while index < due - 1:
                    if not cap.grab():
                        return
                    index += 1
                ahead = start + index * frame_period - time.time()
                if ahead > 0:
                    time.sleep(ahead)
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if self.spec.is_file or failures > 25:
                    return
                time.sleep(0.05)
                continue
            failures = 0
            offset = (skip + clip_shift + index * frame_period) if self.spec.is_file else time.time() - start
            ts = (self.session_start + index * frame_period) if self.spec.is_file else time.time()
            mask = None
            if self.motion_width:
                # The background model runs at motion_fps (not every frame): it is the costliest
                # per-frame step; the newest mask is attached to the frames in between.
                rate = min(self.motion_fps, self.fps or self.motion_fps) if self.motion_fps > 0 else (self.fps or 15.0)
                if self.motion is None:
                    self.motion = MotionMask(self.motion_width, self.motion_history_s, rate)
                    self._mask, self._mask_ts = None, -1e9
                if ts - self._mask_ts >= 1.0 / rate * 0.95:
                    self._mask, self._mask_ts = self.motion.apply(frame), ts
                mask = self._mask
            pkt = FramePacket(self.id, frame, index, offset, ts, mask)
            with self._lock:
                self._latest = pkt
                self._latest_seq += 1
            index += 1
            n_stat += 1
            if time.time() - t_stat >= 2.0:
                self.decode_fps = n_stat / (time.time() - t_stat)
                t_stat, n_stat = time.time(), 0

    def _set_status(self, status: str, message: str) -> None:
        changed = status != self.status
        self.status = status
        if changed:
            (logbus.warning if status == "offline" else logbus.info)("camera", message)

    def info(self) -> dict:
        return {
            "id": self.id,
            "name": self.spec.name,
            "source": self.spec.source,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "resolution_weight": round(self.resolution_weight, 3),
            "status": self.status,
            "session_start": self.session_start,
            "is_file": self.spec.is_file,
        }


def assign_resolution_weights(cams: list[CameraStream], mode: str) -> None:
    """Derive w_r per camera from its real resolution unless configured explicitly."""
    pixels = [c.width * c.height for c in cams if c.width and c.height]
    biggest = max(pixels) if pixels else 1
    for c in cams:
        if c.spec.resolution_weight is not None:
            c.resolution_weight = float(c.spec.resolution_weight)
            continue
        ratio = (c.width * c.height) / biggest if biggest else 1.0
        if mode == "pixels":
            c.resolution_weight = ratio
        elif mode == "equal":
            c.resolution_weight = 1.0
        else:
            c.resolution_weight = math.sqrt(ratio)
