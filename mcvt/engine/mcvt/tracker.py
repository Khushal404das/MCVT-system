"""Single-camera tracking with DeepSORT (FR-TR-01 .. FR-TR-04).

DeepSORT's Kalman filter predicts a track's position while it is not detected
(FR-TR-03); a track that misses ``max_age`` consecutive frames is deleted
(FR-TR-02, FR-TR-04).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from deep_sort_realtime.deep_sort.track import TrackState as DSTrackState
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detector import Detection


@dataclass
class TrackState:
    """Output of the tracker for one confirmed track on the current frame."""

    track_id: str
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    label: str
    confidence: float
    detected: bool  # False = Kalman prediction only (vehicle briefly occluded)
    det_index: int | None  # index into this frame's detection list


class CameraTracker:
    """One DeepSORT instance per camera; local ids are unique per camera."""

    def __init__(self, camera_id: str, max_age: int = 30, n_init: int = 3,
                 max_iou_distance: float = 0.7, max_cosine_distance: float = 0.3):
        self.camera_id = camera_id
        self.max_age = max_age
        self.n_init = n_init
        self._ds = DeepSort(
            max_age=max_age,
            n_init=max(1, n_init),
            max_iou_distance=max_iou_distance,
            max_cosine_distance=max_cosine_distance,
            nn_budget=50,
            embedder=None,  # appearance vectors are supplied by the pipeline
        )
        self._alive: set[str] = set()

    def update(self, detections: list[Detection], embeds: list[np.ndarray], frame_shape) -> tuple[list[TrackState], set[str]]:
        """Advance the tracker one frame.

        Returns the visible confirmed tracks and the set of local ids that were
        removed during this update (stale tracks).
        """
        h, w = frame_shape[:2]
        raw, emb, others = [], [], []
        for i, (d, e) in enumerate(zip(detections, embeds)):
            ltwh = d.ltwh()
            if ltwh[2] <= 1 or ltwh[3] <= 1:
                continue
            raw.append((ltwh, d.confidence, d.label))
            emb.append(e)
            others.append(i)
        tracks = self._ds.update_tracks(raw, embeds=emb, others=others)
        out: list[TrackState] = []
        alive_now: set[str] = set()
        for t in tracks:
            if t.is_deleted():
                continue
            tid = str(t.track_id)
            alive_now.add(tid)
            # With n_init <= 1 a just-created (still tentative) track is reported immediately.
            if not t.is_confirmed() and not (self.n_init <= 1 and t.time_since_update == 0):
                continue
            detected = t.time_since_update == 0
            # Only show predictions for a few frames - long predictions drift.
            if not detected and t.time_since_update > 5:
                continue
            ltrb = t.to_ltrb(orig=detected) if detected else t.to_ltrb()
            if ltrb is None:
                ltrb = t.to_ltrb()
            x1, y1, x2, y2 = (float(v) for v in ltrb)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            det_index = t.get_det_supplementary() if detected else None
            out.append(TrackState(tid, (x1, y1, x2, y2), t.get_det_class() or "vehicle",
                                  float(t.get_det_conf() or 0.0), detected, det_index))
        removed = self._alive - alive_now
        self._alive = alive_now
        return out, removed

    def delete(self, track_id: str) -> None:
        """Drop a track whose vehicle was handed to another (re-linked) track of the same camera."""
        for t in self._ds.tracker.tracks:
            if str(t.track_id) == track_id:
                t.state = DSTrackState.Deleted
        self._alive.discard(track_id)
