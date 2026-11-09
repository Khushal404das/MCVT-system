"""Keep overlay boxes on moving vehicles between AI updates (sparse optical flow).

The detector runs a few times per second at most on a CPU, while video is
streamed at the camera's full frame rate. Instead of guessing where a box went,
feature points inside each box are followed from frame to frame with
Lucas-Kanade optical flow and the box is moved with them.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np

_LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class FlowBoxes:
    """Per-camera box follower."""

    def __init__(self, width: int = 480):
        self.width = width
        self.prev: np.ndarray | None = None
        self.items: list[dict] = []
        self.src_ts: float | None = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w > self.width:
            frame = cv2.resize(frame, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def reset(self, analysed_frame: np.ndarray, items: list[dict], ts: float) -> None:
        """New AI result: boxes are exact for ``analysed_frame``."""
        self.prev = self._gray(analysed_frame)
        self.items = copy.deepcopy(items)
        self.src_ts = ts

    def step(self, frame: np.ndarray) -> list[dict]:
        """Move every box from the previous frame to ``frame`` and return the items."""
        g = self._gray(frame)
        if self.prev is None or not self.items or self.prev.shape != g.shape:
            self.prev = g
            return self.items
        H, W = g.shape
        for it in self.items:
            b = it["box"]
            x1, y1, x2, y2 = int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            pts = cv2.goodFeaturesToTrack(self.prev[y1:y2, x1:x2], maxCorners=30, qualityLevel=0.01, minDistance=3)
            if pts is None or len(pts) < 4:
                continue
            pts = pts.reshape(-1, 2) + np.array([x1, y1], np.float32)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, pts.reshape(-1, 1, 2), None, **_LK)
            ok = st.reshape(-1) == 1
            if ok.sum() < 4:
                continue
            d = nxt.reshape(-1, 2)[ok] - pts[ok]
            mag = np.hypot(d[:, 0], d[:, 1])
            # Boxes are a little larger than the car: ignore background points that did not move.
            moving = mag > 0.4
            use = d[moving] if moving.sum() >= max(3, 0.3 * len(d)) else d
            dx, dy = np.median(use, axis=0)
            if abs(dx) > (x2 - x1) or abs(dy) > (y2 - y1):
                continue  # implausible jump - keep the box where it is
            shift = np.array([dx / W, dy / H, dx / W, dy / H], np.float32)
            it["box"] = np.clip(b + shift, 0.0, 1.0)
        self.prev = g
        return self.items
