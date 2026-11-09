"""Vehicle attributes: colour and type.

Each vehicle's colour (from its crops) and type (the detector's class) are stored
with its global id. When a vehicle appears on another camera, ReID compares its
appearance features with the stored vehicles, and a stored vehicle whose colour or
type clearly disagrees cannot give it its id (see ``AttributeGate``), as in the
attribute re-ranking of UWIPL's AI-City Track-2 system.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

_HUES = [  # (upper hue bound on OpenCV 0..180 scale, name)
    (8, "red"), (20, "orange"), (33, "yellow"), (85, "green"),
    (100, "cyan"), (130, "blue"), (160, "purple"), (180, "red"),
]


def dominant_color(crop: np.ndarray) -> str | None:
    """Name the main body colour of a vehicle crop."""
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    c = crop[h // 5: h - h // 5, w // 6: w - w // 6]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    votes: Counter = Counter()
    dark = val < 55
    votes["black"] += int(dark.sum())
    grey = (~dark) & (sat < 45)
    votes["white"] += int((grey & (val >= 185)).sum())
    votes["silver"] += int((grey & (val >= 120) & (val < 185)).sum())
    votes["grey"] += int((grey & (val < 120)).sum())
    chroma = (~dark) & (~grey)
    if chroma.any():
        hues = hue[chroma]
        lower = 0
        for upper, name in _HUES:
            votes[name] += int(((hues >= lower) & (hues < upper)).sum())
            lower = upper
    if not votes:
        return None
    name, _ = votes.most_common(1)[0]
    return name


class ColorNamer:
    """Names a vehicle's body colour, corrected for each camera's colour cast.

    Cameras tint the picture (here C31/C41 are strongly blue), which made white and
    silver cars read as "blue". The cast of each camera is learned from the vehicles it
    sees (most vehicles are white/silver/grey/black, so their median tint is the camera's)
    and removed before naming, in CIE-LAB:
      * chroma (distance from neutral) >= chroma_min on > colored_fraction of the body
        -> a hue name (red, orange, yellow, green, blue, purple);
      * otherwise the brightness of the body (windows and shadows ignored) gives
        white / silver / grey / black.
    """

    # LAB hue angle (degrees) upper bounds: red ~10-40, orange ~60, yellow ~100, green ~140,
    # teal/blue ~195-300, purple ~320
    HUES = ((48, "red"), (80, "orange"), (115, "yellow"), (175, "green"), (310, "blue"), (345, "purple"), (360, "red"))

    def __init__(self, cfg: dict | None = None, state_file: Path | None = None):
        cfg = cfg or {}
        self.chroma_min = float(cfg.get("chroma_min", 14))
        self.colored_fraction = float(cfg.get("colored_fraction", 0.4))
        lv = cfg.get("lightness_levels", [58, 42, 25])  # white / silver / grey lower bounds (L 0..100)
        self.levels = [float(x) for x in lv]
        self.neutral_chroma = float(cfg.get("cast_neutral_chroma", 25))
        self.cast_min = int(cfg.get("cast_min_samples", 8))
        self.cast_samples = int(cfg.get("cast_samples", 200))
        self.state_file = state_file
        self._casts: dict[str, deque] = {}
        self._lock = threading.Lock()
        if state_file and state_file.exists():
            try:
                for cam, vals in json.loads(state_file.read_text()).items():
                    self._casts[cam] = deque((tuple(v) for v in vals), maxlen=self.cast_samples)
            except (ValueError, OSError, TypeError):
                pass

    @staticmethod
    def _body_lab(crop: np.ndarray) -> np.ndarray | None:
        h, w = crop.shape[:2]
        if h < 8 or w < 8:
            return None
        c = crop[h // 5: h - h // 6, w // 6: w - w // 6]
        return cv2.cvtColor(c, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    def cast(self, camera_id: str) -> np.ndarray | None:
        with self._lock:
            d = self._casts.get(camera_id)
            return np.median(np.array(d), axis=0) if d and len(d) >= self.cast_min else None

    def observe(self, camera_id: str, crop: np.ndarray) -> str | None:
        """Learn the camera cast from this crop and name its colour (None until the camera's
        cast is known, i.e. after ``cast_min_samples`` vehicles)."""
        lab = self._body_lab(crop)
        if lab is None:
            return None
        ab = lab[:, 1:] - 128
        neutral = ab[np.hypot(ab[:, 0], ab[:, 1]) < self.neutral_chroma]
        if len(neutral):
            with self._lock:
                self._casts.setdefault(camera_id, deque(maxlen=self.cast_samples)).append(
                    tuple(float(x) for x in np.median(neutral, axis=0)))
        cast = self.cast(camera_id)
        return None if cast is None else self._name(lab, cast)

    def _name(self, lab: np.ndarray, cast: np.ndarray) -> str:
        a, b = lab[:, 1] - 128 - cast[0], lab[:, 2] - 128 - cast[1]
        colored = np.hypot(a, b) >= self.chroma_min
        if colored.mean() > self.colored_fraction:
            ang = (np.degrees(np.arctan2(b[colored], a[colored])) + 360) % 360
            votes, lo = Counter(), 0
            for up, name in self.HUES:
                votes[name] += int(((ang >= lo) & (ang < up)).sum())
                lo = up
            return votes.most_common(1)[0][0]
        L = np.sort(lab[~colored, 0] * (100.0 / 255.0))
        body = float(np.median(L[len(L) // 3:]))  # brightest two thirds: the body, not windows/shadow
        for level, name in zip(self.levels, ("white", "silver", "grey")):
            if body >= level:
                return name
        return "black"

    def save(self) -> None:
        if not self.state_file:
            return
        with self._lock:
            data = {c: list(d) for c, d in self._casts.items()}
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(data))
        except OSError:
            pass


# Neutral colours form a ladder: neighbours (white/silver, silver/grey, grey/black) are often
# confused between cameras; two or more steps apart is a different paint.
_LADDER = {"white": 0, "silver": 1, "grey": 2, "black": 3}


def color_relation(a: str | None, b: str | None, max_steps: int = 1) -> str | None:
    """'equal', 'near' (neutral shades at most ``max_steps`` apart on the white-silver-grey-black
    ladder) or 'conflict'; None if either is unknown."""
    if not a or not b:
        return None
    if a == b:
        return "equal"
    if a in _LADDER and b in _LADDER and abs(_LADDER[a] - _LADDER[b]) <= max_steps:
        return "near"
    return "conflict"


def _majority(counter: Counter | None, min_votes: float, min_share: float):
    if not counter:
        return None
    total = sum(counter.values())
    name, n = counter.most_common(1)[0]
    return name if total >= min_votes and n / total >= min_share else None


class AttributeGate:
    """Decides whether a stored vehicle's colour/type rule it out for a new track."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.use_color = bool(cfg.get("color", True))
        self.use_type = bool(cfg.get("type", True))
        self.min_votes = float(cfg.get("min_votes", 2))  # stored vehicle
        # a new track: moving cars get their id from their first crop, so one reading has to count
        self.track_min_votes = float(cfg.get("track_min_votes", 1))
        self.min_share = float(cfg.get("min_share", 0.6))
        # how far apart neutral shades may be for one car: cameras with different exposure call
        # the same white car "white" on one camera and "grey" on the next (C11 -> C21 here)
        self.neutral_max_steps = int(cfg.get("neutral_max_steps", 1))
        # type votes are detector confidences summed over detections; a motorbike passing a car can
        # lend it a stray label, so several confident detections are needed
        self.type_min_votes = float(cfg.get("type_min_votes", 3.0))
        # classes that can be the same vehicle (the detector mixes car/truck for SUVs and pickups)
        groups = cfg.get("type_groups", [["car", "truck", "bus"], ["motorcycle", "bicycle"]])
        self.type_group = {str(c): i for i, g in enumerate(groups) for c in g}

    def describe(self, colors: Counter | None, types: Counter | None,
                 min_votes: float | None = None) -> tuple[str | None, str | None]:
        return (_majority(colors, self.min_votes if min_votes is None else min_votes, self.min_share),
                _majority(types, self.type_min_votes, self.min_share))

    def conflict(self, colors: Counter | None, types: Counter | None,
                 stored_colors: Counter | None, stored_types: Counter | None) -> str | None:
        """Reason the two cannot be one vehicle (e.g. 'colour white vs black'), else None."""
        if not self.enabled:
            return None
        c1, t1 = self.describe(colors, types, self.track_min_votes)
        c2, t2 = self.describe(stored_colors, stored_types)
        if self.use_type and t1 and t2 and t1 != t2:
            g1, g2 = self.type_group.get(t1, t1), self.type_group.get(t2, t2)
            if g1 != g2:
                return f"type {t1} vs {t2}"
        if self.use_color and color_relation(c1, c2, self.neutral_max_steps) == "conflict":
            return f"colour {c1} vs {c2}"
        return None


class VoteBox:
    """Majority vote that stabilises noisy per-frame attributes."""

    def __init__(self):
        self.counter: Counter = Counter()

    def add(self, value, weight: float = 1.0) -> None:
        if value:
            self.counter[value] += weight

    def value(self):
        return self.counter.most_common(1)[0][0] if self.counter else None
