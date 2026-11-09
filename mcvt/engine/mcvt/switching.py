"""Camera scoring and switching engine (FR-CS-01 .. FR-CS-06).

* Score of a camera for a vehicle:  bbox_area / frame_area * w_r   (FR-CS-01)
* A small Dynamic Bayesian Network (an HMM over "which camera is best") keeps a
  belief per camera that combines the new scores with the score history
  (FR-CS-02).
* A switch needs the candidate to beat the active camera by the hysteresis
  margin (FR-CS-03) for N consecutive frames (FR-CS-04).
* If the active camera loses the vehicle for ``fallback_frames`` frames, the
  engine falls back to the best remaining camera immediately (FR-CS-06).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

log = logging.getLogger("switching")


def camera_score(bbox_area: float, frame_area: float, resolution_weight: float) -> float:
    """FR-CS-01 score."""
    if frame_area <= 0:
        return 0.0
    return float(bbox_area) / float(frame_area) * float(resolution_weight)


@dataclass
class SwitchDecision:
    active: str | None
    switched: bool
    previous: str | None
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    belief: dict[str, float] = field(default_factory=dict)


@dataclass
class OverlapDetection:
    camera_id: str
    bbox_area: float
    frame_area: float
    resolution_weight: float = 1.0

    @property
    def score(self) -> float:
        return camera_score(self.bbox_area, self.frame_area, self.resolution_weight)


class CameraSwitchingEngine:
    """Decides which camera should be on the live view for one tracked vehicle."""

    def __init__(self, hysteresis: float = 0.15, buffer_frames: int = 5, fallback_frames: int = 3,
                 stay_probability: float = 0.85, evidence_gain: float = 8.0):
        self.hysteresis = float(hysteresis)
        self.buffer_frames = int(buffer_frames)
        self.fallback_frames = int(fallback_frames)
        self.stay = float(stay_probability)
        self.gain = float(evidence_gain)
        self.active: str | None = None
        self.belief: dict[str, float] = {}
        self._candidate: str | None = None
        self._candidate_frames = 0
        self._missing = 0
        self.last_reason = ""

    # ------------------------------------------------------------------ DBN
    def _update_belief(self, scores: dict[str, float]) -> None:
        cams = set(self.belief) | set(scores)
        if not cams:
            return
        n = len(cams)
        prior = {c: self.belief.get(c, 1.0 / n) for c in cams}
        total = sum(prior.values()) or 1.0
        prior = {c: p / total for c, p in prior.items()}
        move = (1.0 - self.stay) / (n - 1) if n > 1 else 0.0
        post = {}
        for c in cams:
            predicted = prior[c] * self.stay + (1.0 - prior[c]) * move if n > 1 else 1.0
            post[c] = predicted * math.exp(self.gain * scores.get(c, 0.0))
        z = sum(post.values()) or 1.0
        self.belief = {c: p / z for c, p in post.items() if p / z > 1e-6 or c in scores}

    # -------------------------------------------------------------- decision
    def selectBestCamera(self, scores: dict[str, float]) -> str | None:
        """Feed one frame of scores (only cameras that see the vehicle) and
        return the camera that should be active."""
        return self.update(scores).active

    select_best_camera = selectBestCamera

    def update(self, scores: dict[str, float]) -> SwitchDecision:
        scores = {c: float(s) for c, s in scores.items() if s is not None and s > 0}
        self._update_belief(scores)
        prev = self.active

        # Initial selection: no active camera yet.
        if self.active is None:
            if not scores:
                return self._decide(False, prev, "Vehicle not visible in any camera", scores)
            best = max(scores, key=lambda c: (scores[c], self.belief.get(c, 0)))
            self.active = best
            self._reset_candidate()
            return self._decide(True, prev, f"Initial camera {best} selected (score {scores[best]:.2f})", scores)

        # Fallback when the active camera lost the vehicle (FR-CS-06).
        if self.active not in scores:
            self._missing += 1
            if self._missing >= self.fallback_frames and scores:
                best = max(scores, key=lambda c: (self.belief.get(c, 0), scores[c]))
                self.active = best
                self._missing = 0
                self._reset_candidate()
                return self._decide(True, prev, f"Camera {prev} lost the vehicle for {self.fallback_frames} frames; "
                                                f"falling back to {best} (score {scores[best]:.2f})", scores)
            return self._decide(False, prev, f"Camera {self.active} lost the vehicle ({self._missing} frame(s))", scores)
        self._missing = 0

        active_score = scores[self.active]
        # Candidates must beat the active camera by the hysteresis margin (FR-CS-03).
        better = {c: s for c, s in scores.items() if c != self.active and s - active_score > self.hysteresis}
        if not better:
            self._reset_candidate()
            challenger = max((c for c in scores if c != self.active), key=lambda c: scores[c], default=None)
            if challenger is not None and scores[challenger] > active_score:
                reason = (f"No switch: {challenger} leads {self.active} by {scores[challenger] - active_score:.2f}, "
                          f"below the hysteresis threshold {self.hysteresis:.2f}")
            else:
                reason = f"Camera {self.active} is still the best view"
            return self._decide(False, prev, reason, scores)

        # Among qualifying cameras prefer the one the DBN believes in most.
        cand = max(better, key=lambda c: (self.belief.get(c, 0), better[c]))
        if cand == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = cand, 1
        if self._candidate_frames >= self.buffer_frames:  # FR-CS-04
            self.active = cand
            self._reset_candidate()
            return self._decide(True, prev, f"Switched {prev} -> {cand}: score {scores[cand]:.2f} vs {active_score:.2f} "
                                            f"for {self.buffer_frames} consecutive frames", scores)
        return self._decide(False, prev, f"{cand} is better ({scores[cand]:.2f} vs {active_score:.2f}); "
                                         f"waiting {self._candidate_frames}/{self.buffer_frames} frames", scores)

    def _reset_candidate(self) -> None:
        self._candidate, self._candidate_frames = None, 0

    def _decide(self, switched: bool, prev: str | None, reason: str, scores: dict[str, float]) -> SwitchDecision:
        self.last_reason = reason
        if not switched:
            log.debug(reason)
        return SwitchDecision(self.active, switched, prev, reason, dict(scores), dict(self.belief))

    def set_active(self, camera_id: str | None) -> None:
        self.active = camera_id
        self._reset_candidate()
        self._missing = 0

    # -------------------------------------------------------------- overlap
    @staticmethod
    def handleOverlapZone(detections: list[OverlapDetection]) -> tuple[str | None, list[str]]:
        """Same vehicle seen by several cameras at once (FR-CS-05): keep the
        best-scoring camera, return it plus the cameras whose duplicate is suppressed."""
        if not detections:
            return None, []
        best = max(detections, key=lambda d: d.score)
        return best.camera_id, [d.camera_id for d in detections if d.camera_id != best.camera_id]

    handle_overlap_zone = handleOverlapZone
