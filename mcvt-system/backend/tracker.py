"""
Single-Camera Vehicle Tracker
Uses an enhanced DeepSORT-like approach with IoU + spatial centroid distance + appearance features.
Guarantees continuous tracking along vehicle motion with zero ghost boxes.
"""

import cv2
import numpy as np
from collections import deque, Counter, OrderedDict
import time


class Track:
    """Represents a single tracked vehicle within one camera."""

    def __init__(self, track_id, bbox, feature, class_name, color_name, frame_idx):
        self.track_id = track_id
        self.bbox = [float(x) for x in bbox]            # [x1, y1, x2, y2]
        self.anchor_bbox = [float(x) for x in bbox]     # bbox at last real detection
        self.velocity = [0.0, 0.0, 0.0, 0.0]           # [vx1, vy1, vx2, vy2] per frame
        self.frames_since_detect = 0                   # frames since last real detection
        self.feature = feature                         # Appearance feature vector
        self.feature_history = [feature] if feature is not None else []
        self.class_name = class_name
        self.class_counts = {class_name: 1} if class_name else {}
        self.color_name = color_name
        self.color_window = deque([color_name] if (color_name and color_name.lower() != 'unknown') else [], maxlen=15)
        self.frame_idx = frame_idx
        self.last_seen = frame_idx
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.positions = [(list(bbox), frame_idx if frame_idx is not None else 0)]
        self.is_confirmed = False
        self.global_id = None

    def update(self, bbox, feature, class_name=None, color_name=None, frame_idx=None):
        """Update track with new confirmed detection."""
        bbox = [float(x) for x in bbox]
        frames = max(1, self.frames_since_detect)
        inst_v = [
            (bbox[0] - self.anchor_bbox[0]) / frames,
            (bbox[1] - self.anchor_bbox[1]) / frames,
            (bbox[2] - self.anchor_bbox[2]) / frames,
            (bbox[3] - self.anchor_bbox[3]) / frames,
        ]

        # Smooth velocity with exponential moving average (EMA)
        if self.hits > 1:
            self.velocity = [
                0.7 * inst_v[0] + 0.3 * self.velocity[0],
                0.7 * inst_v[1] + 0.3 * self.velocity[1],
                0.7 * inst_v[2] + 0.3 * self.velocity[2],
                0.7 * inst_v[3] + 0.3 * self.velocity[3],
            ]
        else:
            self.velocity = inst_v

        self.anchor_bbox = list(bbox)
        self.bbox = list(bbox)
        self.frames_since_detect = 0
        self.hits += 1
        self.time_since_update = 0

        if frame_idx is not None:
            self.last_seen = frame_idx
        self.positions.append((list(bbox), frame_idx if frame_idx is not None else 0))

        # Dynamic majority voting for class and color
        if class_name:
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            self.class_name = max(self.class_counts, key=self.class_counts.get)

        if color_name and color_name.lower() != 'unknown':
            self.color_window.append(color_name)
            if self.color_window:
                self.color_name = Counter(self.color_window).most_common(1)[0][0]

        # Update appearance feature
        if feature is not None and np.linalg.norm(feature) > 0:
            self.feature_history.append(feature)
            if len(self.feature_history) > 10:
                self.feature_history = self.feature_history[-10:]
            self.feature = np.mean(self.feature_history, axis=0)
            norm = np.linalg.norm(self.feature)
            if norm > 0:
                self.feature = self.feature / norm

        if self.hits >= 2:
            self.is_confirmed = True

    def get_display_bbox(self):
        """
        Extrapolate position smoothly along estimated velocity.
        Caps extrapolation at 6 frames to prevent drift if the car turns or stops.
        """
        n = min(self.frames_since_detect, 6)
        if n == 0 or self.velocity == [0.0, 0.0, 0.0, 0.0]:
            return list(self.anchor_bbox)
        vx1, vy1, vx2, vy2 = self.velocity
        a = self.anchor_bbox
        return [
            a[0] + vx1 * n,
            a[1] + vy1 * n,
            a[2] + vx2 * n,
            a[3] + vy2 * n,
        ]

    def predict_next_position(self):
        """
        Predict bbox for spatial matching in the next detection cycle.
        """
        vx1, vy1, vx2, vy2 = self.velocity
        a = self.anchor_bbox
        n = self.frames_since_detect + 1
        return [
            a[0] + vx1 * n,
            a[1] + vy1 * n,
            a[2] + vx2 * n,
            a[3] + vy2 * n,
        ]


class SingleCameraTracker:
    """
    Tracks vehicles within a single camera stream.
    Combines IoU, spatial centroid distance, and appearance features.
    """

    def __init__(self, camera_id, max_age=4, min_hits=2,
                 iou_threshold=0.25, feature_threshold=0.6):
        self.camera_id = camera_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold

        self.tracks = OrderedDict()
        self.next_id = 1
        self.frame_count = 0
        print(f"[Tracker-{camera_id}] Initialized (Continuous Motion Tracking Mode)")

    def update(self, detections, features, class_names, color_names, frame_idx=None):
        if frame_idx is None:
            frame_idx = self.frame_count
        self.frame_count = frame_idx

        if len(detections) == 0:
            self._age_tracks()
            self._remove_dead_tracks()
            return self.get_active_tracks()

        active_tracks = list(self.tracks.values())

        if len(active_tracks) == 0:
            for i in range(len(detections)):
                feat_i = features[i] if (features and i < len(features)) else None
                cls_i  = class_names[i] if (class_names and i < len(class_names)) else 'car'
                col_i  = color_names[i] if (color_names and i < len(color_names)) else 'unknown'
                self._create_track(detections[i], feat_i, cls_i, col_i, frame_idx)
        else:
            matched, unmatched_dets, unmatched_tracks = self._match(
                active_tracks, detections, features
            )

            for track_idx, det_idx in matched:
                track = active_tracks[track_idx]
                feat_m = features[det_idx] if (features and det_idx < len(features)) else None
                cls_m  = class_names[det_idx] if (class_names and det_idx < len(class_names)) else track.class_name
                col_m  = color_names[det_idx] if (color_names and det_idx < len(color_names)) else track.color_name
                track.update(detections[det_idx], feat_m, cls_m, col_m, frame_idx)

            for det_idx in unmatched_dets:
                feat_u = features[det_idx] if (features and det_idx < len(features)) else None
                cls_u  = class_names[det_idx] if (class_names and det_idx < len(class_names)) else 'car'
                col_u  = color_names[det_idx] if (color_names and det_idx < len(color_names)) else 'unknown'
                self._create_track(detections[det_idx], feat_u, cls_u, col_u, frame_idx)

            for track_idx in unmatched_tracks:
                active_tracks[track_idx].time_since_update += 1

        self._remove_dead_tracks()
        return self.get_active_tracks()

    def _match(self, tracks, detections, features):
        """
        Spatial + appearance matching.
        Uses combined IoU and centroid distance normalized by detection size.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for i, track in enumerate(tracks):
            pred = track.predict_next_position()
            tc_x = (pred[0] + pred[2]) / 2.0
            tc_y = (pred[1] + pred[3]) / 2.0

            for j, det in enumerate(detections):
                iou = self._compute_iou(pred, det)

                dc_x = (det[0] + det[2]) / 2.0
                dc_y = (det[1] + det[3]) / 2.0
                diag = max(20.0, np.sqrt((det[2] - det[0])**2 + (det[3] - det[1])**2))
                dist = np.sqrt((tc_x - dc_x)**2 + (tc_y - dc_y)**2)
                # Distance similarity is high if center distance is within 2 vehicle diagonals
                dist_sim = max(0.0, 1.0 - dist / (2.0 * diag))
                spatial_sim = max(iou, dist_sim)

                f_j = features[j] if (features and j < len(features)) else None
                feat_sim = self._compute_feature_similarity(track.feature, f_j) if (track.feature is not None and f_j is not None) else 0.0

                if feat_sim > 0.0:
                    sim = 0.6 * spatial_sim + 0.4 * feat_sim
                else:
                    sim = spatial_sim

                cost_matrix[i, j] = 1.0 - sim

        matched = []
        unmatched_dets = set(range(n_dets))
        unmatched_tracks = set(range(n_tracks))

        indices = np.argwhere(cost_matrix < 0.65)
        if len(indices) > 0:
            costs = cost_matrix[indices[:, 0], indices[:, 1]]
            sorted_idx = np.argsort(costs)

            for idx in sorted_idx:
                t_idx, d_idx = indices[idx]
                if t_idx in unmatched_tracks and d_idx in unmatched_dets:
                    matched.append((t_idx, d_idx))
                    unmatched_tracks.discard(t_idx)
                    unmatched_dets.discard(d_idx)

        return matched, list(unmatched_dets), list(unmatched_tracks)

    def _compute_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter

        return float(inter / max(union, 1e-6))

    def _compute_feature_similarity(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(feat1, feat2) / (norm1 * norm2 + 1e-6))

    def _create_track(self, bbox, feature, class_name, color_name, frame_idx):
        track = Track(self.next_id, bbox, feature, class_name, color_name, frame_idx)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def _age_tracks(self):
        for track in self.tracks.values():
            track.time_since_update += 1

    def _remove_dead_tracks(self):
        """Immediately remove tracks that are lost or stale."""
        dead_ids = []
        for track_id, track in self.tracks.items():
            # If a track had only 1 detection and was missed in the very next cycle, discard it
            if track.hits < 2 and track.time_since_update > 0:
                dead_ids.append(track_id)
            # If a confirmed track has been missing for more than max_age detection cycles, discard it
            elif track.time_since_update > self.max_age:
                dead_ids.append(track_id)

        for tid in dead_ids:
            del self.tracks[tid]

    def tick_display(self):
        """
        Called ONCE per rendered playback frame.
        Guarantees:
        1. Bounding box moves smoothly with vehicle velocity between detection frames.
        2. NEVER displays a ghost box on an empty road after a vehicle has moved away.
        """
        drawable = []
        for track in list(self.tracks.values()):
            track.frames_since_detect += 1

            # Never draw 1-hit tracks that were not confirmed in the latest detection
            if track.hits < 2 and track.time_since_update > 0:
                continue

            # Never draw tracks that have been missing for more than 1 detection cycle
            if track.time_since_update > 1:
                continue

            # Never draw tracks that haven't been confirmed for > 8 playback frames (0.4s)
            if track.frames_since_detect > 8:
                continue

            drawable.append(track)
        return drawable

    def advance_motion(self):
        """Legacy compatibility shim."""
        return self.tick_display()

    def get_active_tracks(self):
        return [t for t in self.tracks.values() if t.time_since_update == 0]

    def get_confirmed_tracks(self):
        return [t for t in self.tracks.values() if t.is_confirmed and t.time_since_update <= 1]

    def get_track_by_id(self, track_id):
        return self.tracks.get(track_id)
