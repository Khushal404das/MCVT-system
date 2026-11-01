"""
Single-Camera Vehicle Tracker
Uses a simplified DeepSORT-like approach with IoU + feature matching.
Maintains track IDs within a single camera view.
"""

import cv2
import numpy as np
from collections import deque, Counter
from collections import OrderedDict
import time


class Track:
    """Represents a single tracked vehicle within one camera."""
    
    _next_id = 1  # Class-level ID counter

    def __init__(self, track_id, bbox, feature, class_name, color_name, frame_idx):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.feature = feature  # Appearance feature vector
        self.feature_history = [feature]  # Keep history for averaging
        self.class_name = class_name
        self.class_counts = {class_name: 1} if class_name else {}
        self.color_name = color_name
        # Sliding window of recent color observations (last 15 frames)
        # Keeps color label fresh as vehicle approaches / changes illumination.
        self.color_window = deque([color_name] if (color_name and color_name.lower() != 'unknown') else [], maxlen=15)
        self.frame_idx = frame_idx
        self.last_seen = frame_idx
        self.age = 0  # Frames since creation
        self.hits = 1  # Number of successful matches
        self.time_since_update = 0
        self.positions = [(bbox, frame_idx)]  # Position history
        self.is_confirmed = False
        self.global_id = None  # Assigned during cross-camera matching

    def update(self, bbox, feature, class_name=None, color_name=None, frame_idx=None):
        """Update track with new detection."""
        self.bbox = bbox
        if frame_idx is not None:
            self.last_seen = frame_idx
        self.hits += 1
        self.time_since_update = 0
        self.positions.append((bbox, frame_idx if frame_idx is not None else 0))
        
        # Dynamic majority voting for vehicle class
        if class_name:
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
            self.class_name = max(self.class_counts, key=self.class_counts.get)
            
        # Dynamic majority voting for vehicle color (sliding window, last 15 frames)
        if color_name and color_name.lower() != 'unknown':
            self.color_window.append(color_name)
            if self.color_window:
                self.color_name = Counter(self.color_window).most_common(1)[0][0]

        # Update feature with exponential moving average
        if feature is not None and np.linalg.norm(feature) > 0:
            self.feature_history.append(feature)
            # Keep last 10 features
            if len(self.feature_history) > 10:
                self.feature_history = self.feature_history[-10:]
            # Average features for more robust representation
            self.feature = np.mean(self.feature_history, axis=0)
            norm = np.linalg.norm(self.feature)
            if norm > 0:
                self.feature = self.feature / norm
        
        if self.hits >= 3:
            self.is_confirmed = True

    def get_center(self):
        """Get center point of current bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def predict_next_position(self):
        """Simple linear prediction of next position."""
        if len(self.positions) < 2:
            return self.bbox
        
        prev_bbox, _ = self.positions[-2]
        curr_bbox = self.bbox
        
        # Velocity
        dx = curr_bbox[0] - prev_bbox[0]
        dy = curr_bbox[1] - prev_bbox[1]
        dw = (curr_bbox[2] - curr_bbox[0]) - (prev_bbox[2] - prev_bbox[0])
        dh = (curr_bbox[3] - curr_bbox[1]) - (prev_bbox[3] - prev_bbox[1])
        
        predicted = [
            curr_bbox[0] + dx,
            curr_bbox[1] + dy,
            curr_bbox[2] + dx + dw,
            curr_bbox[3] + dy + dh
        ]
        return predicted


class SingleCameraTracker:
    """
    Tracker for vehicles within a single camera view.
    Uses IoU + feature similarity for data association.
    """

    def __init__(self, camera_id, max_age=30, min_hits=3, 
                 iou_threshold=0.3, feature_threshold=0.6):
        """
        Args:
            camera_id: Identifier for this camera
            max_age: Maximum frames to keep a track without updates
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for matching
            feature_threshold: Minimum feature similarity for matching
        """
        self.camera_id = camera_id
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        self.tracks = OrderedDict()  # track_id -> Track
        self.next_id = 1
        self.frame_count = 0
        
        print(f"[Tracker-{camera_id}] Initialized")

    def update(self, detections, features, class_names, color_names, frame_idx=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: list of [x1, y1, x2, y2] bounding boxes
            features: list of feature vectors (same length as detections)
            class_names: list of vehicle class names
            color_names: list of dominant color names
            frame_idx: current frame index
            
        Returns:
            list of active Track objects
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        self.frame_count = frame_idx

        if len(detections) == 0:
            # Age all tracks
            self._age_tracks()
            return self.get_active_tracks()

        # Compute cost matrix (IoU + feature similarity)
        active_tracks = list(self.tracks.values())
        
        if len(active_tracks) == 0:
            # All detections are new tracks
            for i in range(len(detections)):
                self._create_track(detections[i], features[i], 
                                   class_names[i], color_names[i], frame_idx)
        else:
            # Hungarian-like matching
            matched, unmatched_dets, unmatched_tracks = self._match(
                active_tracks, detections, features
            )

            # Update matched tracks
            for track_idx, det_idx in matched:
                track = active_tracks[track_idx]
                track.update(detections[det_idx], features[det_idx],
                             class_names[det_idx], color_names[det_idx], frame_idx)

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                self._create_track(detections[det_idx], features[det_idx],
                                   class_names[det_idx], color_names[det_idx], frame_idx)

            # Age unmatched tracks
            for track_idx in unmatched_tracks:
                active_tracks[track_idx].time_since_update += 1

        # Remove dead tracks
        self._remove_dead_tracks()

        return self.get_active_tracks()

    def _match(self, tracks, detections, features):
        """Match detections to existing tracks using IoU + feature similarity."""
        n_tracks = len(tracks)
        n_dets = len(detections)
        
        # Cost matrix
        cost_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        
        for i, track in enumerate(tracks):
            predicted_bbox = track.predict_next_position()
            for j in range(n_dets):
                iou = self._compute_iou(predicted_bbox, detections[j])
                feat_sim = self._compute_feature_similarity(track.feature, features[j])
                
                # Combined cost (lower is better)
                # Weight: 40% IoU, 60% feature similarity
                cost_matrix[i, j] = 1.0 - (0.4 * iou + 0.6 * feat_sim)
        
        # Greedy matching (simpler than Hungarian, works well for small numbers)
        matched = []
        unmatched_dets = set(range(n_dets))
        unmatched_tracks = set(range(n_tracks))
        
        # Sort by cost (best matches first)
        indices = np.argwhere(cost_matrix < 0.7)  # Only consider reasonable matches
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
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union = area1 + area2 - inter
        
        return inter / max(union, 1e-6)

    def _compute_feature_similarity(self, feat1, feat2):
        """Compute cosine similarity between feature vectors."""
        if feat1 is None or feat2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(feat1, feat2) / (norm1 * norm2 + 1e-6))

    def _create_track(self, bbox, feature, class_name, color_name, frame_idx):
        """Create a new track."""
        track = Track(self.next_id, bbox, feature, class_name, color_name, frame_idx)
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def _age_tracks(self):
        """Increment time_since_update for all tracks."""
        for track in self.tracks.values():
            track.time_since_update += 1

    def _remove_dead_tracks(self):
        """Remove tracks that haven't been seen recently."""
        dead_ids = []
        for track_id, track in self.tracks.items():
            if track.time_since_update > self.max_age:
                dead_ids.append(track_id)
        for tid in dead_ids:
            del self.tracks[tid]

    def get_active_tracks(self):
        """Get all currently active (visible) tracks."""
        return [t for t in self.tracks.values() 
                if t.time_since_update == 0 and t.hits >= 1]

    def get_confirmed_tracks(self):
        """Get all confirmed tracks (seen enough times)."""
        return [t for t in self.tracks.values() 
                if t.is_confirmed and t.time_since_update < 5]

    def get_track_by_id(self, track_id):
        """Get a specific track by its ID."""
        return self.tracks.get(track_id)
