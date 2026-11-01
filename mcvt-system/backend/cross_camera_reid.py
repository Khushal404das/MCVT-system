"""
Cross-Camera Vehicle Re-Identification Engine
Matches vehicles across multiple cameras using feature similarity.

Approach:
- Computes pairwise feature distances between vehicles from different cameras
- Uses a global gallery with unique IDs per physical vehicle
- Merges duplicate global IDs when a vehicle is confirmed to be the same
- Considers color + class consistency to reduce false positives
"""

import numpy as np
from collections import defaultdict
import time
import threading


class VehicleGallery:
    """
    Maintains a gallery of tracked vehicles across all cameras.
    Each entry represents one PHYSICAL vehicle with a UNIQUE global ID.
    Multiple cameras may report the same physical vehicle — they all share one global ID.
    """

    def __init__(self):
        self.entries = {}           # global_id -> GalleryEntry
        self.next_global_id = 1
        self.lock = threading.Lock()

    def add_entry(self, feature, camera_id, local_track_id, class_name, color_name, bbox, frame_idx):
        """Add a new vehicle to the gallery with a new global ID."""
        with self.lock:
            gid = self.next_global_id
            self.next_global_id += 1
            self.entries[gid] = GalleryEntry(
                global_id=gid,
                feature=feature.copy(),
                camera_id=camera_id,
                local_track_id=local_track_id,
                class_name=class_name,
                color_name=color_name,
                bbox=bbox,
                frame_idx=frame_idx
            )
            return gid

    def update_entry(self, global_id, feature, camera_id, local_track_id, bbox, frame_idx,
                     class_name=None, color_name=None):
        """Update an existing gallery entry with a new observation."""
        with self.lock:
            if global_id in self.entries:
                entry = self.entries[global_id]
                entry.update(feature, camera_id, local_track_id, bbox, frame_idx, class_name, color_name)

    def merge_entries(self, keep_id, remove_id):
        """
        Merge two gallery entries that represent the same physical vehicle.
        The entry with keep_id absorbs the history of remove_id, then remove_id is deleted.
        """
        with self.lock:
            if keep_id not in self.entries or remove_id not in self.entries:
                return
            keep = self.entries[keep_id]
            drop = self.entries[remove_id]
            # Absorb feature history
            for f in drop.feature_history:
                keep.feature_history.append(f)
            if len(keep.feature_history) > 30:
                keep.feature_history = keep.feature_history[-30:]
            # Recompute mean feature
            keep.feature = np.mean(keep.feature_history, axis=0)
            norm = np.linalg.norm(keep.feature)
            if norm > 0:
                keep.feature = keep.feature / norm
            # Merge camera history
            keep.camera_history.extend(drop.camera_history)
            # Use earliest creation time
            if drop.created_at < keep.created_at:
                keep.created_at = drop.created_at
            del self.entries[remove_id]
            print(f"[Gallery] Merged G{remove_id} → G{keep_id} (same physical vehicle)")

    def get_entry(self, global_id):
        return self.entries.get(global_id)

    def get_all_entries(self):
        return list(self.entries.values())

    def get_recent_entries(self, max_age_seconds=30):
        recent = []
        for entry in self.entries.values():
            if (time.time() - entry.last_updated) < max_age_seconds:
                recent.append(entry)
        return recent


class GalleryEntry:
    """A single physical vehicle in the global gallery."""

    def __init__(self, global_id, feature, camera_id, local_track_id,
                 class_name, color_name, bbox, frame_idx):
        self.global_id = global_id
        self.feature = feature.copy()
        self.feature_history = [feature.copy()]
        self.class_name = class_name
        self.color_name = color_name
        self.last_bbox = bbox
        self.last_frame_idx = frame_idx
        self.created_at = time.time()
        self.last_updated = time.time()

        # camera_id -> {'local_id': ..., 'last_seen': timestamp, 'last_frame': frame_idx}
        self.active_cameras = {
            camera_id: {
                'local_id': local_track_id,
                'last_seen': time.time(),
                'last_frame': frame_idx
            }
        }
        # Full camera visit log
        self.camera_history = [(camera_id, local_track_id, frame_idx, time.time())]

        # Most recently updated camera
        self.current_camera = camera_id
        self.current_local_id = local_track_id

    def update(self, feature, camera_id, local_track_id, bbox, frame_idx,
               class_name=None, color_name=None):
        """Update with a new observation from any camera."""
        # Accumulate feature history (cap at 30)
        self.feature_history.append(feature.copy())
        if len(self.feature_history) > 30:
            self.feature_history = self.feature_history[-30:]

        # Recompute mean feature
        self.feature = np.mean(self.feature_history, axis=0)
        norm = np.linalg.norm(self.feature)
        if norm > 0:
            self.feature = self.feature / norm

        self.last_bbox = bbox
        self.last_frame_idx = frame_idx
        self.last_updated = time.time()

        if class_name:
            self.class_name = class_name
        if color_name and color_name not in ('Unknown', 'unknown'):
            self.color_name = color_name

        # Update per-camera presence record
        now = time.time()
        if camera_id not in self.active_cameras or \
                self.active_cameras[camera_id]['local_id'] != local_track_id:
            self.camera_history.append((camera_id, local_track_id, frame_idx, now))
        self.active_cameras[camera_id] = {
            'local_id': local_track_id,
            'last_seen': now,
            'last_frame': frame_idx
        }
        self.current_camera = camera_id
        self.current_local_id = local_track_id

    def is_currently_in_camera(self, camera_id, max_gap_sec=2.5):
        """Return True if this vehicle was recently seen in `camera_id`."""
        if camera_id not in self.active_cameras:
            return False
        return (time.time() - self.active_cameras[camera_id]['last_seen']) < max_gap_sec

    def get_cameras_visited(self):
        return list(set(h[0] for h in self.camera_history))

    def to_dict(self):
        return {
            'global_id': self.global_id,
            'class_name': self.class_name,
            'color_name': self.color_name,
            'current_camera': self.current_camera,
            'current_local_id': self.current_local_id,
            'last_bbox': self.last_bbox,
            'cameras_visited': self.get_cameras_visited(),
            'camera_history': [
                {'camera': h[0], 'track_id': h[1], 'frame': h[2]}
                for h in self.camera_history
            ],
            'age': round(time.time() - self.created_at, 1)
        }


class CrossCameraReID:
    """
    Cross-camera vehicle re-identification engine.

    Each PHYSICAL vehicle has ONE unique global ID across all cameras.
    When the same vehicle is spotted in a new camera it is matched by:
      1. Deep CNN feature cosine similarity (primary, 75 %)
      2. Color consistency                  (bonus,   15 %)
      3. Vehicle-class consistency          (bonus,   10 %)

    Duplicate IDs that represent the same vehicle are merged automatically.
    """

    # Cosine similarity is mapped to [0,1] via (sim+1)/2.
    # A perfectly identical crop across cameras typically scores ~0.72–0.85
    # after lighting/angle variation, so threshold is set at 0.60 combined.
    FEATURE_THRESHOLD = 0.60     # minimum combined score to accept a match
    FEATURE_ONLY_MIN  = 0.62     # if feature sim alone is this high, accept regardless of color
    MERGE_THRESHOLD   = 0.72     # auto-merge two gallery entries if score >= this

    def __init__(self, similarity_threshold=0.60, color_weight=0.15,
                 type_weight=0.10, feature_weight=0.75):
        self.similarity_threshold = similarity_threshold
        self.color_weight  = color_weight
        self.type_weight   = type_weight
        self.feature_weight = feature_weight

        self.gallery    = VehicleGallery()
        # (camera_id, local_track_id) -> global_id
        self.id_mapping = {}
        # reverse: global_id -> set of (camera_id, local_track_id)
        self.gid_to_keys = defaultdict(set)

        # Vehicle tracking
        self.tracked_vehicle_id       = None
        self.auto_switch_callback     = None
        self.journey_complete_callback = None

        # Stats
        self.match_count     = 0
        self.merge_count     = 0
        self.new_vehicle_count = 0

        print("[CrossCameraReID] Initialized — unique-global-ID mode")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_tracks(self, camera_id, tracks):
        """
        Process all active tracks from one camera frame.

        For each track:
          - If already mapped → update gallery.
          - Else → search gallery for best matching entry across other cameras.
            * Strong match (>= FEATURE_THRESHOLD) → assign existing global ID.
            * No match → mint a new global ID.

        After assigning, run an optional merge pass to unify duplicate IDs
        that represent the same vehicle seen concurrently in multiple cameras.

        Returns dict: local_track_id -> global_id
        """
        results = {}

        # ---- Journey-completion check for tracked vehicle ----
        if self.tracked_vehicle_id is not None:
            tracked_entry = self.gallery.get_entry(self.tracked_vehicle_id)
            if tracked_entry is not None:
                time_since_seen = time.time() - tracked_entry.last_updated
                if time_since_seen > 4.5:
                    gid = self.tracked_vehicle_id
                    print(f"[CrossCameraReID] G{gid} journey ended ({time_since_seen:.1f}s idle). Stopping.")
                    self.stop_tracking()
                    if self.journey_complete_callback:
                        self.journey_complete_callback(gid)

        # ---- Per-track processing ----
        new_assignments = []   # (track, global_id) for merge candidate check

        for track in tracks:
            if track.feature is None or np.linalg.norm(track.feature) == 0:
                continue

            key = (camera_id, track.track_id)

            # Already mapped in this camera
            if key in self.id_mapping:
                global_id = self.id_mapping[key]
                self.gallery.update_entry(
                    global_id, track.feature, camera_id,
                    track.track_id, track.bbox, track.last_seen,
                    track.class_name, track.color_name
                )
                results[track.track_id] = global_id
                self._maybe_fire_switch(camera_id, global_id)
                continue

            # Search gallery for a cross-camera match
            best_match, best_score = self._find_best_match(
                track.feature, track.class_name, track.color_name, camera_id
            )

            if best_match is not None:
                global_id = best_match
                self.id_mapping[key] = global_id
                self.gid_to_keys[global_id].add(key)
                self.gallery.update_entry(
                    global_id, track.feature, camera_id,
                    track.track_id, track.bbox, track.last_seen,
                    track.class_name, track.color_name
                )
                self.match_count += 1
                print(f"[ReID] Matched (cam={camera_id}, L{track.track_id}) → G{global_id} "
                      f"(score={best_score:.3f}, color={track.color_name}, class={track.class_name})")
            else:
                # New vehicle
                global_id = self.gallery.add_entry(
                    track.feature, camera_id, track.track_id,
                    track.class_name, track.color_name,
                    track.bbox, track.last_seen
                )
                self.id_mapping[key] = global_id
                self.gid_to_keys[global_id].add(key)
                self.new_vehicle_count += 1
                print(f"[ReID] New vehicle G{global_id} in cam={camera_id}, L{track.track_id} "
                      f"(color={track.color_name}, class={track.class_name})")

            results[track.track_id] = global_id
            new_assignments.append((track, global_id))
            self._maybe_fire_switch(camera_id, global_id)

        # ---- Merge pass: unify duplicate global IDs ----
        # If two gallery entries score very highly against each other
        # (e.g., same vehicle assigned different global IDs across cameras),
        # merge the newer one into the older one and remap all keys.
        self._run_merge_pass(camera_id)

        # After merges, fix any remapped IDs in results
        for local_id, gid in list(results.items()):
            canonical = self._canonical_gid(gid)
            if canonical != gid:
                results[local_id] = canonical

        return results

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _find_best_match(self, query_feature, query_class, query_color, query_camera):
        """
        Search the gallery for the best matching entry from a DIFFERENT camera.

        Strategy:
         1. Prioritise the currently tracked vehicle (prevents identity switching).
         2. Then search all other recent entries.
         3. Skip entries that are currently active in query_camera (same camera tracker handles them).
        
        Returns: (global_id, score) or (None, 0.0)
        """
        best_gid   = None
        best_score = 0.0

        # 1. Priority check for tracked vehicle
        if self.tracked_vehicle_id is not None:
            tracked_entry = self.gallery.get_entry(self.tracked_vehicle_id)
            if tracked_entry is not None:
                # Only match from other camera; allow if tracked vehicle left that camera (>1.5s)
                active_elsewhere = (
                    tracked_entry.current_camera != query_camera and
                    tracked_entry.is_currently_in_camera(tracked_entry.current_camera, max_gap_sec=1.5)
                )
                if not active_elsewhere:
                    score, feat_sim = self._compute_score(
                        query_feature, query_class, query_color, tracked_entry
                    )
                    if score >= self.FEATURE_THRESHOLD or feat_sim >= self.FEATURE_ONLY_MIN:
                        return self.tracked_vehicle_id, score

        # 2. General gallery search
        for entry in self.gallery.get_recent_entries():
            # Skip self-camera: if this vehicle is currently in query_camera,
            # the single-camera tracker handles continuity via id_mapping.
            if entry.is_currently_in_camera(query_camera, max_gap_sec=2.5):
                continue

            score, feat_sim = self._compute_score(
                query_feature, query_class, query_color, entry
            )

            if (score >= self.similarity_threshold or feat_sim >= self.FEATURE_ONLY_MIN) \
                    and score > best_score:
                best_score = score
                best_gid   = entry.global_id

        return best_gid, best_score

    def _compute_score(self, query_feature, query_class, query_color, entry):
        """
        Compute combined similarity score against a gallery entry.
        Uses max similarity across all stored feature history vectors.
        Returns (combined_score, feature_only_score).
        """
        feats = entry.feature_history if entry.feature_history else [entry.feature]
        feat_sim = max(self._cosine_sim(query_feature, f) for f in feats)

        color_sim = 1.0 if (
            query_color and entry.color_name and
            query_color.lower() == entry.color_name.lower()
        ) else 0.0

        type_sim = 1.0 if (
            query_class and entry.class_name and
            query_class.lower() == entry.class_name.lower()
        ) else 0.0

        combined = (self.feature_weight * feat_sim +
                    self.color_weight  * color_sim +
                    self.type_weight   * type_sim)
        return combined, feat_sim

    def _cosine_sim(self, f1, f2):
        """Cosine similarity mapped to [0, 1]."""
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 0.0
        raw = float(np.dot(f1, f2) / (n1 * n2))
        return float(max(0.0, (raw + 1.0) / 2.0))

    # ------------------------------------------------------------------
    # Merge pass — unify duplicate global IDs
    # ------------------------------------------------------------------

    def _run_merge_pass(self, camera_id):
        """
        Scan recent gallery entries. If two entries that were LAST SEEN in
        DIFFERENT cameras score >= MERGE_THRESHOLD against each other, merge
        the newer one (higher global_id) into the older one (lower global_id).
        Re-maps all id_mapping keys accordingly.
        """
        entries = self.gallery.get_recent_entries(max_age_seconds=20)
        n = len(entries)
        if n < 2:
            return

        to_merge = []   # list of (keep_gid, drop_gid)
        checked  = set()

        for i in range(n):
            for j in range(i + 1, n):
                a, b = entries[i], entries[j]
                if a.global_id == b.global_id:
                    continue
                pair = (min(a.global_id, b.global_id), max(a.global_id, b.global_id))
                if pair in checked:
                    continue
                checked.add(pair)

                # Only consider cross-camera duplicates
                if a.current_camera == b.current_camera:
                    continue

                # Must be same vehicle class
                if a.class_name and b.class_name and a.class_name != b.class_name:
                    continue

                # Feature score between their mean features
                score, feat_sim = self._compute_score(b.feature, b.class_name, b.color_name, a)
                if score >= self.MERGE_THRESHOLD or feat_sim >= 0.75:
                    keep_gid = min(a.global_id, b.global_id)
                    drop_gid = max(a.global_id, b.global_id)
                    to_merge.append((keep_gid, drop_gid))
                    print(f"[ReID] Merge candidate: G{drop_gid} → G{keep_gid} "
                          f"(score={score:.3f}, feat={feat_sim:.3f})")

        for keep_gid, drop_gid in to_merge:
            if drop_gid not in self.gallery.entries:
                continue   # already merged in this pass
            self.gallery.merge_entries(keep_gid, drop_gid)
            self.merge_count += 1
            # Remap all id_mapping entries that pointed to drop_gid
            for k in list(self.id_mapping.keys()):
                if self.id_mapping[k] == drop_gid:
                    self.id_mapping[k] = keep_gid
                    self.gid_to_keys[keep_gid].add(k)
            # Remap gid_to_keys
            if drop_gid in self.gid_to_keys:
                self.gid_to_keys[keep_gid].update(self.gid_to_keys.pop(drop_gid))
            # If tracked vehicle was remapped
            if self.tracked_vehicle_id == drop_gid:
                self.tracked_vehicle_id = keep_gid
                print(f"[ReID] Tracked vehicle remapped G{drop_gid} → G{keep_gid}")

    def _canonical_gid(self, gid):
        """Return current canonical global ID (in case gid was merged away)."""
        # If gid still exists in gallery, it's canonical
        if gid in self.gallery.entries:
            return gid
        # Walk id_mapping to find where this gid was remapped
        for v in self.id_mapping.values():
            if v != gid and gid not in self.gallery.entries:
                break
        # Best effort: return as-is (caller will re-read from results dict)
        return gid

    # ------------------------------------------------------------------
    # Auto-switch helpers
    # ------------------------------------------------------------------

    def _maybe_fire_switch(self, camera_id, global_id):
        """Fire the auto-switch callback if this is the tracked vehicle."""
        if (self.tracked_vehicle_id == global_id and
                self.auto_switch_callback is not None):
            self.auto_switch_callback(camera_id, global_id)

    # ------------------------------------------------------------------
    # Tracking control
    # ------------------------------------------------------------------

    def set_tracked_vehicle(self, global_id, callback=None, journey_complete_callback=None):
        """
        Mark a vehicle for active cross-camera tracking.

        Args:
            global_id: The global ID to follow.
            callback: Called with (camera_id, global_id) each time the vehicle
                      is spotted in a (new) camera.
            journey_complete_callback: Called with (global_id) when the vehicle
                      has been absent from all cameras for > 4.5 s.
        """
        self.tracked_vehicle_id        = global_id
        self.auto_switch_callback      = callback
        self.journey_complete_callback = journey_complete_callback

        entry = self.gallery.get_entry(global_id)
        if entry is not None:
            entry.last_updated = time.time()
            if callback and entry.current_camera:
                callback(entry.current_camera, global_id)
        print(f"[CrossCameraReID] Now tracking vehicle G{global_id}")

    def stop_tracking(self):
        """Stop auto-following the current vehicle."""
        self.tracked_vehicle_id   = None
        self.auto_switch_callback = None
        print("[CrossCameraReID] Stopped tracking")

    def get_tracked_vehicle_info(self):
        if self.tracked_vehicle_id is None:
            return None
        entry = self.gallery.get_entry(self.tracked_vehicle_id)
        return entry.to_dict() if entry else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_vehicles(self):
        """Return all recently-seen unique physical vehicles."""
        return [e.to_dict() for e in self.gallery.get_recent_entries()]

    def get_vehicle_by_camera(self, camera_id):
        vehicles = []
        for entry in self.gallery.get_recent_entries():
            if entry.is_currently_in_camera(camera_id, max_gap_sec=3.0):
                vehicles.append(entry.to_dict())
        return vehicles

    def get_stats(self):
        return {
            'total_vehicles':       len(self.gallery.entries),
            'active_vehicles':      len(self.gallery.get_recent_entries()),
            'cross_camera_matches': self.match_count,
            'merges':               self.merge_count,
            'new_vehicles':         self.new_vehicle_count,
            'tracked_vehicle':      self.tracked_vehicle_id
        }
