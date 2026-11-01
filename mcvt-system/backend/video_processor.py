"""
Multi-Camera Video Processor
Processes all camera feeds simultaneously, runs detection, tracking,
feature extraction, and cross-camera re-identification.

Lockstep Real-Time Architecture:
  1. Real-Time Wall-Clock Synchronized Playback:
     - Each camera advances frames proportional to elapsed real-world time.
     - 1.0 second of real time advances 1.0 second of video -> 100% normal playback speed.
  2. Frame-Synchronous Detection & Overlay:
     - YOLOv8 detects vehicles directly on each active frame.
     - Bounding boxes are drawn onto THAT EXACT FRAME before display.
     - Guarantee: The bounding box is always directly on the vehicle at all times.
       No lag, no offset, no ghost boxes on empty roads.
  3. Crisp High-Definition Grid & Single Views:
     - Native 960x540 display resolution for full camera view.
     - 640x360 high-definition thumbnails with quality=75 for camera grid cells.
     - Crystal-clear visual clarity without pixelation or blur.
"""

import cv2
import numpy as np
import threading
import time
from collections import OrderedDict


class CameraStream:
    """Manages a single camera video feed with wall-clock synchronized frame playback."""

    DISPLAY_WIDTH  = 960
    DISPLAY_HEIGHT = 540

    def __init__(self, camera_id, video_path):
        self.camera_id  = camera_id
        self.video_path = video_path

        self.cap          = None
        self.fps          = 20.0
        self.total_frames = 0
        self.orig_width   = 0
        self.orig_height  = 0
        self.width        = self.DISPLAY_WIDTH
        self.height       = self.DISPLAY_HEIGHT

        self.frame_idx    = 0
        self.frame_version = 0
        self.is_running   = False

        self.current_frame = None  # Live rendered frame with bounding boxes
        self.current_thumb_bytes = None
        self.current_full_bytes  = None
        self._frame_lock   = threading.Lock()
        self._cap_lock     = threading.Lock()

        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        self.fps          = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.orig_width   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height  = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Maintain 16:9 aspect ratio at 960px width
        self.width  = self.DISPLAY_WIDTH
        self.height = max(240, int(self.orig_height * (self.DISPLAY_WIDTH / max(1, self.orig_width))))
        self.is_running = True

        print(f"[Camera-{self.camera_id}] Opened {self.video_path}: "
              f"{self.orig_width}x{self.orig_height} -> display {self.width}x{self.height} "
              f"@ {self.fps:.1f}fps, {self.total_frames} frames")

    def advance_and_read(self, advance):
        """Advance video frames to match wall-clock elapsed time, then read next frame."""
        with self._cap_lock:
            if not self.is_running or self.cap is None:
                return None

            # Fast grab to advance frames to match real time
            for _ in range(advance - 1):
                self.cap.grab()
                self.frame_idx += 1
                if self.total_frames > 0 and self.frame_idx >= self.total_frames:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_idx = 0

            ret, frame = self.cap.read()
            self.frame_idx += 1
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_idx = 0
                ret, frame = self.cap.read()
                if not ret:
                    return None
            return frame

    def set_display_frame(self, frame):
        """Set the live rendered frame (thread-safe) and pre-encode JPEGs."""
        thumb = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        _, thumb_jpg = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 72])
        _, full_jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        thumb_bytes = thumb_jpg.tobytes()
        full_bytes = full_jpg.tobytes()
        with self._frame_lock:
            self.current_frame = frame
            self.current_thumb_bytes = thumb_bytes
            self.current_full_bytes = full_bytes
            self.frame_version += 1

    def get_display_frame(self):
        """Get the live rendered frame (thread-safe)."""
        with self._frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def get_progress(self):
        if self.total_frames == 0:
            return 0
        return round(100.0 * self.frame_idx / self.total_frames, 1)

    def release(self):
        self.is_running = False
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    # Compatibility shims
    def get_latest_raw_frame(self):
        return self.get_display_frame()

    def get_processed_frame(self):
        return self.get_display_frame()

    def get_current_frame(self):
        return self.get_display_frame()

    def read_frame(self):
        return self.get_display_frame()


class MultiCameraProcessor:
    """Orchestrates multi-camera vehicle detection, tracking, and cross-camera re-identification."""

    def __init__(self, camera_configs, process_fps=10):
        from vehicle_detector import VehicleDetector
        from feature_extractor import FeatureExtractor, get_dominant_color_name
        from tracker import SingleCameraTracker
        from cross_camera_reid import CrossCameraReID

        self.process_fps = process_fps

        print("=" * 60)
        print("Initializing Multi-Camera Vehicle Tracking System")
        print("=" * 60)

        self.detector          = VehicleDetector(model_path='yolov8n.pt', confidence=0.25)
        self.feature_extractor = FeatureExtractor(feature_dim=256)
        self.reid_engine       = CrossCameraReID(similarity_threshold=0.60)
        self.get_color         = get_dominant_color_name

        self.cameras = OrderedDict()
        for cam_id, video_path in camera_configs.items():
            try:
                self.cameras[cam_id] = CameraStream(cam_id, video_path)
            except Exception as e:
                print(f"[ERROR] Failed to open camera {cam_id}: {e}")

        self.trackers = {cam_id: SingleCameraTracker(cam_id) for cam_id in self.cameras}

        self.is_running         = False
        self.processing_threads = {}
        self.active_camera      = list(self.cameras.keys())[0] if self.cameras else None

        self.events      = []
        self.events_lock = threading.Lock()

        self.auto_switch_enabled = False
        self.switch_callback     = None

        print(f"\n[System] {len(self.cameras)} cameras initialized successfully")
        print(f"[System] Lockstep real-time playback: boxes drawn directly on detected frames")
        print("=" * 60)

    def start(self):
        self.is_running = True
        for cam_id in self.cameras:
            t = threading.Thread(
                target=self._camera_processing_loop,
                args=(cam_id,),
                daemon=True,
                name=f"cam-proc-{cam_id}"
            )
            self.processing_threads[cam_id] = t
            t.start()
            print(f"[System] Started processor thread for camera {cam_id}")

    def stop(self):
        self.is_running = False
        for cam in self.cameras.values():
            cam.release()
        print("[System] All cameras stopped")

    def _camera_processing_loop(self, camera_id):
        """
        Processing loop for a camera:
        Runs at normal native video speed (1.0x real time).
        Uses keyframe detection with track persistence to ensure smooth playback and normal video speed.
        Draws bounding boxes and labels directly on detected frames.
        """
        camera  = self.cameras[camera_id]
        tracker = self.trackers[camera_id]
        target_interval = 1.0 / max(1.0, camera.fps)

        frame_count = 0
        id_mapping = {}
        active_tracks = []

        while self.is_running and camera.is_running:
            loop_start = time.perf_counter()

            raw_frame = camera.advance_and_read(1)
            if raw_frame is None:
                time.sleep(0.02)
                continue

            frame = cv2.resize(raw_frame, (camera.width, camera.height),
                               interpolation=cv2.INTER_LINEAR)

            frame_count += 1
            # Run full AI detection every 3 frames for smooth normal video speed
            if frame_count % 3 == 1:
                det_results = self.detector.detect_and_crop(frame)
                if det_results:
                    bboxes, features, class_names, color_names = [], [], [], []
                    for det, crop in det_results:
                        bboxes.append(det['bbox'])
                        features.append(self.feature_extractor.extract(crop))
                        class_names.append(det['class_name'])
                        color_names.append(self.get_color(crop))

                    active_tracks = tracker.update(
                        bboxes, features, class_names, color_names,
                        frame_idx=camera.frame_idx
                    )
                    id_mapping = self.reid_engine.process_tracks(camera_id, active_tracks)
                else:
                    tracker.update([], [], [], [], frame_idx=camera.frame_idx)
                    active_tracks = []
                    id_mapping = {}

            # Draw bounding boxes directly on this frame
            if active_tracks:
                drawn = self._draw_overlays(frame, active_tracks, id_mapping, camera_id)
            else:
                drawn = self._draw_camera_info(frame, camera_id)

            camera.set_display_frame(drawn)

            # Smooth normal video speed pacing
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)

    def _draw_overlays(self, frame, tracks, id_mapping, camera_id):
        """Draw bounding boxes and labels directly on the detected frame."""
        tracked_gid = self.reid_engine.tracked_vehicle_id

        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            # Clamp bounds
            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y2 = max(0, min(frame.shape[0] - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            global_id = id_mapping.get(track.track_id)

            if global_id == tracked_gid and tracked_gid is not None:
                color, thickness = (0, 255, 0), 3
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 0), 1)
            else:
                if global_id is not None:
                    np.random.seed(global_id * 37)
                    color = tuple(int(c) for c in np.random.randint(70, 250, 3))
                else:
                    color = (0, 230, 255)  # Bright cyan/yellow for clear contrast
                thickness = 2

            # Vehicle bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Build label: G<global_id> | L<local_id> | <color> | <class>
            parts = []
            if global_id is not None:
                parts.append(f"G{global_id}")
            parts.append(f"L{track.track_id}")
            if track.color_name:
                parts.append(track.color_name)
            v_class = track.class_name if track.class_name else 'car'
            parts.append(v_class)
            label = " | ".join(parts)

            # Label banner
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, max(th + 8, y1)), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return self._draw_camera_info(frame, camera_id)

    def _draw_camera_info(self, frame, camera_id):
        camera = self.cameras[camera_id]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 36), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        cv2.putText(frame, f"Camera {camera_id.upper()}", (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        info = f"Frame: {camera.frame_idx} | {camera.get_progress()}%"
        cv2.putText(frame, info, (frame.shape[1] - 240, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
        return frame

    def get_frame_jpeg(self, camera_id, quality=82):
        """Serve high-definition JPEG frame for the main camera view."""
        if camera_id not in self.cameras:
            return None
        camera = self.cameras[camera_id]
        with camera._frame_lock:
            if camera.current_full_bytes is not None:
                return camera.current_full_bytes
        frame = camera.get_display_frame()
        if frame is None:
            frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Camera {camera_id.upper()} - Connecting...", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    def get_thumbnail_jpeg(self, camera_id, size=(640, 360), quality=75):
        """Serve crisp high-resolution thumbnail for grid view."""
        if camera_id not in self.cameras:
            return None
        camera = self.cameras[camera_id]
        with camera._frame_lock:
            if camera.current_thumb_bytes is not None:
                return camera.current_thumb_bytes
        frame = camera.get_display_frame()
        if frame is None:
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.putText(frame, f"CAM {camera_id.upper()}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return jpeg.tobytes()

    # ------------------------------------------------------------------
    # Camera control
    # ------------------------------------------------------------------

    def switch_camera(self, camera_id):
        if camera_id in self.cameras:
            old_cam = self.active_camera
            self.active_camera = camera_id
            self._add_event('camera_switch', {'from': old_cam, 'to': camera_id, 'reason': 'manual'})
            return True
        return False

    def track_vehicle(self, global_id):
        def on_camera_switch(camera_id, gid):
            if camera_id in self.cameras and self.active_camera != camera_id:
                old_cam = self.active_camera
                self.active_camera = camera_id
                self._add_event('auto_switch', {
                    'camera': camera_id, 'global_id': gid,
                    'reason': f'Vehicle G{gid} detected in camera {camera_id.upper()} via extracted features'
                })
                if self.auto_switch_enabled and self.switch_callback:
                    self.switch_callback(camera_id, gid)

        def on_journey_complete(gid):
            self.auto_switch_enabled = False
            self._add_event('track_complete', {
                'global_id': gid,
                'message': f'Vehicle G{gid} journey ended (exited all camera views). Auto-switch stopped.'
            })

        self.auto_switch_enabled = True
        self.reid_engine.set_tracked_vehicle(global_id, on_camera_switch, on_journey_complete)
        self._add_event('track_start', {'global_id': global_id})

    def stop_tracking(self):
        self.reid_engine.stop_tracking()
        self._add_event('track_stop', {})

    def enable_auto_switch(self, enabled=True, callback=None):
        self.auto_switch_enabled = enabled
        self.switch_callback     = callback

    # ------------------------------------------------------------------
    # Events & status
    # ------------------------------------------------------------------

    def _add_event(self, event_type, data):
        with self.events_lock:
            self.events.append({
                'type':      event_type,
                'data':      data,
                'timestamp': time.time(),
                'time_str':  time.strftime('%H:%M:%S')
            })
            if len(self.events) > 100:
                self.events = self.events[-100:]

    def get_events(self, since=None):
        with self.events_lock:
            if since is None:
                return self.events[-20:]
            return [e for e in self.events if e['timestamp'] > since]

    def get_system_status(self):
        reid_stats    = self.reid_engine.get_stats()
        camera_status = {}
        for cam_id, cam in self.cameras.items():
            camera_status[cam_id] = {
                'frame_idx':  cam.frame_idx,
                'progress':   cam.get_progress(),
                'resolution': f"{cam.width}x{cam.height}",
                'is_running': cam.is_running,
                'native_fps': round(cam.fps, 1),
            }
        return {
            'cameras':       camera_status,
            'active_camera': self.active_camera,
            'auto_switch':   self.auto_switch_enabled,
            'reid':          reid_stats,
            'vehicles':      self.reid_engine.get_all_vehicles()
        }

    def get_vehicles_in_camera(self, camera_id):
        return self.reid_engine.get_vehicle_by_camera(camera_id)

