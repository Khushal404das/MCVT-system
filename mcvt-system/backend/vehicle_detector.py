"""
Vehicle Detection Module using YOLOv8
Detects vehicles (car, truck, bus, motorcycle) in camera frames
"""

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    YOLO = None


class VehicleDetector:
    """YOLOv8-based vehicle detector with adaptive background subtraction fallback."""

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, model_path='yolov8n.pt', confidence=0.2, device='cpu'):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Minimum confidence threshold
            device: 'cpu' or 'cuda'
        """
        self.confidence = confidence
        self.device = device
        self.model = None
        self.bg_subtractor = None
        
        if HAS_YOLO:
            try:
                self.model = YOLO(model_path)
                print(f"[VehicleDetector] Loaded YOLOv8 ({model_path}) on {device}")
            except Exception as e:
                print(f"[VehicleDetector] Warning: Could not initialize YOLOv8: {e}")
                self.model = None

        if self.model is None:
            print("[VehicleDetector] Initializing OpenCV motion-based vehicle detector fallback...")
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)

    def detect(self, frame):
        """
        Detect vehicles in a frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            list of dicts: [{
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str,
                'center': (cx, cy)
            }, ...]
        """
        if frame is None or frame.size == 0:
            return []

        if self.model is not None:
            return self._detect_yolo(frame)
        else:
            return self._detect_fallback(frame)

    def _detect_yolo(self, frame):
        """Detect using YOLOv8."""
        try:
            results = self.model(frame, imgsz=480, conf=self.confidence, device=self.device, verbose=False)
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id not in self.VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])

                    w = max(1, x2 - x1)
                    h = max(1, y2 - y1)
                    ar = w / float(h)
                    area = w * h

                    cls_name = self.VEHICLE_CLASSES[cls_id]

                    # Sanity check 1: Motorcycles are narrow and tall; cars are wide
                    if cls_name == 'motorcycle':
                        if ar > 1.1 or w > 60 or area > 3000:
                            cls_name = 'car'
                            cls_id = 2

                    # Sanity check 2: Normal passenger cars, hatchbacks, and SUVs are not trucks
                    elif cls_name in ['truck', 'bus']:
                        if area < 35000 or (0.75 <= ar <= 2.5 and h < 260):
                            cls_name = 'car'
                            cls_id = 2

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': round(conf, 3),
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'center': (cx, cy)
                    })

            # Class-agnostic NMS to eliminate duplicate boxes for the same vehicle
            return self._apply_nms(detections, iou_thresh=0.45)
        except Exception as e:
            print(f"[VehicleDetector] Error in YOLO detection: {e}")
            return self._detect_fallback(frame)


    @staticmethod
    def _apply_nms(detections, iou_thresh=0.45):
        """Class-agnostic Non-Maximum Suppression."""
        if len(detections) <= 1:
            return detections
        dets = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
        kept = []
        while dets:
            best = dets.pop(0)
            kept.append(best)
            b_box = best['bbox']
            remaining = []
            for d in dets:
                d_box = d['bbox']
                x1 = max(b_box[0], d_box[0])
                y1 = max(b_box[1], d_box[1])
                x2 = min(b_box[2], d_box[2])
                y2 = min(b_box[3], d_box[3])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                a2 = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
                iou = inter / max(a1 + a2 - inter, 1e-6)
                if iou < iou_thresh:
                    remaining.append(d)
            dets = remaining
        return kept

    def _detect_fallback(self, frame):
        """Fallback detector using OpenCV MOG2 and contour filtering."""
        if self.bg_subtractor is None:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=16, detectShadows=True)
            
        fg_mask = self.bg_subtractor.apply(frame)
        # Remove shadows (gray value 127) and keep only foreground (255)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        h, w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter contours by size typical for vehicles
            if 400 < area < (w * h * 0.45):
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect_ratio = cw / float(ch)
                if 0.5 < aspect_ratio < 4.0:
                    cx = x + cw // 2
                    cy = y + ch // 2
                    detections.append({
                        'bbox': [int(x), int(y), int(x + cw), int(y + ch)],
                        'confidence': round(min(0.95, float(area) / 10000.0 + 0.5), 2),
                        'class_id': 2,
                        'class_name': 'car',
                        'center': (cx, cy)
                    })

        return detections[:15]  # Limit to top detections

    def detect_and_crop(self, frame):
        """
        Detect vehicles and return cropped images along with detections.
        
        Returns:
            list of (detection_dict, cropped_image) tuples
        """
        detections = self.detect(frame)
        results = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Ensure bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue  # Skip tiny detections

            crop = frame[y1:y2, x1:x2].copy()
            results.append((det, crop))

        return results
