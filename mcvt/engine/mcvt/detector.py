"""Vehicle detection with a pretrained YOLOv8 model (FR-VD-01 .. FR-VD-05)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

VEHICLE_CLASS_NAMES = ("car", "motorcycle", "bus", "truck")


@dataclass
class Detection:
    """One detected vehicle: (x1, y1, x2, y2), class label and confidence (FR-VD-04)."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def ltwh(self) -> list[float]:
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class VehicleDetector:
    """Wraps an Ultralytics YOLOv8 model and keeps only vehicle classes."""

    def __init__(self, model_path: str | Path, confidence: float = 0.6, classes=VEHICLE_CLASS_NAMES,
                 imgsz: int = 640, device: str = "auto", min_box_area_ratio: float = 0.0,
                 agnostic_nms: bool = True, nms_iou: float = 0.7):
        from ultralytics import YOLO

        # NMS IoU: of two boxes overlapping by at least this IoU only the more confident one is kept.
        self.nms_iou = float(nms_iou)

        # True = one box per vehicle even when the model scores it as two classes (car + truck);
        # class-wise NMS kept both, which became two tracks for one vehicle.
        self.agnostic_nms = bool(agnostic_nms)
        self.model = YOLO(str(model_path))
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = resolve_device(device)
        self.min_box_area_ratio = float(min_box_area_ratio)
        names = self.model.names  # {index: name} from the weights - not hard-coded
        wanted = {c.lower() for c in classes}
        self.class_ids = [i for i, n in names.items() if n.lower() in wanted]
        self.names = names
        if not self.class_ids:
            raise ValueError(f"None of the classes {sorted(wanted)} exist in {model_path}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect vehicles in a single BGR frame."""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection on several camera frames in one forward pass (FR-VD-05)."""
        if not frames:
            return []
        results = self.model.predict(
            frames,
            conf=self.confidence,
            iou=self.nms_iou,
            classes=self.class_ids,
            imgsz=self.imgsz,
            device=self.device,
            agnostic_nms=self.agnostic_nms,
            verbose=False,
        )
        out: list[list[Detection]] = []
        for frame, res in zip(frames, results):
            h, w = frame.shape[:2]
            min_area = self.min_box_area_ratio * w * h
            dets: list[Detection] = []
            boxes = res.boxes
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                    if k not in self.class_ids or c < self.confidence:
                        continue
                    d = Detection(float(x1), float(y1), float(x2), float(y2), self.names[int(k)], float(c))
                    if d.area >= min_area:
                        dets.append(d)
            out.append(dets)
        return out
