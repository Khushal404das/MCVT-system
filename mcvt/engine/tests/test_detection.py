"""Module 1 - Vehicle Detection (TC-VD-01 .. TC-VD-05)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

VEHICLES = {"car", "bus", "truck", "motorcycle"}


def test_tc_vd_01_detect_vehicle_in_clear_frame(detector, frames):
    dets = detector.detect(frames["clear_car"])
    cars = [d for d in dets if d.label == "car"]
    assert cars, "no car detected"
    best = max(cars, key=lambda d: d.confidence)
    assert best.confidence >= 0.80
    assert best.x2 > best.x1 and best.y2 > best.y1  # FR-VD-04 bounding box


def test_tc_vd_02_no_detection_on_empty_frame(detector, frames):
    assert detector.detect(frames["empty"]) == []


def test_tc_vd_03_detect_multiple_vehicles(detector, frames):
    # A 2x2 mosaic of the gate scene = four separate vehicles in one frame.
    img = frames["clear_car"]
    mosaic = np.vstack([np.hstack([img, img]), np.hstack([img, img])])
    mosaic = cv2.resize(mosaic, (img.shape[1], img.shape[0]))
    dets = [d for d in detector.detect(mosaic) if d.label in VEHICLES and d.confidence >= 0.75]
    assert len(dets) >= 4
    # Every box is distinct (no two boxes overlap by more than 30 % IoU).
    for i, a in enumerate(dets):
        for b in dets[i + 1:]:
            ix = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
            iy = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
            inter = ix * iy
            assert inter / (a.area + b.area - inter) < 0.3


def test_tc_vd_04_detection_under_low_light(detector, frames):
    # The dataset has no night footage: simulate night by darkening + adding sensor noise.
    img = frames["clear_car"].astype(np.float32) / 255.0
    night = np.clip((img ** 1.8) * 0.45 * 255 + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    assert night.mean() < frames["clear_car"].mean() * 0.5
    dets = [d for d in detector.detect(night) if d.label in VEHICLES]
    assert dets and max(d.confidence for d in dets) >= 0.60


def test_tc_vd_05_filter_non_vehicle_objects(detector):
    import ultralytics

    bus = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"  # 4 pedestrians + 1 bus
    if not bus.exists():
        pytest.skip("ultralytics sample image not available")
    dets = detector.detect(cv2.imread(str(bus)))
    assert dets, "the vehicle in the frame was not detected"
    assert {d.label for d in dets} <= VEHICLES  # pedestrians are excluded (FR-VD-02)
