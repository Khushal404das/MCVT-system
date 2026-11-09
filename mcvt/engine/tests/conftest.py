"""Shared fixtures for the MCVT test suite (maps to test_cases.pdf)."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcvt.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def detector(cfg):
    from mcvt.detector import VehicleDetector

    return VehicleDetector(cfg.path("detection.model"), confidence=cfg.get("detection.confidence", 0.6),
                           classes=cfg.get("detection.classes"), imgsz=cfg.get("detection.imgsz", 640), device="auto")


@pytest.fixture(scope="session")
def extractor(cfg):
    from mcvt.detector import resolve_device
    from mcvt.reid.extractor import FeatureExtractor

    return FeatureExtractor(cfg.path("reid.weights"), cfg.get("reid.input_size", [256, 256]), resolve_device("auto"))


def read_frame(cfg, camera_id: str, frame_index: int) -> np.ndarray:
    specs = {s.id: s for s in cfg.camera_specs()}
    if camera_id not in specs:
        pytest.skip(f"camera {camera_id} not available in the configured dataset")
    cap = cv2.VideoCapture(specs[camera_id].source)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        pytest.skip(f"could not read frame {frame_index} of {camera_id}")
    return frame


@pytest.fixture(scope="session")
def frames(cfg):
    """Real frames from the project dataset used by several tests."""
    return {
        "clear_car": read_frame(cfg, "C21", 6000),     # car at the main gate, daylight
        "empty": read_frame(cfg, "C31", 6000),         # walkway and lawn, no vehicles
        "parking_a": read_frame(cfg, "C61", 6000),     # parking lot
        "parking_b": read_frame(cfg, "C61", 12000),    # same parking lot ~10 minutes later
    }
