"""Module 6: Configuration, shared clock & single-camera tracking (TC-CF / TC-TR)."""

from __future__ import annotations

import numpy as np
import pytest

from mcvt.config import clock_seconds, format_clock
from mcvt.detector import Detection
from mcvt.tracker import CameraTracker


# ------------------------------------------------------------ configuration
def test_tc_cf_01_all_thresholds_come_from_config(cfg):
    """TC-CF-01: every key threshold is set in config.yaml (no hard coding)."""
    for key in ("detection.model", "detection.confidence", "reid.weights", "reid.similarity_threshold"):
        assert cfg.get(key) is not None, f"{key} missing from config.yaml"
    assert 0 < float(cfg.get("detection.confidence")) < 1
    assert 0 < float(cfg.get("reid.similarity_threshold")) < 1


def test_tc_cf_02_six_cameras_configured(cfg):
    """TC-CF-02: the six campus cameras C11..C61 are found and have a video source."""
    ids = {s.id for s in cfg.camera_specs()}
    assert {"C11", "C21", "C31", "C41", "C51", "C61"} <= ids
    for s in cfg.camera_specs():
        assert s.source


def test_tc_cf_03_shared_clock_conversion():
    """TC-CF-03: the shared camera clock converts both ways (08:00:00 <-> 28800 s)."""
    assert clock_seconds("08:00:00") == 28800
    assert clock_seconds("08:07:28") - clock_seconds("08:00:00") == 448   # C41/C51 offset, 7 min 28 s
    assert format_clock(28800 + 123.9) == "08:02:03"
    assert format_clock(clock_seconds("23:59:59") + 2) == "00:00:01"      # wraps at midnight


# ----------------------------------------------------------------- tracking
def _det(x, y=100.0, w=120.0, h=80.0):
    return Detection(x, y, x + w, y + h, "car", 0.9)


def _emb(seed):
    v = np.random.default_rng(seed).standard_normal(128).astype(np.float32)
    return v / np.linalg.norm(v)


FRAME = (720, 1280, 3)


def test_tc_tr_01_moving_vehicle_keeps_its_track_id():
    """TC-TR-01: a car moving 10 px per frame keeps one local track ID."""
    tr = CameraTracker("C11", max_age=10, n_init=1)
    e = _emb(1)
    ids = set()
    for i in range(20):
        tracks, _ = tr.update([_det(100 + 10 * i)], [e], FRAME)
        ids |= {t.track_id for t in tracks}
    assert len(ids) == 1


def test_tc_tr_02_two_vehicles_get_different_ids():
    """TC-TR-02: two cars in the same frame get two different local IDs."""
    tr = CameraTracker("C11", max_age=10, n_init=1)
    tracks = []
    for i in range(5):
        tracks, _ = tr.update([_det(100 + 5 * i), _det(800 - 5 * i, y=400)], [_emb(1), _emb(2)], FRAME)
    assert len({t.track_id for t in tracks}) == 2


def test_tc_tr_03_short_occlusion_is_predicted():
    """TC-TR-03: when a car is hidden for 2 frames the Kalman filter predicts its box."""
    tr = CameraTracker("C11", max_age=10, n_init=1)
    e = _emb(1)
    for i in range(8):
        tr.update([_det(100 + 10 * i)], [e], FRAME)
    tracks, _ = tr.update([], [], FRAME)
    assert len(tracks) == 1 and tracks[0].detected is False


def test_tc_tr_04_lost_vehicle_is_removed_after_max_age():
    """TC-TR-04: a car missing for more than max_age frames is reported as removed."""
    tr = CameraTracker("C11", max_age=3, n_init=1)
    e = _emb(1)
    for i in range(5):
        tr.update([_det(100 + 10 * i)], [e], FRAME)
    removed = set()
    for _ in range(6):
        _, gone = tr.update([], [], FRAME)
        removed |= gone
    assert len(removed) == 1


def test_tc_tr_05_detection_box_helpers():
    """TC-TR-05: box area and (left, top, width, height) are computed correctly."""
    d = Detection(10, 20, 110, 70, "car", 0.8)
    assert d.area == 5000
    assert d.ltwh() == [10, 20, 100, 50]
    assert Detection(50, 50, 40, 40, "car", 0.8).area == 0     # inverted box -> no area
