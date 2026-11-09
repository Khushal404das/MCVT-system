"""Module 3 - Re-Identification / Cross-Camera Matching (TC-RI-01 .. TC-RI-05)."""

import time

import numpy as np
import pytest

from mcvt.db import Database
from mcvt.matcher import CrossCameraMatcher, TrackIdentity


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "test.sqlite")


@pytest.fixture(scope="module")
def crop_detector(cfg, detector):
    """Detector used only to cut test crops: the stronger yolov8s when present,
    so ReID is tested independently of the detector chosen for live use."""
    from mcvt.detector import VehicleDetector

    strong = cfg.path("detection.model").parent / "yolov8s.pt"
    if strong.exists():
        return VehicleDetector(strong, confidence=0.5, imgsz=640)
    return detector


@pytest.fixture(scope="module")
def parked_crops(crop_detector, frames):
    """Crops of parked cars at two different times (same vehicles, ~10 min apart)."""
    detector = crop_detector
    def cars(img):
        dets = sorted((d for d in detector.detect(img) if d.label == "car"), key=lambda d: -d.area)
        return [(d, img[int(d.y1):int(d.y2), int(d.x1):int(d.x2)]) for d in dets]

    def iou(p, q):
        ix = max(0.0, min(p.x2, q.x2) - max(p.x1, q.x1))
        iy = max(0.0, min(p.y2, q.y2) - max(p.y1, q.y1))
        inter = ix * iy
        return inter / (p.area + q.area - inter)

    a, b = cars(frames["parking_a"]), cars(frames["parking_b"])
    # The same parked car = the box that stays in place between the two frames.
    pairs = sorted(((iou(x[0], y[0]), x, y) for x in a for y in b), key=lambda t: -t[0])
    if not pairs or pairs[0][0] < 0.7 or len(a) < 2:
        pytest.skip("no parked car detected in both frames")
    _, car, same = pairs[0]
    other = next(x for x in a if x is not car)
    return car[1], same[1], other[1]


def identity(extractor, cam, tid, crops):
    t = TrackIdentity(cam, tid)
    for f in extractor.extractFeatures(crops):
        t.add(f)
    return t


def test_tc_ri_01_new_vehicle_gets_new_global_id(db, extractor, parked_crops):
    m = CrossCameraMatcher(threshold=0.75, db=db)
    res = m.matchAcrossCameras(identity(extractor, "C1", "1", [parked_crops[0]]))
    m.flush()
    assert res.is_new and res.gid == "VH-001"
    assert db.one("SELECT COUNT(*) n FROM embeddings WHERE gid='VH-001'")["n"] == 1
    assert db.one("SELECT gid FROM vehicles")["gid"] == "VH-001"


def test_tc_ri_02_match_same_vehicle_across_two_cameras(db, extractor, parked_crops):
    m = CrossCameraMatcher(threshold=0.75, db=db)
    first = m.matchAcrossCameras(identity(extractor, "C1", "1", [parked_crops[0]]))
    m.release(first.gid, "C1", "1")
    track2 = identity(extractor, "C2", "7", [parked_crops[1]])
    sim = m.computeCosineSimilarity(track2.query(), m.gallery[first.gid][0])
    second = m.matchAcrossCameras(track2)
    assert sim >= 0.75
    assert second.gid == first.gid and not second.is_new


def test_tc_ri_03_reject_match_for_different_vehicle(db, extractor, parked_crops):
    m = CrossCameraMatcher(threshold=0.75, db=db)
    first = m.matchAcrossCameras(identity(extractor, "C1", "1", [parked_crops[0]]))
    other = identity(extractor, "C2", "3", [parked_crops[2]])
    assert m.computeCosineSimilarity(other.query(), m.gallery[first.gid][0]) < 0.75
    res = m.matchAcrossCameras(other)
    assert res.is_new and res.gid == "VH-002"
    assert db.one("SELECT COUNT(*) n FROM vehicles")["n"] == 2


def test_tc_ri_04_reidentify_after_occlusion(db, extractor, parked_crops):
    m = CrossCameraMatcher(threshold=0.75, db=db)
    first = m.matchAcrossCameras(identity(extractor, "C1", "1", [parked_crops[0]]))
    m.release(first.gid, "C1", "1")  # track lost behind an obstacle
    again = m.matchAcrossCameras(identity(extractor, "C1", "9", [parked_crops[1]]))  # new local track, same camera
    assert again.gid == first.gid and not again.is_new and again.similarity >= 0.75


def test_same_camera_two_tracks_never_share_an_id(db, extractor, parked_crops):
    """NFR-R-03 guard: two simultaneous tracks in one camera are two vehicles."""
    m = CrossCameraMatcher(threshold=0.75, db=db)
    a = m.matchAcrossCameras(identity(extractor, "C1", "1", [parked_crops[0]]))
    b = m.matchAcrossCameras(identity(extractor, "C1", "2", [parked_crops[1]]))
    assert a.gid != b.gid


def test_tc_ri_05_feature_extraction_performance(extractor, parked_crops):
    import cv2

    crop = cv2.resize(parked_crops[0], (128, 256))
    extractor.extractFeatures([crop])  # warm-up
    times, feats = [], []
    for _ in range(100):
        t = time.perf_counter()
        feats.append(extractor.extractFeatures([crop])[0])
        times.append(time.perf_counter() - t)
    avg_ms = 1000 * sum(times) / len(times)
    print(f"\nTC-RI-05 average extraction time: {avg_ms:.1f} ms/crop on {extractor.device}")
    assert all(f.shape == (2048,) and f.dtype == np.float32 and np.isfinite(f).all() for f in feats)
    if extractor.device.type == "cuda":
        assert avg_ms <= 30.0
    else:
        pytest.xfail(f"30 ms target needs a GPU (SRS 2.6); CPU-only measured {avg_ms:.0f} ms/crop")
