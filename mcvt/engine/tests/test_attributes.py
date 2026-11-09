"""Vehicle colour/type stored with the global id and used to rule out ReID matches."""

from collections import Counter

import cv2
import numpy as np

from mcvt.attributes import AttributeGate, ColorNamer, color_relation
from mcvt.matcher import CrossCameraMatcher, TrackIdentity
from tests.test_handoff import make_pair


def car(bgr, cast=(0, 0, 0)):
    """A flat-coloured 'vehicle' crop, optionally tinted like a camera would."""
    img = np.zeros((80, 120, 3), np.uint8)
    img[:] = np.clip(np.array(bgr, np.int32) + np.array(cast, np.int32), 0, 255).astype(np.uint8)
    return img


def warmed_namer(cam, cast=(0, 0, 0)):
    n = ColorNamer({"cast_min_samples": 3})
    for g in (90, 140, 200, 60):  # neutral vehicles teach the camera's tint
        n.observe(cam, car((g, g, g), cast))
    return n


def test_camera_tint_is_removed_before_naming():
    blue_tint = (22, 0, -8)  # BGR: a camera that makes everything bluish (like C31/C41)
    n = warmed_namer("C41", blue_tint)
    assert n.observe("C41", car((235, 235, 235), blue_tint)) == "white"
    assert n.observe("C41", car((25, 25, 25), blue_tint)) == "black"
    assert n.observe("C41", car((30, 30, 200), blue_tint)) == "red"


def test_no_colour_until_camera_tint_is_known():
    n = ColorNamer({"cast_min_samples": 5})
    assert n.observe("C11", car((230, 230, 230))) is None


def test_colour_relation_ladder():
    assert color_relation("white", "silver") == "near"
    assert color_relation("white", "black") == "conflict"
    assert color_relation("red", "white") == "conflict"
    assert color_relation("grey", None) is None


def test_type_groups_allow_car_truck_but_not_motorcycle():
    g = AttributeGate({"type_min_votes": 1})
    assert g.conflict(None, Counter(car=2), None, Counter(truck=2)) is None
    assert g.conflict(None, Counter(car=2), None, Counter(motorcycle=2)) == "type car vs motorcycle"


def ident(cam, tid, v, colors, types=None, first_ts=None):
    t = TrackIdentity(cam, tid, first_ts=first_ts)
    t.add(v)
    t.colors.update(colors)
    t.types.update(types or {"car": 3.0})
    return t


def test_same_features_but_different_colour_gets_new_id():
    a, b = make_pair(0.80)  # close enough to match on features alone
    m = CrossCameraMatcher(threshold=0.75, gate=AttributeGate({}))
    first = m.matchAcrossCameras(ident("C11", "1", a, {"white": 3}), ts=1000)
    m.release(first.gid, "C11", "1")
    res = m.matchAcrossCameras(ident("C21", "2", b, {"black": 3}), ts=1010)
    assert res.is_new and res.gid != first.gid
    assert m.last_rejections and "colour" in m.last_rejections[0][2]
    assert m.attributes(res.gid)[0] == "black" and m.attributes(first.gid)[0] == "white"


def test_same_features_and_colour_gets_same_id():
    a, b = make_pair(0.80)
    m = CrossCameraMatcher(threshold=0.75, gate=AttributeGate({}))
    first = m.matchAcrossCameras(ident("C11", "1", a, {"white": 3}), ts=1000)
    m.release(first.gid, "C11", "1")
    res = m.matchAcrossCameras(ident("C21", "2", b, {"silver": 2, "white": 1}), ts=1010)
    assert not res.is_new and res.gid == first.gid


def test_stored_vehicle_needs_several_colour_readings():
    a, b = make_pair(0.80)
    m = CrossCameraMatcher(threshold=0.75, gate=AttributeGate({"min_votes": 2, "track_min_votes": 1}))
    first = m.matchAcrossCameras(ident("C11", "1", a, {"white": 1}), ts=1000)  # one reading only
    m.release(first.gid, "C11", "1")
    res = m.matchAcrossCameras(ident("C21", "2", b, {"black": 1}), ts=1010)
    assert res.gid == first.gid


def test_new_track_first_crop_colour_counts():
    a, b = make_pair(0.80)
    m = CrossCameraMatcher(threshold=0.75, gate=AttributeGate({"min_votes": 2, "track_min_votes": 1}))
    first = m.matchAcrossCameras(ident("C11", "1", a, {"white": 3}), ts=1000)
    m.release(first.gid, "C11", "1")
    res = m.matchAcrossCameras(ident("C21", "2", b, {"black": 1}), ts=1010)
    assert res.is_new


def test_attributes_follow_a_merge():
    m = CrossCameraMatcher(threshold=0.75, gate=AttributeGate({}))
    m.add_attributes("VH-001", "white")
    m.add_attributes("VH-002", "white")
    m.merge("VH-002", "VH-001")
    assert "VH-002" not in m.attrs and m.attrs["VH-001"]["colors"]["white"] == 2
