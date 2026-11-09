"""Camera hand-off: time-aware matching thresholds and learned camera transitions."""

import numpy as np

from mcvt.matcher import CrossCameraMatcher, TrackIdentity


def vec(seed, base=None, mix=0.0):
    v = np.random.default_rng(seed).normal(size=64).astype(np.float32)
    if base is not None:
        v = mix * base + (1 - mix) * v
    return v / np.linalg.norm(v)


def track(cam, tid, v):
    t = TrackIdentity(cam, tid)
    t.add(v)
    return t


def make_pair(target_sim):
    """Two unit vectors with cosine similarity ~= target_sim."""
    a = vec(1)
    b = vec(2)
    b = b - (b @ a) * a
    b /= np.linalg.norm(b)
    return a, (target_sim * a + np.sqrt(1 - target_sim ** 2) * b).astype(np.float32)


def matcher():
    return CrossCameraMatcher(threshold=0.75, handoff_threshold=0.60, follow_threshold=0.55, handoff_window=90)


def test_vehicle_that_left_matches_in_next_camera_at_lower_threshold():
    a, b = make_pair(0.65)  # same car seen from a different angle
    m = matcher()
    first = m.matchAcrossCameras(track("C11", "1", a), ts=1000)
    m.touch(first.gid, "C11", 1000)
    m.release(first.gid, "C11", "1")
    res = m.matchAcrossCameras(track("C21", "5", b), ts=1012)
    assert res.gid == first.gid and not res.is_new
    assert m.expected_next("C11")[0]["camera"] == "C21"  # transition learned


def test_no_lowered_threshold_while_vehicle_still_visible_or_after_window():
    a, b = make_pair(0.65)
    m = matcher()
    first = m.matchAcrossCameras(track("C11", "1", a), ts=1000)
    m.touch(first.gid, "C11", 1000)
    # still visible on C11 -> strict threshold -> a new id
    assert m.matchAcrossCameras(track("C21", "5", b), ts=1005).is_new
    m2 = matcher()
    g = m2.matchAcrossCameras(track("C11", "1", a), ts=1000).gid
    m2.touch(g, "C11", 1000)
    m2.release(g, "C11", "1")
    assert m2.matchAcrossCameras(track("C21", "5", b), ts=1200).is_new  # outside the 90 s window


def test_followed_vehicle_uses_follow_threshold():
    a, b = make_pair(0.57)
    m = matcher()
    g = m.matchAcrossCameras(track("C11", "1", a), ts=1000).gid
    m.touch(g, "C11", 1000)
    m.release(g, "C11", "1")
    m.followed = {g}
    assert m.matchAcrossCameras(track("C41", "3", b), ts=1030).gid == g


def test_new_id_is_merged_into_vehicle_that_just_left():
    a, b = make_pair(0.66)
    m = matcher()
    g = m.matchAcrossCameras(track("C11", "1", a), ts=1000).gid
    m.touch(g, "C11", 1000)
    # the new track got its own id before the first one was released
    t2 = track("C21", "8", b)
    new = m.matchAcrossCameras(t2, ts=1001)
    assert new.is_new
    m.release(g, "C11", "1")
    import time as _t
    m.last_seen[g] = (_t.time() - 5, "C11")
    found = m.find_merge(t2)
    assert found and found[0] == g
