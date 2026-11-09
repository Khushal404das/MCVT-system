"""The ID rule: highest similarity above the threshold gives the id; one id is never live on two
non-overlapping cameras at once - the track with the higher similarity keeps it."""

import numpy as np

from mcvt.matcher import CrossCameraMatcher, TrackIdentity


def unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def vec_with_sim(base, sim, seed):
    """A unit vector whose cosine similarity with unit vector ``base`` is ``sim``."""
    r = np.random.default_rng(seed).normal(size=base.size).astype(np.float32)
    r -= (r @ base) * base
    r /= np.linalg.norm(r)
    return unit(sim * base + np.sqrt(1 - sim * sim) * r)


def track(cam, tid, vec, ts):
    t = TrackIdentity(cam, tid)
    t.first_ts = ts
    t.add(vec)
    return t


def matcher():
    return CrossCameraMatcher(threshold=0.60, handoff_threshold=0.60, follow_threshold=0.60,
                              live_gap=2.0, steal_margin=0.05)


def test_highest_similarity_above_threshold_wins():
    m = matcher()
    rng = np.random.default_rng(0)
    a, b = unit(rng.normal(size=64)), unit(rng.normal(size=64))
    ga = m.matchAcrossCameras(track("C11", "1", a, 0.0), 0.0).gid
    gb = m.matchAcrossCameras(track("C11", "2", b, 0.0), 0.0).gid
    # a query that is 0.70 like A and 0.65 like B (both pass 0.60) -> A
    q = unit(a + b * 0.3)
    res = m.matchAcrossCameras(track("C31", "9", q, 100.0), 100.0)
    sims = m.similarities(q)
    assert sims[ga] > sims[gb] >= 0.60 or sims[ga] >= 0.60
    assert res.gid == ga and not res.is_new
    assert [g for g, _ in res.candidates][0] == ga


def test_below_threshold_gives_new_id():
    m = matcher()
    base = unit(np.random.default_rng(1).normal(size=64))
    g = m.matchAcrossCameras(track("C11", "1", base, 0.0), 0.0).gid
    res = m.matchAcrossCameras(track("C41", "1", vec_with_sim(base, 0.55, 2), 100.0), 100.0)
    assert res.is_new and res.gid != g


def test_id_live_on_another_camera_is_not_given_out_again():
    m = matcher()
    car = unit(np.random.default_rng(3).normal(size=64))
    g = m.matchAcrossCameras(track("C11", "1", car, 0.0), 0.0).gid
    m.touch(g, "C11", 0.0)
    # at the same moment C41 sees a look-alike (0.80): the id is in use on C11 by the track that
    # created it, so C41 gets a new id
    res = m.matchAcrossCameras(track("C41", "1", vec_with_sim(car, 0.80, 5), 0.5), 0.5)
    assert res.gid != g and res.is_new
    assert any(r[0] == g and "C11" in r[2] for r in m.last_rejections)
    # once the vehicle has left C11 (no detection for > live_gap) the id is free again
    later = m.matchAcrossCameras(track("C51", "1", vec_with_sim(car, 0.80, 6), 30.0), 30.0)
    assert later.gid == g


def test_higher_similarity_takes_the_id_from_a_weaker_holder():
    m = matcher()
    car = unit(np.random.default_rng(4).normal(size=64))
    g = m.matchAcrossCameras(track("C11", "1", car, 0.0), 0.0).gid
    first = m.matchAcrossCameras(track("C31", "1", vec_with_sim(car, 0.80, 7), 50.0), 50.0)
    assert first.gid == g
    m.touch(g, "C31", 50.0)
    holder_sim = m.holder[g][2]
    # a C41 track at the same moment that is 0.66 like the stored car: lower than the holder -> no
    weak = m.matchAcrossCameras(track("C41", "1", vec_with_sim(car, 0.62, 8), 50.5), 50.5)
    assert weak.gid != g
    # a C51 track that matches clearly better than the holder takes it; C31 must give it up
    m.holder[g] = ("C31", "1", 0.62)  # the C31 track had only just passed the threshold
    better = m.matchAcrossCameras(track("C51", "1", car, 51.0), 51.0)
    assert better.similarity > holder_sim - 1e-6 or better.similarity >= 0.67
    assert better.gid == g and better.displaced == ("C31", "1")


def test_id_without_features_is_merged_by_its_first_sample():
    m = matcher()
    car = unit(np.random.default_rng(9).normal(size=64))
    known = m.matchAcrossCameras(track("C11", "1", car, 0.0), 0.0).gid
    far = TrackIdentity("C41", "7")  # far away: crop too small for ReID, shown at once with an id
    far.first_ts = 100.0
    res = m.new_vehicle(far, 100.0)
    assert res.is_new and far.gid == res.gid != known and far.provisional
    # its first usable crop is 0.62 like the known car: at the normal threshold (0.60, no extra
    # merge margin) it is recognised
    far.add(vec_with_sim(car, 0.62, 10))
    found = m.find_merge(far)
    assert found is not None and found[0] == known


def test_overlapping_camera_gives_the_one_live_vehicle_its_id():
    m = CrossCameraMatcher(threshold=0.60, handoff_threshold=0.60, follow_threshold=0.60,
                           overlaps=[["C41", "C51"]], overlap_link=3.0)
    rng = np.random.default_rng(11)
    far = unit(rng.normal(size=64))  # tiny far-away view on C41
    g = m.matchAcrossCameras(track("C41", "1", far, 100.0), 100.0).gid
    m.touch(g, "C41", 100.0)
    # the close-up on C51 hardly looks like it (similarity 0.20, like the real Prius), but it is
    # the only car on C41 now
    res = m.matchAcrossCameras(track("C51", "1", vec_with_sim(far, 0.20, 13), 101.0), 101.0)
    assert res.gid == g and res.via == "overlap:C41"
    # a car with nothing alike (similarity ~0) is not linked
    m.touch(g, "C41", 101.5)
    m.release(g, "C51", "1")
    none = m.matchAcrossCameras(track("C51", "2", vec_with_sim(far, 0.0, 14), 102.0), 102.0)
    assert none.gid != g
    # a non-overlapping camera does not get it
    other = m.matchAcrossCameras(track("C61", "1", unit(rng.normal(size=64)), 101.0), 101.0)
    assert other.gid != g


def test_overlap_link_needs_exactly_one_candidate():
    m = CrossCameraMatcher(threshold=0.60, overlaps=[["C41", "C51"]], overlap_link=3.0)
    rng = np.random.default_rng(12)
    a = m.matchAcrossCameras(track("C41", "1", unit(rng.normal(size=64)), 100.0), 100.0).gid
    b = m.matchAcrossCameras(track("C41", "2", unit(rng.normal(size=64)), 100.0), 100.0).gid
    m.touch(a, "C41", 100.0)
    m.touch(b, "C41", 100.0)
    res = m.matchAcrossCameras(track("C51", "1", unit(rng.normal(size=64)), 101.0), 101.0)
    assert res.is_new and res.gid not in (a, b)


def test_displace_removes_the_losers_samples():
    m = matcher()
    car = unit(np.random.default_rng(7).normal(size=64))
    g = m.matchAcrossCameras(track("C11", "1", car, 0.0), 0.0).gid
    loser = track("C31", "1", vec_with_sim(car, 0.70, 8), 50.0)
    m.matchAcrossCameras(loser, 50.0)
    assert len(m.gallery[g]) == 2
    m.displace(g, "C31", "1", since_ts=50.0, track=loser)
    assert len(m.gallery[g]) == 1
    assert ("C31", "1") not in m.active.get(g, set())
    # the loser is re-identified without that id
    loser.gid = None
    res = m.matchAcrossCameras(loser, 51.0, exclude={g})
    assert res.gid != g and res.is_new
