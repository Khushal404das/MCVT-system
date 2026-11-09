"""Module 2 - Camera Switching Engine (TC-CS-01 .. TC-CS-05)."""

from mcvt.switching import CameraSwitchingEngine, OverlapDetection, camera_score


def engine():
    return CameraSwitchingEngine(hysteresis=0.15, buffer_frames=5, fallback_frames=3)


def test_tc_cs_01_switch_to_higher_resolution_camera():
    e = engine()
    assert e.selectBestCamera({"C1": 0.55, "C2": 0.82}) == "C2"


def test_tc_cs_02_hysteresis_prevents_unnecessary_switching():
    e = engine()
    e.set_active("C1")
    for _ in range(10):
        assert e.selectBestCamera({"C1": 0.71, "C2": 0.78}) == "C1"
    assert "below the hysteresis threshold" in e.last_reason


def test_tc_cs_03_switch_only_after_five_consecutive_frames():
    e = engine()
    e.set_active("C1")
    history = [e.update({"C1": 0.40, "C2": 0.80}) for _ in range(5)]
    assert [d.switched for d in history] == [False, False, False, False, True]
    assert history[-1].active == "C2"


def test_tc_cs_03_streak_resets_when_candidate_drops():
    e = engine()
    e.set_active("C1")
    for _ in range(4):
        e.update({"C1": 0.40, "C2": 0.80})
    e.update({"C1": 0.40, "C2": 0.45})  # C2 no longer clearly better
    for _ in range(4):
        assert e.update({"C1": 0.40, "C2": 0.80}).active == "C1"
    assert e.update({"C1": 0.40, "C2": 0.80}).active == "C2"


def test_tc_cs_04_overlap_zone_selects_larger_bbox():
    frame_area = 1280 * 720
    c1 = OverlapDetection("C1", 2400, frame_area, 1.0)
    c2 = OverlapDetection("C2", 5100, frame_area, 1.0)
    active, suppressed = CameraSwitchingEngine.handleOverlapZone([c1, c2])
    assert active == "C2" and suppressed == ["C1"]
    assert c2.score == camera_score(5100, frame_area, 1.0)


def test_tc_cs_05_fallback_when_active_camera_loses_vehicle():
    e = engine()
    e.set_active("C1")
    decisions = [e.update({"C2": 0.74, "C3": 0.30}) for _ in range(3)]
    assert [d.active for d in decisions] == ["C1", "C1", "C2"]
    assert decisions[-1].switched


def test_score_formula_uses_resolution_weight():
    assert camera_score(100, 1000, 0.5) == 0.05
