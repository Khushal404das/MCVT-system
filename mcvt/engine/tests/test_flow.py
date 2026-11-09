"""Optical-flow box following between AI updates."""

import numpy as np

from mcvt.flow import FlowBoxes


def test_box_follows_moving_vehicle(frames):
    frame = frames["clear_car"]
    h, w = frame.shape[:2]
    # The car at the gate: take a box around a textured region and move the whole image.
    box = np.array([0.40, 0.30, 0.60, 0.55], np.float32)
    fb = FlowBoxes(width=480)
    fb.reset(frame, [{"box": box.copy()}], ts=0.0)
    dx, dy = 24, 12  # pixels at full resolution
    moved = np.roll(np.roll(frame, dy, axis=0), dx, axis=1)
    out = fb.step(moved)[0]["box"]
    assert abs((out[0] - box[0]) * w - dx) < 6
    assert abs((out[1] - box[1]) * h - dy) < 6


def test_box_stays_when_nothing_moves(frames):
    frame = frames["clear_car"]
    box = np.array([0.40, 0.30, 0.60, 0.55], np.float32)
    fb = FlowBoxes(width=480)
    fb.reset(frame, [{"box": box.copy()}], ts=0.0)
    out = fb.step(frame.copy())[0]["box"]
    assert np.allclose(out, box, atol=0.005)
