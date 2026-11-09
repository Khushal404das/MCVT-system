"""Module 4 - Video Stitching and Email Delivery (TC-VE-01 .. TC-VE-05)."""

from __future__ import annotations

import base64
import socketserver
import subprocess
import threading
from email import message_from_bytes
from pathlib import Path

import cv2
import numpy as np
import pytest

from mcvt.emailer import EmailService
from mcvt.stitcher import Clip, VideoStitcher, find_ffmpeg


# --------------------------------------------------------------------- helpers
def make_clip(path: Path, color: str, seconds: float, size: str = "640x360") -> str:
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c={color}:s={size}:r=15:d={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return str(path)


def video_info(path: str) -> tuple[float, list[np.ndarray]]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return len(frames) / fps, frames


def dominant(frame: np.ndarray) -> str:
    b, g, r = frame[frame.shape[0] // 2:, :].reshape(-1, 3).mean(axis=0)  # lower half (label is at the top)
    return max((("red", r), ("green", g), ("blue", b)), key=lambda x: x[1])[0]


@pytest.fixture()
def stitcher():
    return VideoStitcher(width=640, height=360, fps=15, crf=30, min_segment=0.2, label=False)


# ------------------------------------------------------------------- stitching
def test_tc_ve_01_stitch_clips_in_chronological_order(tmp_path, stitcher):
    a = make_clip(tmp_path / "a.mp4", "red", 15)
    b = make_clip(tmp_path / "b.mp4", "green", 14)
    c = make_clip(tmp_path / "c.mp4", "blue", 13)
    for clip in [Clip(c, "C3", 27, 40, 0), Clip(a, "C1", 0, 15, 0), Clip(b, "C2", 14, 28, 0)]:  # random order
        stitcher.addClip(clip)
    res = stitcher.stitch(tmp_path / "out.mp4")
    assert res.ok, res.message
    assert [s.clip.camera_id for s in res.segments] == ["C1", "C2", "C3"]
    duration, frames = video_info(res.output)
    assert abs(duration - 40) <= 1.0
    order = []
    for f in frames[:: 15]:
        col = dominant(f)
        if not order or order[-1] != col:
            order.append(col)
    assert order == ["red", "green", "blue"]


def test_tc_ve_02_deduplicate_overlapping_segments(tmp_path, stitcher):
    low = make_clip(tmp_path / "c1_720.mp4", "red", 8, "1280x720")
    high = make_clip(tmp_path / "c2_1080.mp4", "blue", 8, "1920x1080")
    stitcher.addClip(Clip(low, "C1", 10, 18, 0, 1280, 720))
    stitcher.addClip(Clip(high, "C2", 10, 18, 0, 1920, 1080))
    segments = stitcher.deduplicateOverlaps()
    assert len(segments) == 1 and segments[0].clip.camera_id == "C2"
    res = stitcher.stitch(tmp_path / "out.mp4")
    duration, frames = video_info(res.output)
    assert abs(duration - 8) <= 0.5  # no time window appears twice
    assert {dominant(f) for f in frames[::10]} == {"blue"}


def test_tc_ve_04_handle_missing_clip_gracefully(tmp_path, stitcher, caplog):
    a = make_clip(tmp_path / "a.mp4", "red", 4)
    c = make_clip(tmp_path / "c.mp4", "blue", 4)
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"not a video")
    stitcher.addClip(Clip(a, "C1", 0, 4, 0))
    stitcher.addClip(Clip(str(tmp_path / "deleted.mp4"), "C2", 4, 8, 0))
    stitcher.addClip(Clip(str(corrupt), "C3", 8, 12, 0))
    stitcher.addClip(Clip(c, "C4", 12, 16, 0))
    with caplog.at_level("WARNING"):
        res = stitcher.stitch(tmp_path / "partial.mp4")
    assert res.ok and len(res.skipped) == 2
    assert "missing or unreadable" in caplog.text
    duration, _ = video_info(res.output)
    assert abs(duration - 8) <= 0.5


# ----------------------------------------------------------------------- email
class _FakeSMTP(socketserver.StreamRequestHandler):
    """Tiny SMTP server: AUTH PLAIN/LOGIN, stores the received message."""

    def write(self, line: str):
        self.wfile.write((line + "\r\n").encode())

    def handle(self):
        srv = self.server
        self.write("220 fake ESMTP")
        while True:
            line = self.rfile.readline().decode(errors="replace").strip()
            if not line:
                return
            cmd = line.upper()
            if cmd.startswith("EHLO") or cmd.startswith("HELO"):
                self.wfile.write(b"250-fake\r\n250-AUTH PLAIN LOGIN\r\n250 SIZE 52428800\r\n")
            elif cmd.startswith("AUTH PLAIN"):
                parts = line.split()
                blob = parts[2] if len(parts) > 2 else (self.write("334 ") or self.rfile.readline().strip().decode())
                _, user, pwd = base64.b64decode(blob).decode().split("\0")
                self._auth(user, pwd)
            elif cmd.startswith("AUTH LOGIN"):
                self.write("334 VXNlcm5hbWU6")
                user = base64.b64decode(self.rfile.readline().strip()).decode()
                self.write("334 UGFzc3dvcmQ6")
                pwd = base64.b64decode(self.rfile.readline().strip()).decode()
                self._auth(user, pwd)
            elif cmd.startswith(("MAIL", "RCPT", "RSET", "NOOP")):
                self.write("250 OK")
            elif cmd == "DATA":
                self.write("354 go ahead")
                data = b""
                while True:
                    chunk = self.rfile.readline()
                    if chunk in (b".\r\n", b".\n", b""):
                        break
                    data += chunk[1:] if chunk.startswith(b"..") else chunk
                srv.messages.append(data)
                self.write("250 queued")
            elif cmd == "QUIT":
                self.write("221 bye")
                return
            else:
                self.write("502 not implemented")

    def _auth(self, user, pwd):
        if (user, pwd) == self.server.credentials:
            self.write("235 authenticated")
        else:
            self.write("535 5.7.8 Username and Password not accepted")


@pytest.fixture()
def smtp_server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _FakeSMTP)
    srv.daemon_threads = True
    srv.messages, srv.credentials = [], ("operator@example.com", "app-password")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


@pytest.fixture()
def journey_video(tmp_path):
    return Path(make_clip(tmp_path / "vehicle_VH001_journey.mp4", "green", 2))


def test_tc_ve_03_email_sent_with_attachment(smtp_server, journey_video):
    svc = EmailService("127.0.0.1", smtp_server.server_address[1], "operator@example.com", "app-password",
                       security="none", timeout=10)
    svc.recipient = "recipient@example.com"
    assert svc.attachVideo(journey_video)
    assert svc.sendEmail(subject="Journey video for vehicle VH-001", body="Attached.") is True
    msg = message_from_bytes(smtp_server.messages[0])
    assert msg["Subject"] == "Journey video for vehicle VH-001"
    att = [p for p in msg.walk() if p.get_filename()]
    assert att and att[0].get_filename() == journey_video.name
    assert att[0].get_payload(decode=True) == journey_video.read_bytes()


def test_tc_ve_05_email_failure_on_invalid_credentials(smtp_server, journey_video):
    svc = EmailService("127.0.0.1", smtp_server.server_address[1], "operator@example.com", "wrong-password",
                       security="none", timeout=10)
    svc.attachVideo(journey_video)
    assert svc.sendEmail("recipient@example.com") is False  # no exception escapes
    assert "SMTP login failed" in svc.last_error
    assert smtp_server.messages == []
