"""Journey-video requests: collect clips -> stitch -> email (UC-12 .. UC-14).

Requests are queued in the ``jobs`` table by the PHP web app; this worker
picks them up, so a slow FFmpeg or SMTP call never blocks live tracking.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from . import logbus
from .emailer import EmailService
from .stitcher import Clip, VideoStitcher

log = logging.getLogger("jobs")


def resolve_gid(db, gid: str) -> str:
    """Follow merge links (VH-007 merged into VH-002 -> VH-002)."""
    seen = set()
    while gid not in seen:
        seen.add(gid)
        row = db.one("SELECT merged_into FROM vehicles WHERE gid=?", (gid,))
        if not row or not row["merged_into"]:
            return gid
        gid = row["merged_into"]
    return gid


def collect_clips(db, gid: str, padding: float, max_clip: float) -> list[Clip]:
    """All clips of a vehicle from the clip index (FR-VS-01)."""
    rows = db.query("SELECT * FROM sightings WHERE gid=? ORDER BY start_ts", (gid,))
    clips = []
    for r in rows:
        start_off = max(0.0, r["start_offset"] - padding)
        lead = r["start_offset"] - start_off
        start_ts = r["start_ts"] - lead
        end_ts = r["end_ts"] + padding
        if max_clip and end_ts - start_ts > max_clip:
            end_ts = start_ts + max_clip
        clips.append(Clip(path=r["source"], camera_id=r["camera_id"], start_ts=start_ts, end_ts=end_ts,
                          source_start=start_off, width=r["width"] or 0, height=r["height"] or 0,
                          score=r["best_score"] or 0.0))
    return clips


class JourneyWorker(threading.Thread):
    def __init__(self, cfg, db, bridge, stop_event: threading.Event):
        super().__init__(name="journeys", daemon=True)
        self.cfg, self.db, self.bridge, self.stop_event = cfg, db, bridge, stop_event
        self.out_dir = cfg.path("paths.storage") / "journeys"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _status(self, job_id: int, gid: str, status: str, message: str, output: str | None = None) -> None:
        self.db.update_job(job_id, status, message, output)
        self.bridge.send_event({"type": "job", "id": job_id, "gid": gid, "status": status, "message": message,
                                "output": output})

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                job = self.db.claim_job()
            except Exception:
                log.exception("could not read job queue")
                job = None
            if job is None:
                self.stop_event.wait(1.0)
                continue
            try:
                self.process(job)
            except Exception as exc:  # a failing job must never stop the engine
                log.exception("job %s crashed", job["id"])
                self._status(job["id"], job["gid"], "failed", f"Unexpected error: {exc}")

    def process(self, job) -> None:
        job_id, gid = int(job["id"]), resolve_gid(self.db, job["gid"])
        cfg = self.cfg
        logbus.info("journey", f"Building the full journey video for vehicle {gid} (request #{job_id}).")
        clips = collect_clips(self.db, gid, float(cfg.get("stitching.clip_padding", 1.0)),
                              float(cfg.get("stitching.max_clip_seconds", 90)))
        if not clips:
            self._status(job_id, gid, "failed", f"No recorded clips exist yet for vehicle {gid}.")
            logbus.warning("journey", f"Request #{job_id}: no clips recorded for {gid}.")
            return
        self._status(job_id, gid, "running", f"Stitching {len(clips)} clip(s) from "
                                             f"{len({c.camera_id for c in clips})} camera(s)")
        stitcher = VideoStitcher(cfg.get("stitching.ffmpeg", "auto"), cfg.get("stitching.output_width", 1280),
                                 cfg.get("stitching.output_height", 720), cfg.get("stitching.fps", 15),
                                 cfg.get("stitching.crf", 27), cfg.get("stitching.preset", "veryfast"),
                                 cfg.get("stitching.min_segment_seconds", 0.5), cfg.get("stitching.label_clips", True))
        stitcher.masks = {cid: [float(v) for v in ov["timestamp_mask"]]
                          for cid, ov in (cfg.get("cameras.overrides") or {}).items()
                          if isinstance(ov, dict) and ov.get("timestamp_mask")}
        for c in clips:
            stitcher.addClip(c)
        out = self.out_dir / f"{gid}_journey_{job_id}.mp4"
        t = time.time()
        result = stitcher.stitch(out)
        for s in result.skipped:
            logbus.warning("journey", f"Request #{job_id}: clip {Path(s.split('@')[0]).name} was missing or unreadable and was skipped.")
        if not result.ok:
            self._status(job_id, gid, "failed", result.message)
            logbus.error("journey", f"Request #{job_id}: stitching failed - {result.message}.")
            return
        rel = out.relative_to(cfg.path("paths.storage")).as_posix()
        cams = " -> ".join(dict.fromkeys(s.clip.camera_id for s in result.segments))
        logbus.info("journey", f"Journey video for {gid} ready: {result.duration:.0f} s across {cams} "
                               f"(built in {time.time() - t:.0f} s).")
        recipient = (job["recipient"] or "").strip()
        if not recipient:
            self._status(job_id, gid, "done", f"Video ready ({result.duration:.0f} s, {cams}).", rel)
            return
        self._status(job_id, gid, "sending", f"Video ready, emailing it to {recipient}", rel)
        mail = EmailService.from_config(cfg)
        mail.recipient = recipient
        mail.attachVideo(out)
        first = min(s.start_ts for s in result.segments)
        last = max(s.end_ts for s in result.segments)
        body = (f"Journey video for vehicle {gid}.\n\n"
                f"Cameras (in order): {cams}\n"
                f"From {datetime.fromtimestamp(first):%Y-%m-%d %H:%M:%S} to {datetime.fromtimestamp(last):%H:%M:%S}\n"
                f"Length: {result.duration:.0f} seconds, {len(result.segments)} segment(s).\n\n"
                f"Sent automatically by the MCVT multi-camera vehicle tracking system.")
        subject = str(cfg.get("email.subject", "MCVT journey video for vehicle {gid}")).format(gid=gid)
        if mail.sendEmail(subject=subject, body=body):
            self._status(job_id, gid, "sent", f"Emailed to {recipient} ({result.duration:.0f} s video).", rel)
            logbus.info("email", f"Journey video for {gid} was emailed to {recipient}.")
        else:
            self._status(job_id, gid, "email_failed", f"Video is ready but email failed: {mail.last_error}", rel)
            logbus.error("email", f"Could not email the journey video for {gid}: {mail.last_error}")
