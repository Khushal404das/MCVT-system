"""Journey video stitching with FFmpeg (FR-VS-01 .. FR-VS-05)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

log = logging.getLogger("stitcher")


def find_ffmpeg(configured: str | None = "auto") -> str:
    """Locate an ffmpeg binary: configured path, PATH, or the imageio-ffmpeg wheel."""
    if configured and configured != "auto":
        return configured
    from shutil import which

    exe = which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class Clip:
    """A stretch of one camera's video in which the vehicle is visible."""

    path: str
    camera_id: str
    start_ts: float  # wall-clock start of the clip
    end_ts: float
    source_start: float  # seconds into the source file where the clip starts
    width: int = 0
    height: int = 0
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    @property
    def quality(self) -> tuple[int, float]:
        return (self.width * self.height, self.score)


@dataclass
class Segment:
    clip: Clip
    start_ts: float
    end_ts: float

    @property
    def source_start(self) -> float:
        return self.clip.source_start + (self.start_ts - self.clip.start_ts)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class StitchResult:
    ok: bool
    output: str | None
    segments: list[Segment] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    duration: float = 0.0
    message: str = ""


class VideoStitcher:
    """Collects clips, removes overlaps, and concatenates them into one MP4."""

    def __init__(self, ffmpeg: str = "auto", width: int = 1280, height: int = 720, fps: int = 15,
                 crf: int = 27, preset: str = "veryfast", min_segment: float = 0.5, label: bool = True):
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self.crf, self.preset = int(crf), preset
        self.min_segment = float(min_segment)
        self.label = label
        self.masks: dict[str, list[float]] = {}  # camera id -> burned-in timestamp area to cover
        self.clips: list[Clip] = []

    def addClip(self, clip: Clip) -> None:
        self.clips.append(clip)

    add_clip = addClip

    # ---------------------------------------------------------------- order
    def ordered(self) -> list[Clip]:
        """Clips sorted by start timestamp (FR-VS-02)."""
        return sorted(self.clips, key=lambda c: (c.start_ts, c.end_ts))

    def deduplicateOverlaps(self) -> list[Segment]:
        """Split the timeline so every instant is covered by exactly one clip -
        the one with the highest resolution (then score) (FR-VS-03, NFR-R-04)."""
        clips = [c for c in self.ordered() if c.duration > 0]
        if not clips:
            return []
        bounds = sorted({t for c in clips for t in (c.start_ts, c.end_ts)})
        segments: list[Segment] = []
        for a, b in zip(bounds, bounds[1:]):
            covering = [c for c in clips if c.start_ts <= a and c.end_ts >= b]
            if not covering:
                continue
            best = max(covering, key=lambda c: c.quality)
            if segments and segments[-1].clip is best and abs(segments[-1].end_ts - a) < 1e-6:
                segments[-1].end_ts = b
            else:
                segments.append(Segment(best, a, b))
        return [s for s in segments if s.duration >= self.min_segment]

    deduplicate_overlaps = deduplicateOverlaps

    # ----------------------------------------------------------------- stitch
    @staticmethod
    def _valid(path: str) -> bool:
        if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
            return False
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        return bool(ok)

    def _cut(self, seg: Segment, out: Path, with_label: bool) -> bool:
        vf = (f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
              f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2,fps={self.fps},setsar=1")
        mask = self.masks.get(seg.clip.camera_id)
        if mask:  # the camera's own burned-in timestamp (cameras.overrides.<id>.timestamp_mask)
            x1, y1, x2, y2 = mask
            vf = (f"drawbox=x=iw*{x1:.4f}:y=ih*{y1:.4f}:w=iw*{x2 - x1:.4f}:h=ih*{y2 - y1:.4f}:"
                  f"color=0x121212:t=fill,") + vf
        if with_label:
            vf += (f",drawtext=text='{seg.clip.camera_id}':x=16:y=16:fontsize=28:fontcolor=white:"
                   f"box=1:boxcolor=black@0.55:boxborderw=8")
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{max(0.0, seg.source_start):.3f}",
               "-i", seg.clip.path, "-t", f"{seg.duration:.3f}", "-an", "-vf", vf,
               "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-pix_fmt", "yuv420p", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            log.debug("ffmpeg failed: %s", res.stderr.strip()[-400:])
            return False
        return True

    def stitch(self, output: str | Path) -> StitchResult:
        """Produce one H.264 MP4 (FR-VS-04). Missing/corrupt clips are skipped (FR-VS-05)."""
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        skipped: list[str] = []
        valid_cache: dict[str, bool] = {}
        for c in self.clips:
            if c.path not in valid_cache:
                valid_cache[c.path] = self._valid(c.path)
                if not valid_cache[c.path]:
                    log.warning("Clip file %s is missing or unreadable - skipping it", c.path)
                    skipped.append(c.path)
        kept = [c for c in self.clips if valid_cache.get(c.path)]
        original, self.clips = self.clips, kept
        try:
            segments = self.deduplicateOverlaps()
        finally:
            self.clips = original
        if not segments:
            return StitchResult(False, None, [], skipped, 0.0, "No valid clips to stitch")

        with tempfile.TemporaryDirectory(prefix="mcvt_stitch_") as tmp:
            parts: list[Path] = []
            used: list[Segment] = []
            label = self.label
            for i, seg in enumerate(segments):
                part = Path(tmp) / f"part_{i:04d}.mp4"
                ok = self._cut(seg, part, label)
                if not ok and label:
                    label = False  # e.g. ffmpeg build without drawtext / fonts
                    ok = self._cut(seg, part, False)
                if ok:
                    parts.append(part)
                    used.append(seg)
                else:
                    log.warning("Could not cut %s at %.1fs - skipping that part", seg.clip.path, seg.source_start)
                    skipped.append(f"{seg.clip.path}@{seg.source_start:.1f}")
            if not parts:
                return StitchResult(False, None, [], skipped, 0.0, "FFmpeg could not read any clip")
            listing = Path(tmp) / "list.txt"
            listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
            cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                   "-c", "copy", "-movflags", "+faststart", str(output)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return StitchResult(False, None, used, skipped, 0.0, f"FFmpeg concat failed: {res.stderr.strip()[-300:]}")
        total = sum(s.duration for s in used)
        msg = f"Stitched {len(used)} segment(s), {total:.1f} s"
        if skipped:
            msg += f"; skipped {len(skipped)} missing/corrupt clip(s)"
        return StitchResult(True, str(output), used, skipped, total, msg)
