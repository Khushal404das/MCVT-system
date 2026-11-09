"""Create low-resolution "analysis streams" for recorded camera files.

IP cameras normally offer a full-quality main stream and a small sub-stream
for analytics. Recorded files only have the main stream, and decoding several
1440p videos in real time can use up a CPU-only machine. This one-time step
writes a small copy of every file camera (size/FPS from ``cameras.analysis`` in
config.yaml) to storage/streams/. The engine analyses and streams these
copies, while journey videos are still cut from the full-quality originals
(timestamps are identical).

Usage:  python prepare_streams.py [--config ../config.yaml] [--jobs 2]
Already-converted files are skipped, so it is safe to run on every start.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcvt.config import load_config, analysis_path  # noqa: E402
from mcvt.stitcher import find_ffmpeg  # noqa: E402


def convert(ffmpeg: str, src: str, dst: Path, width: int, fps: float) -> str:
    if dst.exists():
        return f"{dst.name}: already prepared"
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(".part.mp4")
    t = time.time()
    # Keep every original frame; only cap the rate at `fps` (e.g. 30) for faster cameras.
    cap = cv2.VideoCapture(src)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    cap.release()
    out_fps = min(src_fps, fps)
    vf = f"scale={width}:-2" if src_fps <= fps else f"fps={fps},scale={width}:-2"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-an", "-vf", vf,
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-g", str(int(out_fps * 2)), str(part)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not part.exists():
        part.unlink(missing_ok=True)
        return f"{dst.name}: FAILED {res.stderr.strip()[-300:]}"
    part.replace(dst)
    return f"{dst.name}: done in {time.time() - t:.0f} s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("cameras.analysis.enabled", False):
        print("cameras.analysis.enabled is false - nothing to do")
        return
    width = int(cfg.get("cameras.analysis.width", 960))
    fps = float(cfg.get("cameras.analysis.fps", 10))
    ffmpeg = find_ffmpeg(cfg.get("stitching.ffmpeg", "auto"))
    todo = [(s.source, analysis_path(cfg, s)) for s in cfg.camera_specs() if s.is_file]
    print(f"Preparing {len(todo)} analysis stream(s) at {width}px / {fps:g} FPS ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for msg in pool.map(lambda t: convert(ffmpeg, t[0], t[1], width, fps), todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
