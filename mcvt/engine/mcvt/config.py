"""Configuration loading: config.yaml + .env, camera discovery.

Nothing camera-specific is hard-coded: cameras come from the configured
discovery folder and/or the explicit ``cameras.sources`` list, and their
resolution, FPS and resolution weight are read from the streams themselves.
"""

from __future__ import annotations

import os
import re
import glob
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path) -> dict[str, str]:
    """Read KEY=VALUE pairs from a .env file into os.environ (existing vars win)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        values[key] = val
        os.environ.setdefault(key, val)
    return values


def ensure_env(path: Path) -> None:
    """Create .env with a random ENGINE_TOKEN if it does not exist yet."""
    if path.exists():
        load_env(path)
        if os.environ.get("ENGINE_TOKEN"):
            return
    token = secrets.token_hex(24)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\nENGINE_TOKEN={token}\n")
    os.environ["ENGINE_TOKEN"] = token


@dataclass
class CameraSpec:
    """Static description of one camera source."""

    id: str
    source: str  # full-quality stream / file (used for journey clips)
    name: str
    resolution_weight: float | None = None
    is_file: bool = False
    analysis_source: str | None = None  # optional low-res stream used for AI + live view
    extra: dict[str, Any] = field(default_factory=dict)


class Config:
    """Thin wrapper around the parsed YAML with dotted-path access."""

    def __init__(self, data: dict[str, Any], root: Path):
        self.data = data
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default: str | None = None) -> Path:
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(dotted)
        p = Path(str(value))
        return p if p.is_absolute() else (self.root / p).resolve()

    # ------------------------------------------------------------------ cameras
    def camera_specs(self) -> list[CameraSpec]:
        """Return all configured cameras (discovered files + explicit sources)."""
        specs: dict[str, CameraSpec] = {}
        disc = self.get("cameras.discover") or {}
        if disc.get("directory"):
            folder = Path(disc["directory"])
            folder = folder if folder.is_absolute() else (self.root / folder).resolve()
            pattern = disc.get("pattern", "*.mp4")
            regex = re.compile(disc.get("id_regex") or r"(.+)", re.IGNORECASE)
            prefix = disc.get("id_prefix", "")
            for f in sorted(glob.glob(str(folder / pattern))):
                stem = Path(f).stem
                m = regex.search(stem)
                cam_id = f"{prefix}{m.group(1)}" if m else stem
                specs[cam_id] = CameraSpec(id=cam_id, source=f, name=cam_id, is_file=True)
        for item in self.get("cameras.sources") or []:
            cam_id = str(item["id"])
            src = str(item["source"])
            is_file = not re.match(r"^[a-z]+://", src, re.IGNORECASE)
            if is_file and not Path(src).is_absolute():
                src = str((self.root / src).resolve())
            specs[cam_id] = CameraSpec(
                id=cam_id,
                source=src,
                name=str(item.get("name", cam_id)),
                resolution_weight=item.get("resolution_weight"),
                is_file=is_file,
                analysis_source=item.get("analysis_source"),  # e.g. the camera's RTSP sub-stream
            )
        overrides = self.get("cameras.overrides") or {}
        result = []
        for cam_id, spec in specs.items():
            ov = overrides.get(cam_id) or {}
            if ov.get("enabled", True) is False:
                continue
            spec.name = str(ov.get("name", spec.name))
            if ov.get("resolution_weight") is not None:
                spec.resolution_weight = float(ov["resolution_weight"])
            if ov.get("start_seconds"):
                spec.extra["start_seconds"] = float(ov["start_seconds"])  # skip to align recordings
            if ov.get("start_clock"):
                spec.extra["start_clock"] = str(ov["start_clock"])  # recording start, e.g. "08:07:28"
            if ov.get("analysis_start_seconds"):
                spec.extra["analysis_start_seconds"] = float(ov["analysis_start_seconds"])
            if ov.get("clip_offset_seconds"):
                spec.extra["clip_offset_seconds"] = float(ov["clip_offset_seconds"])
            # shown clock at the camera's first played frame (cameras.clock_start unless overridden)
            clock = ov.get("clock_start", self.get("cameras.clock_start"))
            if clock:
                spec.extra["clock_start"] = clock_seconds(clock)
            # the camera's own burned-in timestamp [x1, y1, x2, y2] (0..1 of the frame): covered
            if ov.get("timestamp_mask"):
                spec.extra["timestamp_mask"] = [float(v) for v in ov["timestamp_mask"]]
            if spec.is_file and not spec.analysis_source and self.get("cameras.analysis.enabled", False):
                proxy = analysis_path(self, spec)
                if proxy.exists():
                    spec.analysis_source = str(proxy)
            result.append(spec)
        return result


def clock_seconds(text) -> float:
    """"08:00:00" -> 28800.0 (seconds of the day)."""
    parts = [float(p) for p in str(text).strip().split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def format_clock(seconds: float) -> str:
    """28923.4 -> "08:02:03"."""
    s = int(seconds) % 86400
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def analysis_path(cfg: "Config", spec: CameraSpec) -> Path:
    """Where the low-resolution analysis copy of a file camera is stored."""
    width = int(cfg.get("cameras.analysis.width", 960))
    fps = cfg.get("cameras.analysis.fps", 10)
    return cfg.path("cameras.analysis.directory", "storage/streams") / f"{spec.id}_{width}p{fps}.mp4"


def load_config(config_path: str | Path | None = None) -> Config:
    """Load config.yaml and the sibling .env file."""
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    path = path.resolve()
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    ensure_env(path.parent / ".env")
    return Config(data, path.parent)
