"""SQLite persistence: vehicles, embeddings (feature DB), sightings (clips), jobs, logs."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PROJECT_ROOT


class Database:
    """Thread-safe wrapper around one SQLite connection (WAL mode)."""

    def __init__(self, path: Path, schema: Path | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        schema = schema or PROJECT_ROOT / "schema.sql"
        with self._lock:
            self.conn.executescript(schema.read_text(encoding="utf-8"))
            # start_clock / end_clock: shared camera clock (seconds of the day, cameras.clock_start)
            self._add_missing_columns("sightings", {"media": "TEXT", "boxes": "TEXT",
                                                    "start_clock": "REAL", "end_clock": "REAL"})
            self.conn.commit()

    def _add_missing_columns(self, table: str, columns: dict[str, str]) -> None:
        """Upgrade a database created by an older schema.sql."""
        have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in have:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    # ----------------------------------------------------------------- helpers
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, tuple(params))
            self.conn.commit()
            return cur

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            self.conn.executemany(sql, rows)
            self.conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchall()

    def one(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.conn.execute(sql, tuple(params)).fetchone()

    # ----------------------------------------------------------------- cameras
    def upsert_camera(self, cam: dict[str, Any]) -> None:
        self.execute(
            """INSERT INTO cameras(id,name,source,width,height,fps,resolution_weight,status,session_start,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, source=excluded.source,
                 width=excluded.width, height=excluded.height, fps=excluded.fps,
                 resolution_weight=excluded.resolution_weight, status=excluded.status,
                 session_start=excluded.session_start, updated_at=excluded.updated_at""",
            (cam["id"], cam["name"], cam["source"], cam.get("width"), cam.get("height"),
             cam.get("fps"), cam.get("resolution_weight"), cam.get("status"),
             cam.get("session_start"), time.time()),
        )

    def set_camera_status(self, cam_id: str, status: str) -> None:
        self.execute("UPDATE cameras SET status=?, updated_at=? WHERE id=?", (status, time.time(), cam_id))

    # ---------------------------------------------------------------- vehicles
    def next_vehicle_seq(self) -> int:
        row = self.one("SELECT COALESCE(MAX(seq),0)+1 AS n FROM vehicles")
        return int(row["n"]) if row else 1

    def create_vehicle(self, gid: str, seq: int, ts: float, camera_id: str, vclass: str | None) -> None:
        self.execute(
            "INSERT INTO vehicles(gid,seq,vclass,first_seen,last_seen,first_camera,last_camera) VALUES(?,?,?,?,?,?,?)",
            (gid, seq, vclass, ts, ts, camera_id, camera_id),
        )

    def add_embeddings(self, rows: list[tuple[str, str, float, np.ndarray]]) -> None:
        self.executemany(
            "INSERT INTO embeddings(gid,camera_id,ts,vec) VALUES(?,?,?,?)",
            [(g, c, t, np.asarray(v, dtype=np.float32).tobytes()) for g, c, t, v in rows],
        )
        counts: dict[str, int] = {}
        for g, *_ in rows:
            counts[g] = counts.get(g, 0) + 1
        self.executemany(
            "UPDATE vehicles SET n_embeddings = n_embeddings + ? WHERE gid=?",
            [(n, g) for g, n in counts.items()],
        )

    def load_gallery(self, since_ts: float, per_gid: int) -> dict[str, list[np.ndarray]]:
        rows = self.query(
            """SELECT e.gid, e.vec FROM embeddings e JOIN vehicles v ON v.gid=e.gid
               WHERE v.merged_into IS NULL AND v.last_seen >= ? ORDER BY e.id DESC""",
            (since_ts,),
        )
        gallery: dict[str, list[np.ndarray]] = {}
        for r in rows:
            lst = gallery.setdefault(r["gid"], [])
            if len(lst) < per_gid:
                lst.append(np.frombuffer(r["vec"], dtype=np.float32).copy())
        return gallery

    def load_vehicle_attributes(self, since_ts: float) -> dict[str, tuple[str | None, str | None]]:
        """gid -> (colour, class) of vehicles seen since ``since_ts``."""
        rows = self.query("SELECT gid, color, vclass FROM vehicles WHERE merged_into IS NULL AND last_seen >= ?",
                          (since_ts,))
        return {r["gid"]: (r["color"], r["vclass"]) for r in rows}

    def merge_vehicle(self, src: str, dst: str) -> None:
        """Fold global id ``src`` into ``dst`` (both observed to be one vehicle)."""
        with self._lock:
            c = self.conn
            c.execute("UPDATE embeddings SET gid=? WHERE gid=?", (dst, src))
            c.execute("UPDATE OR IGNORE sightings SET gid=? WHERE gid=?", (dst, src))
            c.execute("DELETE FROM sightings WHERE gid=?", (src,))
            c.execute("UPDATE switch_events SET gid=? WHERE gid=?", (dst, src))
            c.execute(
                """UPDATE vehicles SET
                     first_seen = MIN(first_seen, (SELECT first_seen FROM vehicles WHERE gid=?)),
                     last_seen  = MAX(last_seen,  (SELECT last_seen  FROM vehicles WHERE gid=?)),
                     n_embeddings = n_embeddings + (SELECT n_embeddings FROM vehicles WHERE gid=?)
                   WHERE gid=?""",
                (src, src, src, dst),
            )
            c.execute("UPDATE vehicles SET merged_into=?, n_embeddings=0 WHERE gid=?", (dst, src))
            c.commit()

    # ---------------------------------------------------------------- logging
    def log(self, level: str, source: str, message: str) -> None:
        self.execute("INSERT INTO logs(ts,level,source,message) VALUES(?,?,?,?)", (time.time(), level, source, message))

    # ------------------------------------------------------------------- jobs
    def claim_job(self) -> sqlite3.Row | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            self.conn.execute("UPDATE jobs SET status='running', message=?, updated_at=? WHERE id=?",
                              ("Collecting clips", time.time(), row["id"]))
            self.conn.commit()
            return row

    def update_job(self, job_id: int, status: str, message: str, output_path: str | None = None) -> None:
        self.execute(
            "UPDATE jobs SET status=?, message=?, output_path=COALESCE(?, output_path), updated_at=? WHERE id=?",
            (status, message, output_path, time.time(), job_id),
        )
