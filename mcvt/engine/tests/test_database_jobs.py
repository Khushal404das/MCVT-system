"""Module 5: Database & journey jobs (TC-DB-01 .. TC-DB-06).

Each test uses a fresh SQLite file in a temporary folder, built from the real
schema.sql, so the project database is never touched.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mcvt.db import Database
from mcvt.jobs import collect_clips, resolve_gid


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.sqlite")
    yield d
    d.conn.close()


def _vehicle(db, gid, ts=1000.0, cam="C11"):
    db.create_vehicle(gid, db.next_vehicle_seq(), ts, cam, "car")


def _sighting(db, gid, cam, start, end, offset, track="1"):
    db.execute(
        """INSERT INTO sightings(gid,camera_id,session_id,local_track_id,source,start_ts,end_ts,
                                 start_offset,end_offset,best_score,width,height)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (gid, cam, "s1", track, f"{cam}.mp4", start, end, offset, offset + (end - start), 0.5, 1920, 1080),
    )


def test_tc_db_01_schema_creates_all_tables(db):
    """TC-DB-01: a new database gets every table the system needs, in WAL mode."""
    tables = {r["name"] for r in db.query("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("users", "cameras", "vehicles", "embeddings", "sightings", "switch_events", "jobs", "logs"):
        assert t in tables, f"table {t} missing"
    assert db.one("PRAGMA journal_mode")[0].lower() == "wal"


def test_tc_db_02_global_ids_are_sequential_and_unique(db):
    """TC-DB-02: vehicle sequence numbers increase and a duplicate global ID is rejected."""
    _vehicle(db, "VH-001")
    _vehicle(db, "VH-002")
    assert db.next_vehicle_seq() == 3
    with pytest.raises(Exception):
        db.create_vehicle("VH-001", 99, 1.0, "C11", "car")


def test_tc_db_03_embeddings_round_trip(db):
    """TC-DB-03: a 2048-d feature vector is stored and read back unchanged."""
    _vehicle(db, "VH-001")
    vec = np.random.default_rng(0).standard_normal(2048).astype(np.float32)
    vec /= np.linalg.norm(vec)
    db.add_embeddings([("VH-001", "C11", 1000.0, vec)])
    gallery = db.load_gallery(since_ts=0, per_gid=5)
    assert np.allclose(gallery["VH-001"][0], vec)
    assert db.one("SELECT n_embeddings FROM vehicles WHERE gid='VH-001'")[0] == 1


def test_tc_db_04_merge_moves_all_data_to_one_vehicle(db):
    """TC-DB-04: merging VH-002 into VH-001 moves features + clips and widens the time range."""
    _vehicle(db, "VH-001", ts=1000.0)
    _vehicle(db, "VH-002", ts=900.0, cam="C21")
    db.add_embeddings([("VH-002", "C21", 900.0, np.ones(2048, np.float32))])
    _sighting(db, "VH-002", "C21", 900.0, 910.0, 50.0)
    db.merge_vehicle("VH-002", "VH-001")

    assert db.one("SELECT merged_into FROM vehicles WHERE gid='VH-002'")[0] == "VH-001"
    assert db.one("SELECT COUNT(*) FROM sightings WHERE gid='VH-001'")[0] == 1
    assert db.one("SELECT COUNT(*) FROM embeddings WHERE gid='VH-002'")[0] == 0
    assert db.one("SELECT first_seen FROM vehicles WHERE gid='VH-001'")[0] == 900.0
    assert "VH-002" not in db.load_gallery(since_ts=0, per_gid=5)   # merged ID no longer matched


def test_tc_db_05_merge_chain_resolves_to_final_id(db):
    """TC-DB-05: a request for an old ID follows merges (VH-003 -> VH-002 -> VH-001)."""
    for g in ("VH-001", "VH-002", "VH-003"):
        _vehicle(db, g)
    db.merge_vehicle("VH-002", "VH-001")
    db.merge_vehicle("VH-003", "VH-002")
    assert resolve_gid(db, "VH-003") == "VH-001"
    assert resolve_gid(db, "VH-001") == "VH-001"


def test_tc_db_06_journey_clips_in_time_order_with_padding(db):
    """TC-DB-06: clips for a journey come out in time order, padded and capped."""
    _vehicle(db, "VH-001")
    _sighting(db, "VH-001", "C31", 1040.0, 1050.0, 40.0, track="3")
    _sighting(db, "VH-001", "C11", 1000.0, 1010.0, 0.5, track="1")
    _sighting(db, "VH-001", "C21", 1020.0, 1200.0, 20.0, track="2")
    clips = collect_clips(db, "VH-001", padding=1.0, max_clip=60.0)

    assert [c.camera_id for c in clips] == ["C11", "C21", "C31"]
    assert clips[0].source_start == 0.0                        # padding never before the file start
    assert clips[2].source_start == pytest.approx(39.0)        # 1 s lead-in
    assert clips[1].end_ts - clips[1].start_ts <= 60.0          # long clip capped


def test_tc_db_07_job_queue_is_claimed_once(db):
    """TC-DB-07: a queued journey request is picked up by exactly one worker."""
    now = time.time()
    db.execute("INSERT INTO jobs(gid,recipient,status,created_at,updated_at) VALUES(?,?,?,?,?)",
               ("VH-001", "user@example.test", "queued", now, now))
    claimed = []
    workers = [threading.Thread(target=lambda: claimed.append(db.claim_job())) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert sum(1 for c in claimed if c is not None) == 1
    db.update_job(1, "done", "Sent", "out.mp4")
    row = db.one("SELECT status, output_path FROM jobs WHERE id=1")
    assert (row["status"], row["output_path"]) == ("done", "out.mp4")
