-- MCVT shared database schema (used by both the Python engine and the PHP web app).
-- All statements are idempotent.

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE,
    password_hash TEXT    NOT NULL,
    role          TEXT    NOT NULL DEFAULT 'operator',
    created_at    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS ws_tokens (
    token      TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    id                TEXT PRIMARY KEY,
    name              TEXT,
    source            TEXT NOT NULL,
    width             INTEGER,
    height            INTEGER,
    fps               REAL,
    resolution_weight REAL,
    status            TEXT,
    session_start     REAL,
    updated_at        REAL
);

CREATE TABLE IF NOT EXISTS vehicles (
    gid           TEXT    PRIMARY KEY,
    seq           INTEGER NOT NULL UNIQUE,
    vclass        TEXT,
    color         TEXT,
    first_seen    REAL,
    last_seen     REAL,
    first_camera  TEXT,
    last_camera   TEXT,
    thumb_path    TEXT,
    thumb_score   REAL    DEFAULT 0,
    n_embeddings  INTEGER DEFAULT 0,
    motion        TEXT,
    merged_into   TEXT
);
CREATE INDEX IF NOT EXISTS idx_vehicles_last_seen ON vehicles(last_seen);

CREATE TABLE IF NOT EXISTS embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    gid       TEXT    NOT NULL,
    camera_id TEXT,
    ts        REAL    NOT NULL,
    vec       BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_gid ON embeddings(gid);

-- One row per continuous appearance of a vehicle in one camera = one clip (F9).
CREATE TABLE IF NOT EXISTS sightings (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    gid            TEXT    NOT NULL,
    camera_id      TEXT    NOT NULL,
    session_id     TEXT    NOT NULL,
    local_track_id TEXT    NOT NULL,
    source         TEXT    NOT NULL,
    start_ts       REAL    NOT NULL,
    end_ts         REAL    NOT NULL,
    start_offset   REAL    NOT NULL,
    end_offset     REAL    NOT NULL,
    best_score     REAL    DEFAULT 0,
    frames         INTEGER DEFAULT 0,
    width          INTEGER,
    height         INTEGER,
    media          TEXT,   -- file analysed (start/end_offset are on its timeline), for journey replay
    boxes          TEXT,   -- JSON [[offset, x1, y1, x2, y2], ...] normalised boxes of the vehicle
    UNIQUE (session_id, camera_id, local_track_id, gid)
);
CREATE INDEX IF NOT EXISTS idx_sightings_gid ON sightings(gid, start_ts);

CREATE TABLE IF NOT EXISTS switch_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    gid      TEXT,
    from_cam TEXT,
    to_cam   TEXT,
    reason   TEXT,
    ts       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    gid          TEXT    NOT NULL,
    recipient    TEXT,
    status       TEXT    NOT NULL DEFAULT 'queued',
    message      TEXT,
    output_path  TEXT,
    requested_by INTEGER,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS logs (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    level   TEXT NOT NULL,
    source  TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_logs_ts ON logs(ts);
