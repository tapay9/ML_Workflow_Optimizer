"""SQLite persistence layer for ML Workflow Optimizer.

Tables
------
model_versions : every trained model that was registered (one row per version)
deployments    : history of which version was live (supports rollback/audit)
predictions    : log of every prediction served by the deployment interface
"""
import os
import sqlite3
from contextlib import contextmanager

DB_PATH = os.environ.get(
    "MLWO_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mlwo.db"),
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    algorithm     TEXT,
    artifact_path TEXT    NOT NULL,
    sha256        TEXT    NOT NULL,
    params_json   TEXT,
    metrics_json  TEXT,
    features_json TEXT,
    schema_json   TEXT,
    stage         TEXT    NOT NULL DEFAULT 'none'
                  CHECK (stage IN ('none','staging','production','archived')),
    notes         TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS deployments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     INTEGER NOT NULL REFERENCES model_versions(id),
    action       TEXT    NOT NULL CHECK (action IN ('deploy','rollback','retire')),
    deployed_by  TEXT,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id     INTEGER NOT NULL REFERENCES model_versions(id),
    input_json   TEXT    NOT NULL,
    output_json  TEXT    NOT NULL,
    latency_ms   REAL,
    created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_mv_name_stage ON model_versions(name, stage);
CREATE INDEX IF NOT EXISTS idx_pred_model    ON predictions(model_id, created_at);
"""


def init_db(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with get_conn(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn(path=None):
    """Open a connection, commit on success, rollback on error, always close."""
    conn = sqlite3.connect(path or DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
