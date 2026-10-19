"""Model versioning: register, list, compare, promote and roll back models."""
import hashlib
import json
import os

import joblib
import pandas as pd

from src.db import get_conn, init_db

MODELS_DIR = os.environ.get(
    "MLWO_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
)
STAGES = ("none", "staging", "production", "archived")
DEFAULT_MODEL_NAME = "hotel_cancellation"   # one registry family; each algorithm is a version


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _row(r):
    if r is None:
        return None
    d = dict(r)
    for k in ("params", "metrics", "features", "schema"):
        raw = d.pop(f"{k}_json", None)
        d[k] = json.loads(raw) if raw else None
    return d


def build_input_schema(X, groups):
    """Describe the RAW input a model expects, from a one-hot encoded matrix.

    X       : encoded feature DataFrame (numeric columns + dummy columns)
    groups  : dict {encoded column -> original column}; numeric columns map to themselves
    Returns {"numeric": [...], "categorical": [...], "example": {...}} where `example`
    is a typical raw record (median numerics, most frequent category levels).
    """
    numeric = [c for c in X.columns if groups[c] == c]
    categorical = []
    for c in X.columns:
        g = groups[c]
        if g != c and g not in categorical:
            categorical.append(g)

    example = {c: float(X[c].median()) for c in numeric}
    for g in categorical:
        dummies = [c for c in X.columns if groups[c] == g]
        top = X[dummies].mean().idxmax()
        example[g] = top[len(g) + 1:]
    for c in numeric:                      # tidy ints
        if float(example[c]).is_integer():
            example[c] = int(example[c])
    return {"numeric": numeric, "categorical": categorical, "example": example}


def register_model(model, name, metrics=None, params=None, features=None,
                   schema=None, algorithm=None, notes=None):
    """Save a trained model as the next version of `name`.

    features : list of the model's input column names (after encoding)
    schema   : optional output of build_input_schema(); lets the deployment interface
               accept raw records (strings for categoricals) and encode them.
    Returns the registry row (dict) for the new version.
    """
    init_db()
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model
    algorithm = algorithm or type(estimator).__name__
    if params is None and hasattr(estimator, "get_params"):
        params = {k: v for k, v in estimator.get_params(deep=False).items()
                  if isinstance(v, (int, float, str, bool, type(None)))}
    with get_conn() as conn:
        row = conn.execute("SELECT COALESCE(MAX(version),0)+1 AS v FROM model_versions WHERE name=?",
                           (name,)).fetchone()
        version = row["v"]
        folder = os.path.join(MODELS_DIR, name, f"v{version}")
        os.makedirs(folder, exist_ok=True)
        artifact = os.path.join(folder, "model.joblib")
        joblib.dump(model, artifact)
        cur = conn.execute(
            """INSERT INTO model_versions
               (name, version, algorithm, artifact_path, sha256,
                params_json, metrics_json, features_json, schema_json, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (name, version, algorithm, os.path.relpath(artifact, MODELS_DIR), _sha256(artifact),
             json.dumps(params, default=str), json.dumps(metrics, default=str),
             json.dumps(list(features)) if features is not None else None,
             json.dumps(schema, default=str) if schema is not None else None, notes))
        return _row(conn.execute("SELECT * FROM model_versions WHERE id=?", (cur.lastrowid,)).fetchone())


def list_models():
    """One summary row per model name."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT name, COUNT(*) AS versions, MAX(version) AS latest_version,
                   MAX(CASE WHEN stage='production' THEN version END) AS production_version
            FROM model_versions GROUP BY name ORDER BY name""").fetchall()
        return [dict(r) for r in rows]


def list_versions(name):
    init_db()
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM model_versions WHERE name=? ORDER BY version DESC",
                            (name,)).fetchall()
        return [_row(r) for r in rows]


def get_version(name, version=None, stage=None):
    """Fetch by explicit version, by stage ('production'), or latest if neither given."""
    init_db()
    with get_conn() as conn:
        if version is not None:
            r = conn.execute("SELECT * FROM model_versions WHERE name=? AND version=?",
                             (name, version)).fetchone()
        elif stage:
            r = conn.execute("SELECT * FROM model_versions WHERE name=? AND stage=? "
                             "ORDER BY version DESC LIMIT 1", (name, stage)).fetchone()
        else:
            r = conn.execute("SELECT * FROM model_versions WHERE name=? ORDER BY version DESC LIMIT 1",
                             (name,)).fetchone()
        return _row(r)


def load_model(name, version=None, stage=None, verify=True):
    """Load the joblib artifact; optionally verify its checksum first."""
    meta = get_version(name, version, stage)
    if meta is None:
        raise LookupError(f"No model found for name={name!r} version={version} stage={stage}")
    path = os.path.join(MODELS_DIR, meta["artifact_path"])
    if verify and _sha256(path) != meta["sha256"]:
        raise RuntimeError(f"Checksum mismatch for {name} v{meta['version']} - artifact was modified")
    return joblib.load(path), meta


def compare_versions(name, metric):
    """Versions of `name` ranked by a metric (descending)."""
    vs = [v for v in list_versions(name) if v["metrics"] and metric in v["metrics"]]
    return sorted(vs, key=lambda v: v["metrics"][metric], reverse=True)


def set_stage(name, version, stage, deployed_by=None):
    """Move a version to a stage. Only one version per name may be 'production';
    the previous production version is archived and the change is audit-logged."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    init_db()
    with get_conn() as conn:
        target = conn.execute("SELECT * FROM model_versions WHERE name=? AND version=?",
                              (name, version)).fetchone()
        if target is None:
            raise LookupError(f"{name} v{version} does not exist")
        if stage == "production":
            conn.execute("UPDATE model_versions SET stage='archived' "
                         "WHERE name=? AND stage='production' AND id<>?", (name, target["id"]))
        conn.execute("UPDATE model_versions SET stage=? WHERE id=?", (stage, target["id"]))
        action = {"production": "deploy", "archived": "retire"}.get(stage)
        if action:
            conn.execute("INSERT INTO deployments (model_id, action, deployed_by) VALUES (?,?,?)",
                         (target["id"], action, deployed_by))
        return _row(conn.execute("SELECT * FROM model_versions WHERE id=?", (target["id"],)).fetchone())


def rollback(name, deployed_by=None):
    """Re-deploy the version that was in production before the current one."""
    init_db()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT d.model_id, mv.version FROM deployments d
            JOIN model_versions mv ON mv.id = d.model_id
            WHERE mv.name=? AND d.action IN ('deploy','rollback')
            ORDER BY d.id DESC""", (name,)).fetchall()
    seen = []
    for r in rows:
        if r["version"] not in seen:
            seen.append(r["version"])
    if len(seen) < 2:
        raise LookupError(f"No earlier production version of {name!r} to roll back to")
    meta = set_stage(name, seen[1], "production", deployed_by)
    with get_conn() as conn:  # relabel the audit entry as a rollback
        conn.execute("UPDATE deployments SET action='rollback' WHERE id="
                     "(SELECT MAX(id) FROM deployments WHERE model_id=?)", (meta["id"],))
    return meta


def deployment_history(name=None, limit=50):
    init_db()
    q = """SELECT d.id, mv.name, mv.version, mv.algorithm, d.action, d.deployed_by, d.created_at
           FROM deployments d JOIN model_versions mv ON mv.id=d.model_id"""
    args = []
    if name:
        q += " WHERE mv.name=?"
        args.append(name)
    q += " ORDER BY d.id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, args).fetchall()]
