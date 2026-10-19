"""Serve predictions from whichever version is currently in production."""
import json
import time

import pandas as pd

from src.db import get_conn
from src.model_registry import get_version, load_model

_cache = {}  # (name, version) -> model


def _get_model(name, version):
    key = (name, version)
    if key not in _cache:
        _cache[key], _ = load_model(name, version=version)
    return _cache[key]


def clear_cache():
    _cache.clear()


def encode_records(records, meta):
    """Turn raw booking dicts into the exact encoded matrix the model was trained on.

    Mirrors the training pipeline: categorical columns are one-hot encoded
    (pd.get_dummies), then columns are aligned to the training feature list.
    Unknown category levels become all-zero (same as handle_unknown='ignore').
    Extra keys such as is_canceled or arrival_date_month are ignored.
    """
    df = pd.DataFrame(records)
    features, schema = meta.get("features"), meta.get("schema")

    if schema:
        need = schema["numeric"] + schema["categorical"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        df = df[need].copy()
        for c in schema["numeric"]:
            try:
                df[c] = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                raise ValueError(f"Field {c!r} must be numeric")
        if df.isna().any().any():
            bad = df.columns[df.isna().any()].tolist()
            raise ValueError(f"Empty values not allowed in: {bad}")
        df = pd.get_dummies(df, columns=schema["categorical"], dtype=int)
        return df.reindex(columns=features, fill_value=0)

    if features:
        missing = [c for c in features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return df[features]
    return df


def predict(name, records, version=None, log=True):
    """Score a list of raw feature dicts with the production (or a pinned) version."""
    meta = get_version(name, version=version) if version else get_version(name, stage="production")
    if meta is None:
        raise LookupError(f"No production version deployed for {name!r}")
    model = _get_model(name, meta["version"])
    df = encode_records(records, meta)

    t0 = time.perf_counter()
    preds = model.predict(df)
    probs = model.predict_proba(df)[:, -1] if hasattr(model, "predict_proba") else None
    latency = (time.perf_counter() - t0) * 1000

    out = []
    for i, p in enumerate(preds):
        item = {"prediction": p.item() if hasattr(p, "item") else p}
        if probs is not None:
            item["probability"] = float(probs[i])
        out.append(item)

    if log:
        with get_conn() as conn:
            conn.executemany(
                "INSERT INTO predictions (model_id, input_json, output_json, latency_ms) VALUES (?,?,?,?)",
                [(meta["id"], json.dumps(records[i], default=str), json.dumps(out[i]),
                  latency / len(out)) for i in range(len(out))])
    return {"model": name, "version": meta["version"], "algorithm": meta["algorithm"],
            "results": out, "latency_ms": round(latency, 2)}


def prediction_stats(name):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT mv.version, COUNT(*) AS n, ROUND(AVG(p.latency_ms),3) AS avg_latency_ms
            FROM predictions p JOIN model_versions mv ON mv.id=p.model_id
            WHERE mv.name=? GROUP BY mv.version ORDER BY mv.version DESC""", (name,)).fetchall()
        return [dict(r) for r in rows]
