"""Serve predictions from whichever version is currently in production."""

import json
import time

import pandas as pd

from src.db import get_conn
from src.model_registry import get_version, load_model


_cache = {}  # (name, version) -> model


# Original columns that were combined during feature engineering.
# These do not need to be entered manually on the Deployment page.
ENGINEERED_SOURCE_COLUMNS = {
    "adults",
    "children",
    "babies",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
}


def _get_model(name, version):
    key = (name, version)

    if key not in _cache:
        _cache[key], _ = load_model(
            name,
            version=version
        )

    return _cache[key]


def clear_cache():
    _cache.clear()


def encode_records(records, meta):
    """
    Turn booking dictionaries into the exact encoded matrix
    the model was trained on.

    Deployment uses the engineered features total_guests and
    total_nights directly.

    The original guest-count and night-count columns are optional
    at deployment time. If they are not supplied, they are filled
    with 0 so the saved model still receives the feature structure
    it was trained with.

    Categorical columns are one-hot encoded and then aligned with
    the exact training feature list.
    """

    df = pd.DataFrame(records)

    features = meta.get("features")
    schema = meta.get("schema")


    # -------------------------------------------------
    # Validate engineered features
    # -------------------------------------------------

    engineered_required = [
        "total_guests",
        "total_nights"
    ]

    missing_engineered = [
        column
        for column in engineered_required
        if column not in df.columns
    ]

    if missing_engineered:
        raise ValueError(
            "Missing engineered deployment fields: "
            + ", ".join(missing_engineered)
        )


    # -------------------------------------------------
    # Encode using saved model schema
    # -------------------------------------------------

    if schema:

        numeric_columns = schema["numeric"]
        categorical_columns = schema["categorical"]

        required_columns = (
            numeric_columns
            + categorical_columns
        )


        # ---------------------------------------------
        # Add omitted source columns
        # ---------------------------------------------
        #
        # The model may have originally been trained with these
        # columns in addition to the engineered features.
        #
        # They are hidden from the Deployment interface because
        # total_guests and total_nights are the features exposed
        # for the live demo.
        #

        for column in ENGINEERED_SOURCE_COLUMNS:

            if (
                column in required_columns
                and column not in df.columns
            ):
                df[column] = 0


        # ---------------------------------------------
        # Check all remaining required fields
        # ---------------------------------------------

        missing = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required fields: {missing}"
            )


        # Keep only fields expected by the model
        df = df[required_columns].copy()


        # ---------------------------------------------
        # Convert numeric features
        # ---------------------------------------------

        for column in numeric_columns:

            try:
                df[column] = pd.to_numeric(
                    df[column]
                )

            except (ValueError, TypeError):

                raise ValueError(
                    f"Field {column!r} must be numeric"
                )


        # ---------------------------------------------
        # Reject empty values
        # ---------------------------------------------

        if df.isna().any().any():

            bad = (
                df.columns[
                    df.isna().any()
                ]
                .tolist()
            )

            raise ValueError(
                f"Empty values not allowed in: {bad}"
            )


        # ---------------------------------------------
        # One-hot encode categorical features
        # ---------------------------------------------

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            dtype=int
        )


        # ---------------------------------------------
        # Align with exact training columns
        # ---------------------------------------------

        return df.reindex(
            columns=features,
            fill_value=0
        )


    # -------------------------------------------------
    # Fallback if only feature metadata exists
    # -------------------------------------------------

    if features:

        # Add hidden source features as zero if the model
        # still expects them.
        for column in ENGINEERED_SOURCE_COLUMNS:

            if (
                column in features
                and column not in df.columns
            ):
                df[column] = 0


        missing = [
            column
            for column in features
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required features: {missing}"
            )


        return df[features]


    return df


def predict(
    name,
    records,
    version=None,
    log=True
):
    """
    Score booking records using either a specified model
    version or whichever version is currently in production.
    """

    if version:

        meta = get_version(
            name,
            version=version
        )

    else:

        meta = get_version(
            name,
            stage="production"
        )


    if meta is None:

        raise LookupError(
            f"No production version deployed for {name!r}"
        )


    # Load the deployed model
    model = _get_model(
        name,
        meta["version"]
    )


    # Prepare booking data
    df = encode_records(
        records,
        meta
    )


    # -------------------------------------------------
    # Run prediction and measure latency
    # -------------------------------------------------

    start_time = time.perf_counter()

    preds = model.predict(df)


    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(df)[:, -1]

    else:

        probs = None


    latency = (
        time.perf_counter() - start_time
    ) * 1000


    # -------------------------------------------------
    # Format prediction results
    # -------------------------------------------------

    out = []


    for i, prediction in enumerate(preds):

        item = {
            "prediction":
                prediction.item()
                if hasattr(prediction, "item")
                else prediction
        }


        if probs is not None:

            item["probability"] = float(
                probs[i]
            )


        out.append(item)


    # -------------------------------------------------
    # Log prediction event to SQLite
    # -------------------------------------------------

    if log:

        with get_conn() as conn:

            conn.executemany(
                """
                INSERT INTO predictions
                (
                    model_id,
                    input_json,
                    output_json,
                    latency_ms
                )
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        meta["id"],

                        json.dumps(
                            records[i],
                            default=str
                        ),

                        json.dumps(
                            out[i]
                        ),

                        latency / len(out)
                    )

                    for i in range(len(out))
                ]
            )


    # -------------------------------------------------
    # Return prediction information
    # -------------------------------------------------

    return {
        "model": name,
        "version": meta["version"],
        "algorithm": meta["algorithm"],
        "results": out,
        "latency_ms": round(
            latency,
            2
        )
    }


def prediction_stats(name):
    """
    Return prediction activity for the final registered model
    versions used by the real-time Monitoring dashboard.
    """

    with get_conn() as conn:

        rows = conn.execute(
            """
            SELECT
                mv.version,
                mv.stage,
                COUNT(*) AS n,
                ROUND(
                    AVG(p.latency_ms),
                    3
                ) AS avg_latency_ms

            FROM predictions p

            JOIN model_versions mv
                ON mv.id = p.model_id

            WHERE mv.name = ?
              AND mv.version IN (7, 8, 9)

            GROUP BY
                mv.version,
                mv.stage

            ORDER BY
                mv.version DESC
            """,
            (name,)
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]