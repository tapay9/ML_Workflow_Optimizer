"""Tests for the SQLite model registry, versioning and deployment API.
Run from the project root:  python -m pytest tests
"""
import os
import sys
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["MLWO_DB_PATH"] = os.path.join(_tmp, "test.db")
os.environ["MLWO_MODELS_DIR"] = os.path.join(_tmp, "models")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "app")]

import pandas as pd
import pytest
from flask import Flask
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from deploy_routes import deploy_bp
from src import model_registry as reg
from src import serving

raw = pd.DataFrame({"lead_time": [1, 50, 200, 10, 300, 5, 120, 80] * 4,
                    "city": ["Delhi", "Goa", "Pune", "Delhi"] * 8})
y = pd.Series([0, 0, 1, 0, 1, 0, 1, 1] * 4)
X = pd.get_dummies(raw, columns=["city"], dtype=int)
groups = {c: ("city" if c.startswith("city_") else c) for c in X.columns}
schema = reg.build_input_schema(X, groups)
NAME = "hotel_cancellation"


@pytest.fixture(scope="module")
def versions():
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    dt = DecisionTreeClassifier(random_state=0).fit(X, y)
    return [reg.register_model(lr, NAME, {"test_acc": .70, "roc_auc": .7}, features=list(X.columns), schema=schema),
            reg.register_model(dt, NAME, {"test_acc": .90, "roc_auc": .9}, features=list(X.columns), schema=schema)]


def test_versions_and_algorithm(versions):
    assert [v["version"] for v in versions] == [1, 2]
    assert [v["algorithm"] for v in versions] == ["LogisticRegression", "DecisionTreeClassifier"]
    assert reg.compare_versions(NAME, "roc_auc")[0]["version"] == 2


def test_schema_example(versions):
    assert schema["numeric"] == ["lead_time"] and schema["categorical"] == ["city"]
    assert set(schema["example"]) == {"lead_time", "city"}


def test_promote_archive_rollback(versions):
    reg.set_stage(NAME, 1, "production")
    reg.set_stage(NAME, 2, "production")
    assert reg.get_version(NAME, 1)["stage"] == "archived"
    assert reg.rollback(NAME)["version"] == 1
    assert reg.get_version(NAME, stage="production")["version"] == 1


def test_encoding_matches_training(versions):
    out = serving.predict(NAME, raw.head(8).to_dict("records"), version=2, log=False)
    model, _ = reg.load_model(NAME, 2)
    assert [r["prediction"] for r in out["results"]] == list(model.predict(X.head(8)))


def test_checksum_detects_tampering(versions):
    meta = reg.get_version(NAME, 2)
    with open(os.path.join(reg.MODELS_DIR, meta["artifact_path"]), "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError):
        reg.load_model(NAME, 2)


def test_api_end_to_end(versions):
    app = Flask(__name__, template_folder=os.path.join(ROOT, "app", "templates"))
    app.add_url_rule("/", "home", lambda: "")
    for ep in ("models", "experiments", "monitoring", "ai_insights", "demo"):
        app.add_url_rule(f"/{ep}", ep, lambda: "")
    app.register_blueprint(deploy_bp)
    c = app.test_client()
    assert c.get("/deploy").status_code == 200
    assert c.post(f"/api/models/{NAME}/versions/1/stage", json={"stage": "production"}).status_code == 200
    ok = c.post(f"/api/predict/{NAME}", json={"record": {"lead_time": 250, "city": "Goa", "is_canceled": 1}})
    assert ok.status_code == 200 and ok.json["version"] == 1 and "probability" in ok.json["results"][0]
    assert c.post(f"/api/predict/{NAME}", json={"record": {"lead_time": 1}}).status_code == 422
    assert c.post(f"/api/predict/{NAME}", json={"record": {"lead_time": "x", "city": "Goa"}}).status_code == 422
    assert c.post(f"/api/models/{NAME}/versions/1/stage", json={"stage": "bogus"}).status_code == 400
    assert c.post("/api/models/nope/versions/9/stage", json={"stage": "staging"}).status_code == 404
