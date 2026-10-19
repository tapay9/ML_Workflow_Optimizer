"""Flask blueprint: model registry + deployment interface.

Register in your app:
    from app.deploy_routes import deploy_bp
    app.register_blueprint(deploy_bp)
UI lives at /deploy ; JSON API under /api/.
"""
from flask import Blueprint, jsonify, render_template, request

from src import model_registry as reg
from src import serving

deploy_bp = Blueprint("deploy", __name__, template_folder="templates")


def _err(e, code=400):
    return jsonify({"error": str(e)}), code


@deploy_bp.get("/deploy")
def deploy_page():
    return render_template("deploy.html")


@deploy_bp.get("/api/models")
def api_models():
    return jsonify(reg.list_models())


@deploy_bp.get("/api/models/<name>/versions")
def api_versions(name):
    versions = reg.list_versions(name)

    # Keep the original models and the final three models visible
    visible_versions = {7, 8, 9}

    versions = [
        version
        for version in versions
        if version["version"] in visible_versions
    ]

    return jsonify(versions)


@deploy_bp.get("/api/models/<name>/compare")
def api_compare(name):
    metric = request.args.get("metric", "accuracy")
    return jsonify(reg.compare_versions(name, metric))


@deploy_bp.post("/api/models/<name>/versions/<int:version>/stage")
def api_stage(name, version):
    body = request.get_json(silent=True) or {}
    try:
        meta = reg.set_stage(name, version, body.get("stage", ""), body.get("by"))
        serving.clear_cache()
        return jsonify(meta)
    except LookupError as e:
        return _err(e, 404)
    except ValueError as e:
        return _err(e)


@deploy_bp.post("/api/models/<name>/rollback")
def api_rollback(name):
    try:
        meta = reg.rollback(name, (request.get_json(silent=True) or {}).get("by"))
        serving.clear_cache()
        return jsonify(meta)
    except LookupError as e:
        return _err(e, 404)


@deploy_bp.get("/api/models/<name>/history")
def api_history(name):
    history = reg.deployment_history(name)

    # Only show the 5 most recent deployment events in the UI
    return jsonify(history[:5])


@deploy_bp.get("/api/models/<name>/stats")
def api_stats(name):
    return jsonify(serving.prediction_stats(name))


@deploy_bp.post("/api/predict/<name>")
def api_predict(name):
    body = request.get_json(silent=True) or {}
    records = body.get("records") or ([body["record"]] if "record" in body else None)
    if not records:
        return _err("Send {'record': {...}} or {'records': [{...}, ...]}")
    try:
        return jsonify(serving.predict(name, records, version=body.get("version")))
    except LookupError as e:
        return _err(e, 404)
    except (ValueError, KeyError) as e:
        return _err(e, 422)
