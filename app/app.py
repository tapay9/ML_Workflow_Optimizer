from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import os
import sys
import json
import joblib

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "src" / "analysis_output"

# Load Pre-trained Model & Feature Columns
MODEL_PATH = OUTPUT_DIR / "dt_model_50_split.joblib"
COLUMNS_PATH = OUTPUT_DIR / "model_columns.joblib"
dt_model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
model_columns = joblib.load(COLUMNS_PATH) if COLUMNS_PATH.exists() else []

MODEL_RESULTS = OUTPUT_DIR / "model_comparison.csv"

# Files produced by the export cell in model_experimentation.ipynb
TUNING_RESULTS = OUTPUT_DIR / "tuning_results.csv"
DEFAULT_VS_TUNED = OUTPUT_DIR / "default_vs_tuned.csv"
FRACTION_RESULTS = OUTPUT_DIR / "fraction_results.csv"
SPLIT_RESULTS = OUTPUT_DIR / "split_results.csv"

# Make `src/` importable no matter how the app is launched
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Model registry + deployment interface (SQLite)
from deploy_routes import deploy_bp
from src.model_registry import DEFAULT_MODEL_NAME, list_versions
from src.serving import predict as serve_prediction, prediction_stats

app.register_blueprint(deploy_bp)


# Load model results (baseline comparison)
def load_model_results():
    results_df = pd.read_csv(MODEL_RESULTS)

    model_results = results_df.to_dict(orient="records")

    best_model = results_df.loc[
        results_df["test_acc"].idxmax()
    ].to_dict()

    return model_results, best_model


def _read_csv_or_none(path):
    """Return a DataFrame, or None if the experiment file hasn't been exported yet."""
    if not path.exists():
        print(f"Missing experiment file: {path.name}")
        return None
    return pd.read_csv(path)


# Load experiment results (tuning, training-size curves, split study)
def load_experiment_results():
    tuning_df = _read_csv_or_none(TUNING_RESULTS)
    compare_df = _read_csv_or_none(DEFAULT_VS_TUNED)
    fraction_df = _read_csv_or_none(FRACTION_RESULTS)
    split_df = _read_csv_or_none(SPLIT_RESULTS)

    tuning = []
    if tuning_df is not None:
        tuning = tuning_df.to_dict(orient="records")
        for row in tuning:
            row["best_params"] = json.loads(row["best_params"])

    default_vs_tuned = (
        compare_df.to_dict(orient="records") if compare_df is not None else []
    )

    # Shape for charting: fraction_curves[param_set][model] = {fractions, train, test}
    fraction_curves = {}
    if fraction_df is not None:
        for (pset, model), grp in fraction_df.groupby(["param_set", "model"]):
            grp = grp.sort_values("fraction")
            fraction_curves.setdefault(pset, {})[model] = {
                "fractions": (grp["fraction"] * 100).round().astype(int).tolist(),
                "train_acc": grp["train_acc_mean"].tolist(),
                "test_acc": grp["test_acc_mean"].tolist(),
            }

    # Best (model, split, fraction) combinations by test accuracy
    top_splits = []
    if split_df is not None:
        top_splits = (
            split_df.sort_values("test_acc", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

    return {
        "tuning": tuning,
        "default_vs_tuned": default_vs_tuned,
        "fraction_curves": fraction_curves,
        "top_splits": top_splits,
    }


# Best tuned model (by test accuracy at 100% of training data)
def get_best_tuned_model(experiment_results):
    tuned = [
        r for r in experiment_results["default_vs_tuned"]
        if r["param_set"] == "tuned"
    ]
    if not tuned:
        return None
    return max(tuned, key=lambda r: r["test_acc"])


# Dashboard
@app.route("/")
def home():
    model_results, best_model = load_model_results()
    experiment_results = load_experiment_results()

    return render_template(
        "index.html",
        model_results=model_results,
        best_model=best_model,
        best_tuned_model=get_best_tuned_model(experiment_results),
    )


# Models
@app.route("/models")
def models():
    model_results, best_model = load_model_results()
    experiment_results = load_experiment_results()

    return render_template(
        "models.html",
        model_results=model_results,
        best_model=best_model,
        tuning=experiment_results["tuning"],
        default_vs_tuned=experiment_results["default_vs_tuned"],
        registry_versions=list_versions(DEFAULT_MODEL_NAME)
    )


# Experiments
@app.route("/experiments")
def experiments():
    model_results, best_model = load_model_results()
    experiment_results = load_experiment_results()

    return render_template(
        "experiments.html",
        model_results=model_results,
        best_model=best_model,
        tuning=experiment_results["tuning"],
        default_vs_tuned=experiment_results["default_vs_tuned"],
        fraction_curves=experiment_results["fraction_curves"],
        top_splits=experiment_results["top_splits"],
    )

# Demo
@app.route("/demo")
def demo():
    model_results, best_model = load_model_results()

    return render_template(
        "demo.html",
        model_results=model_results,
        best_model=best_model
    )

# Real-Time Inference API
# Allows the Demo page to test v7, v8, or v9.
# This does NOT change which model is marked as Production.
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}

        # Model version selected on the Demo page
        selected_version = data.pop("model_version", None)

        if selected_version is not None:
            selected_version = int(selected_version)

        # Only allow the three final registered models
        if selected_version not in [7, 8, 9]:
            return jsonify({
                "error": "Please select model version v7, v8, or v9."
            }), 400

        # Score the booking using the selected registered model.
        # This does not promote/archive/change the model stage.
        result = serve_prediction(
            DEFAULT_MODEL_NAME,
            [data],
            version=selected_version,
            log=True
        )

        prediction_result = result["results"][0]

        prediction = int(
            prediction_result["prediction"]
        )

        probability_canceled = prediction_result.get(
            "probability"
        )

        if probability_canceled is None:
            probability_canceled = (
                1.0 if prediction == 1 else 0.0
            )

        # v8 is the current Production model
        stage = (
            "Production"
            if int(result["version"]) == 8
            else "Archived / Comparison"
        )

        return jsonify({
            "prediction": prediction,

            "status":
                "Canceled"
                if prediction == 1
                else "Not Canceled",

            "probability_canceled":
                float(probability_canceled),

            "probability_not_canceled":
                float(1 - probability_canceled),

            "model": result["model"],
            "version": result["version"],
            "algorithm": result["algorithm"],
            "stage": stage,
            "latency_ms": result["latency_ms"]
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

# Monitoring
@app.route("/monitoring")
def monitoring():
    model_results, best_model = load_model_results()

    live_stats = prediction_stats(DEFAULT_MODEL_NAME)

    return render_template(
        "monitoring.html",
        model_results=model_results,
        best_model=best_model,
        live_stats=live_stats
    )

@app.route("/api/monitoring/stats")
def monitoring_stats():
    live_stats = prediction_stats(DEFAULT_MODEL_NAME)

    total_predictions = sum(stat["n"] for stat in live_stats)

    return {
        "total_predictions": total_predictions,
        "versions_with_activity": len(live_stats),
        "stats": live_stats
    }


# AI Insights
@app.route("/ai-insights")
def ai_insights():
    model_results, best_model = load_model_results()

    performance_insights = None
    optimization_suggestions = None
    deployment_readiness = None
    ai_error = None

    # Final registered model versions used throughout the application.
    final_models = {
        "v7": {
            "model": "Logistic Regression",
            "train_accuracy": 0.8100,
            "test_accuracy": 0.8118,
            "macro_f1": 0.7883,
            "roc_auc": 0.8883,
            "cancellation_recall": 0.6457,
            "train_test_gap": -0.0018,
            "stage": "archived"
        },

        "v8": {
            "model": "Decision Tree",
            "configuration": {
                "max_depth": 12
            },
            "train_accuracy": 0.8508,
            "test_accuracy": 0.8322,
            "macro_f1": 0.8176,
            "roc_auc": 0.9100,
            "cancellation_recall": 0.7410,
            "train_test_gap": 0.0186,
            "stage": "production"
        },

        "v9": {
            "model": "KNN",
            "train_accuracy": 1.0000,
            "test_accuracy": 0.8068,
            "macro_f1": 0.7801,
            "roc_auc": 0.8789,
            "cancellation_recall": 0.6185,
            "train_test_gap": 0.1932,
            "stage": "archived"
        }
    }

    try:
        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        prompt = f"""
You are the AI assistant inside an ML Workflow Optimizer.

The machine learning task predicts whether a hotel booking
will be canceled.

The application has completed its final model evaluation.

ONLY analyze the following final model versions:

{json.dumps(final_models, indent=2)}

Important context:

v7 is the final Logistic Regression model.

v8 is the final Decision Tree model and is currently deployed
to production. It uses max_depth=12.

v9 is the final KNN model.

The Decision Tree v8 was selected for production because it
has the strongest overall final evaluation results.

The KNN v9 has a large train-test gap and shows substantial
overfitting.

Do NOT use or refer to earlier model versions, previous grid-search
results, previous default-vs-tuned results, or earlier experimental
accuracy values.

Do NOT claim that v8 still needs to be deployed.
It is already the current production model.

Analyze the final models and return ONLY valid JSON.

Use exactly this structure:

{{
    "performance_insights": {{
        "performance": "short insight",
        "generalization": "short insight",
        "classification": "short insight",
        "optimization": "short insight"
    }},
    "optimization_suggestions": {{
        "best_model": "short recommendation",
        "overfitting": "short recommendation",
        "performance_tradeoff": "short recommendation",
        "next_experiment": "short recommendation"
    }},
    "deployment_readiness": {{
        "recommended_model": "model name and version",
        "status": "Ready, Needs Improvement, or Not Ready",
        "reason": "short explanation",
        "risk": "short explanation",
        "next_step": "short recommendation"
    }}
}}

Requirements:

Each value should be 1-2 short sentences.

Performance:
Compare the final performance of v7, v8, and v9 using
test accuracy, Macro-F1, and ROC-AUC.

Generalization:
Discuss the train-test gaps. Explain that v7 and v8
generalize well and that v9 shows substantial overfitting.

Classification:
Compare cancellation recall and Macro-F1 across the final models.

Optimization:
Discuss opportunities for improving the final models.
Do not refer to old tuning results.

Best model:
Identify Decision Tree v8 as the strongest overall final model
based on accuracy, Macro-F1, ROC-AUC, and generalization.

Overfitting:
Identify KNN v9 as the model requiring the most attention
because of its 19.32% train-test gap.

Performance tradeoff:
Explain an important tradeoff among the final models.

Next experiment:
Recommend one reasonable future experiment based on the
final v7-v9 results.

For deployment readiness:

Recommended model:
Identify Decision Tree v8.

Status:
Evaluate the readiness of the CURRENT production model.
Status must be exactly one of:
Ready
Needs Improvement
Not Ready

Reason:
Explain the status using the final v8 evaluation metrics.

Risk:
Identify the most important remaining production risk.

Next step:
Recommend an action for monitoring or improving the model
AFTER deployment.

Remember:
Decision Tree v8 is already deployed to production.

Base every insight only on the final v7, v8, and v9 results
provided above.

Do not mention old experiments.

Do not use Markdown.

Do not include text before or after the JSON.
"""

        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ).strip()

        # Guard against accidental ```json fences
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            response_text = response_text.removeprefix("json").strip()

        ai_data = json.loads(response_text)

        performance_insights = ai_data[
            "performance_insights"
        ]

        optimization_suggestions = ai_data[
            "optimization_suggestions"
        ]

        deployment_readiness = ai_data[
            "deployment_readiness"
        ]

    except Exception as error:
        print("Anthropic API error:", error)

        ai_error = (
            "AI recommendations are temporarily unavailable."
        )

    return render_template(
        "ai_insights.html",
        model_results=model_results,
        best_model=best_model,
        performance_insights=performance_insights,
        optimization_suggestions=optimization_suggestions,
        deployment_readiness=deployment_readiness,
        ai_error=ai_error
    )

if __name__ == "__main__":
    app.run(debug=True)