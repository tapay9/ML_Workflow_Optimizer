from flask import Flask, render_template
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import os
import json

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "src" / "analysis_output"

MODEL_RESULTS = OUTPUT_DIR / "model_comparison.csv"

# Files produced by the export cell in model_experimentation.ipynb
TUNING_RESULTS = OUTPUT_DIR / "tuning_results.csv"
DEFAULT_VS_TUNED = OUTPUT_DIR / "default_vs_tuned.csv"
FRACTION_RESULTS = OUTPUT_DIR / "fraction_results.csv"
SPLIT_RESULTS = OUTPUT_DIR / "split_results.csv"


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


# Monitoring
@app.route("/monitoring")
def monitoring():
    model_results, best_model = load_model_results()

    return render_template(
        "monitoring.html",
        model_results=model_results,
        best_model=best_model
    )


# AI Insights
@app.route("/ai-insights")
def ai_insights():
    model_results, best_model = load_model_results()
    experiment_results = load_experiment_results()

    performance_insights = None
    optimization_suggestions = None
    deployment_readiness = None
    ai_error = None

    try:
        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        prompt = f"""
You are the AI assistant inside an ML Workflow Optimizer.

The machine learning task predicts whether a hotel booking
will be canceled.

Here are the baseline model results:

{model_results}

Here are the hyperparameter tuning results (grid search with
cross-validation, best parameters per model):

{experiment_results["tuning"]}

Here is a comparison of default vs tuned hyperparameters on the
held-out test set, with train-test gap (higher gap = more overfitting):

{experiment_results["default_vs_tuned"]}

Here are the best combinations from the train/test split and
training-set-size experiments:

{experiment_results["top_splits"]}

Analyze the results and return ONLY valid JSON.

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
        "recommended_model": "model name",
        "status": "Ready, Needs Improvement, or Not Ready",
        "reason": "short explanation",
        "risk": "short explanation",
        "next_step": "short recommendation"
    }}
}}

Requirements:

Each value should be 1-2 short sentences.

Performance should explain the strongest model performance,
including any gain from hyperparameter tuning.

Generalization should identify whether any model appears
to be overfitting or generalizing well, using the train-test gap.

Classification should discuss useful classification metrics
such as F1, precision, or recall.

Optimization should explain what tuning changed and what
area could still benefit from further tuning.

Best model should recommend the strongest overall model,
considering more than accuracy alone.

Overfitting should explain which model needs attention.

Performance tradeoff should explain an important tradeoff
between the models, such as accuracy, F1, or fit time.

Next experiment should recommend one useful next experiment
or tuning step.

For deployment readiness:

Recommended model should identify the strongest candidate
for deployment based on the available evaluation results.

Status must be exactly one of:
Ready
Needs Improvement
Not Ready

Reason should explain why the recommended model received
that deployment-readiness status.

Risk should identify the most important concern before
deployment.

Next step should recommend one action that should be taken
before deployment.

Do not claim that a model is already deployed.

Base every insight only on the results provided.

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

        response_text = message.content[0].text.strip()

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