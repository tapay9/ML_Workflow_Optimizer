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
MODEL_RESULTS = BASE_DIR / "src" / "analysis_output" / "model_comparison.csv"


# Load model results
def load_model_results():
    results_df = pd.read_csv(MODEL_RESULTS)

    model_results = results_df.to_dict(orient="records")

    best_model = results_df.loc[
        results_df["test_acc"].idxmax()
    ].to_dict()

    return model_results, best_model


# Dashboard
@app.route("/")
def home():
    model_results, best_model = load_model_results()

    return render_template(
        "index.html",
        model_results=model_results,
        best_model=best_model
    )


# Models
@app.route("/models")
def models():
    model_results, best_model = load_model_results()

    return render_template(
        "models.html",
        model_results=model_results,
        best_model=best_model
    )


# Experiments
@app.route("/experiments")
def experiments():
    model_results, best_model = load_model_results()

    return render_template(
        "experiments.html",
        model_results=model_results,
        best_model=best_model
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

    performance_insights = None
    optimization_suggestions = None
    ai_error = None

    try:
        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        prompt = f"""
You are the AI assistant inside an ML Workflow Optimizer.

The machine learning task predicts whether a hotel booking
will be canceled.

Here are the current model results:

{model_results}

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
    }}
}}

Requirements:

Each value should be 1-2 short sentences.

Performance should explain the strongest model performance.

Generalization should identify whether any model appears
to be overfitting or generalizing well.

Classification should discuss useful classification metrics
such as ROC-AUC, precision, recall, or F1.

Optimization should explain what area could benefit from
hyperparameter tuning.

Best model should recommend the strongest overall model,
considering more than accuracy alone.

Overfitting should explain which model needs attention.

Performance tradeoff should explain an important tradeoff
between the models.

Next experiment should recommend one useful next experiment
or tuning step.

Base every insight only on the model results provided.

Do not use Markdown.

Do not include text before or after the JSON.
"""

        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=1200,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        response_text = message.content[0].text.strip()

        ai_data = json.loads(response_text)

        performance_insights = ai_data["performance_insights"]

        optimization_suggestions = ai_data[
            "optimization_suggestions"
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
        ai_error=ai_error
    )


if __name__ == "__main__":
    app.run(debug=True) 