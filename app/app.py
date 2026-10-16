from flask import Flask, render_template
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
import os

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

    ai_insight = None
    ai_error = None

    try:
        client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        prompt = f"""
You are an AI assistant inside an ML Workflow Optimizer.

The model predicts hotel booking cancellations.

Model results:
{model_results}

Give exactly 4 short recommendations:

1. Best Model
2. Overfitting
3. Performance Tradeoff
4. Next Experiment

Use 1-2 sentences for each section.
Keep the entire response under 180 words.
Do not use Markdown symbols such as **, #, or bullet points.
"""

        message = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=450,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        ai_insight = message.content[0].text

    except Exception as error:
        print("Anthropic API error:", error)

        ai_error = (
            "AI recommendations are temporarily unavailable."
        )

    return render_template(
        "ai_insights.html",
        model_results=model_results,
        best_model=best_model,
        ai_insight=ai_insight,
        ai_error=ai_error
    )


if __name__ == "__main__":
    app.run(debug=True)