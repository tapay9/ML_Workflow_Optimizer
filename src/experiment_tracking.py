import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = BASE_DIR / "src" / "analysis_output"
EXPERIMENT_LOG = EXPERIMENT_DIR / "experiment_log.json"


def load_experiments():
    """Load all previously recorded experiments."""
    if not EXPERIMENT_LOG.exists():
        return []

    try:
        with open(EXPERIMENT_LOG, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return []


def log_experiment(
    model_name,
    version,
    hyperparameters,
    metrics,
    metadata=None
):
    """
    Automatically record an ML experiment in JSON format.

    Stores:
    - model name
    - model version
    - hyperparameters
    - evaluation metrics
    - experiment metadata
    - timestamp
    """

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments()

    experiment = {
        "experiment_id": len(experiments) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "version": version,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "metadata": metadata or {}
    }

    experiments.append(experiment)

    with open(EXPERIMENT_LOG, "w", encoding="utf-8") as file:
        json.dump(experiments, file, indent=4)

    return experiment


def get_experiment_summary():
    """Return a simple summary of the stored experiments."""
    experiments = load_experiments()

    return {
        "total_experiments": len(experiments),
        "experiments": experiments
    }


if __name__ == "__main__":
    print(
        json.dumps(
            get_experiment_summary(),
            indent=4
        )
    )