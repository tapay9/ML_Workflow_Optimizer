from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from src.model_analysis import (
    build_input_schema,
    load_data,
    make_models,
    score_all,
)
from src.model_registry import (
    DEFAULT_MODEL_NAME,
    register_model,
)


# -------------------------------------------------
# Project paths
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "hotel_bookings_cleaned.csv"
)


# -------------------------------------------------
# Allowed AI-controlled hyperparameters
# -------------------------------------------------
#
# We only allow parameters that make sense for the
# three models used by ModelPilot. This prevents an
# AI response from passing arbitrary parameters into
# sklearn.
# -------------------------------------------------

ALLOWED_PARAMETERS = {
    "Logistic Regression": {
        "C",
        "max_iter",
    },

    "Decision Tree": {
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "criterion",
    },

    "KNN": {
        "n_neighbors",
        "weights",
        "p",
    },
}


# -------------------------------------------------
# Validate AI suggestion
# -------------------------------------------------

def validate_suggestion(model_name, parameters):

    if model_name not in ALLOWED_PARAMETERS:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if not isinstance(parameters, dict):
        raise ValueError(
            "parameters must be a JSON object."
        )

    allowed = ALLOWED_PARAMETERS[model_name]

    invalid = [
        parameter
        for parameter in parameters
        if parameter not in allowed
    ]

    if invalid:
        raise ValueError(
            "Unsupported parameter(s) for "
            f"{model_name}: {', '.join(invalid)}"
        )

    if not parameters:
        raise ValueError(
            "At least one hyperparameter is required."
        )

    return True


# -------------------------------------------------
# Apply suggested parameters
# -------------------------------------------------

def apply_parameters(model_name, model, parameters):

    model = clone(model)

    if model_name == "Logistic Regression":

        mapped_parameters = {
            f"logisticregression__{key}": value
            for key, value in parameters.items()
        }

        model.set_params(**mapped_parameters)

    elif model_name == "KNN":

        mapped_parameters = {
            f"kneighborsclassifier__{key}": value
            for key, value in parameters.items()
        }

        model.set_params(**mapped_parameters)

    else:

        model.set_params(**parameters)

    return model


# -------------------------------------------------
# Retrain model from accepted AI suggestion
# -------------------------------------------------

def retrain_from_ai_suggestion(
    model_name,
    parameters,
    seed=42,
    test_size=0.20,
):

    validate_suggestion(
        model_name,
        parameters
    )

    # ---------------------------------------------
    # Load the same prepared dataset used by the
    # original ModelPilot training workflow.
    # ---------------------------------------------

    X, y, groups, _ = load_data(
        DATA_PATH,
        max_rows=0,
        seed=seed,
    )

    # ---------------------------------------------
    # Use a reproducible stratified train/test split.
    # ---------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )
    )

    # ---------------------------------------------
    # Build the original model configuration.
    # ---------------------------------------------

    models = make_models(
        seed=seed,
        skip_knn=False,
    )

    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' was not found."
        )

    original_model = models[model_name]

    # ---------------------------------------------
    # Apply the accepted AI hyperparameters.
    # ---------------------------------------------

    candidate_model = apply_parameters(
        model_name,
        original_model,
        parameters,
    )

    # ---------------------------------------------
    # Train candidate model.
    # ---------------------------------------------

    candidate_model.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------
    # Evaluate candidate model.
    # ---------------------------------------------

    train_predictions = (
        candidate_model.predict(X_train)
    )

    test_predictions = (
        candidate_model.predict(X_test)
    )

    test_probabilities = (
        candidate_model.predict_proba(X_test)[:, 1]
    )

    train_accuracy = (
        train_predictions == y_train
    ).mean()

    scores = score_all(
        y_test,
        test_predictions,
        test_probabilities,
    )

    metrics = {
        "train_acc": float(train_accuracy),
        "test_acc": float(
            scores["accuracy"]
        ),
        "macro_f1": float(
            scores["macro_f1"]
        ),
        "roc_auc": float(
            scores["roc_auc"]
        ),
        "overfit_gap": float(
            train_accuracy
            - scores["accuracy"]
        ),
    }

    # ---------------------------------------------
    # Register as a NEW model version.
    #
    # Important:
    # Retraining does NOT automatically replace the
    # Production model. The new version enters the
    # registry and can be reviewed/deployed afterward.
    # ---------------------------------------------

    schema = build_input_schema(
    X,
    groups,
    
    )
    
    registered = register_model(
        model=candidate_model,
        name=DEFAULT_MODEL_NAME,
        metrics=metrics,
        features=list(X.columns),
        schema=schema,
        algorithm=model_name,
        notes=(
            "AI-triggered retraining | "
            f"accepted parameters: {parameters}"
        ),
    )

    # ---------------------------------------------
    # Return result to Flask/API.
    # ---------------------------------------------

    return {
        "success": True,
        "message": (
            f"{model_name} successfully retrained "
            "using the accepted AI suggestion."
        ),
        "model": model_name,
        "parameters": parameters,
        "version": registered["version"],
        "metrics": metrics,
    }