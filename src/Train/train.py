import json
import os

import joblib
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

from src.Train.preprocess import prepare_data
from src.config import BASE_DIR, get_param


load_dotenv(BASE_DIR / ".env")


def evaluate_model(y_test, y_pred):
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }


def save_metrics(metrics_path, results):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    print(f"Metrics saved at: {metrics_path}")


def log_data_info(training_params, data_info):
    mlflow.log_param("target_column", training_params.get("target_column"))
    mlflow.log_param("test_size", training_params.get("test_size"))
    mlflow.log_param("random_state", training_params.get("random_state"))

    mlflow.log_param("n_features", data_info["n_features"])
    mlflow.log_param("feature_names", json.dumps(data_info["feature_names"], ensure_ascii=False))

    mlflow.log_param("x_train_rows", data_info["x_train_rows"])
    mlflow.log_param("x_train_cols", data_info["x_train_cols"])
    mlflow.log_param("x_test_rows", data_info["x_test_rows"])
    mlflow.log_param("x_test_cols", data_info["x_test_cols"])

    mlflow.log_param(
        "full_class_distribution",
        json.dumps(data_info["full_class_distribution"], ensure_ascii=False),
    )
    mlflow.log_param(
        "y_train_distribution_before_smote",
        json.dumps(data_info["y_train_distribution_before_smote"], ensure_ascii=False),
    )
    mlflow.log_param(
        "y_train_distribution_after_smote",
        json.dumps(data_info["y_train_distribution_after_smote"], ensure_ascii=False),
    )
    mlflow.log_param(
        "y_test_distribution",
        json.dumps(data_info["y_test_distribution"], ensure_ascii=False),
    )


def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    data_info,
    model_dir,
    metrics_dir,
    training_params,
):
    model_params = get_param("model", "random_forest", default={})

    model_path = model_dir / "random_forest_model.pkl"
    metrics_path = metrics_dir / "RF_results.json"

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("model_type", "random_forest")
        log_data_info(training_params, data_info)

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        mlflow.log_metrics(results)

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        save_metrics(metrics_path, results)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")


def train_decision_tree(
    X_train,
    X_test,
    y_train,
    y_test,
    data_info,
    model_dir,
    metrics_dir,
    training_params,
):
    model_params = get_param("model", "decision_tree", default={})

    model_path = model_dir / "decision_tree_model.pkl"
    metrics_path = metrics_dir / "DT_results.json"

    with mlflow.start_run(run_name="decision_tree"):
        mlflow.log_param("model_type", "decision_tree")
        log_data_info(training_params, data_info)

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        model = DecisionTreeClassifier(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)

        mlflow.log_metrics(results)

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        save_metrics(metrics_path, results)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")


def train():
    model_dir = BASE_DIR / get_param("artifacts", "model_dir")
    metrics_dir = BASE_DIR / get_param("artifacts", "metrics_dir")
    training_params = get_param("training", default={})

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test, data_info = prepare_data()

    train_random_forest(
        X_train, X_test, y_train, y_test, data_info, model_dir, metrics_dir, training_params
    )
    train_decision_tree(
        X_train, X_test, y_train, y_test, data_info, model_dir, metrics_dir, training_params
    )


if __name__ == "__main__":
    train()