import json
import os

import joblib
import mlflow
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow.sklearn

from src.Train.preprocess import prepare_data
from src.config import BASE_DIR, get_param


load_dotenv(BASE_DIR / ".env")


def evaluate_model(y_test, y_pred):
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
    }


def save_metrics(metrics_path, results):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Metrics saved at: {metrics_path}")


def log_data_info(training_params, data_info):
    mlflow.log_param("target_column", training_params.get("target_column"))
    mlflow.log_param("test_size", training_params.get("test_size"))
    mlflow.log_param("random_state", training_params.get("random_state"))

    mlflow.log_param("n_features", data_info["n_features"])
    mlflow.log_param(
        "feature_names",
        json.dumps(data_info["feature_names"], ensure_ascii=False),
    )

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
        json.dumps(
            data_info["y_train_distribution_before_smote"],
            ensure_ascii=False,
        ),
    )
    mlflow.log_param(
        "y_train_distribution_after_smote",
        json.dumps(
            data_info["y_train_distribution_after_smote"],
            ensure_ascii=False,
        ),
    )
    mlflow.log_param(
        "y_test_distribution",
        json.dumps(data_info["y_test_distribution"], ensure_ascii=False),
    )


def log_model_artifacts(model_path, metrics_path):
    mlflow.log_artifact(str(model_path), artifact_path="models")
    mlflow.log_artifact(str(metrics_path), artifact_path="metrics")


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

    model_filename = get_param(
        "artifacts",
        "model_filename",
        default="random_forest_model.pkl",
    )
    metrics_filename = get_param(
        "artifacts",
        "metrics_filename",
        default="RF_results.json",
    )

    model_path = model_dir / model_filename
    metrics_path = metrics_dir / metrics_filename

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
        log_model_artifacts(model_path, metrics_path)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="champion-model",
            registered_model_name="champion-model",
        )

        feature_names_path = metrics_dir / "feature_names.json"
        with open(feature_names_path, "w", encoding="utf-8") as file:
            json.dump(data_info["feature_names"], file, indent=4, ensure_ascii=False)

        mlflow.log_artifact(str(feature_names_path), artifact_path="metadata")

        print("Random Forest training completed.")
        print(f"Model saved at: {model_path}")
        print(f"Metrics: {results}")


def train():
    model_dir = BASE_DIR / get_param("artifacts", "model_dir")
    metrics_dir = BASE_DIR / get_param("artifacts", "metrics_dir")
    training_params = get_param("training", default={})

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "genz_burnout_prediction")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {experiment_name}")

    X_train, X_test, y_train, y_test, data_info = prepare_data()

    train_random_forest(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        data_info=data_info,
        model_dir=model_dir,
        metrics_dir=metrics_dir,
        training_params=training_params,
    )


if __name__ == "__main__":
    train()