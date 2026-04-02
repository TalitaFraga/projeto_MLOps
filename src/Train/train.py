import json
import os

import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.Train.preprocess import prepare_data
from src.config import BASE_DIR, get_param


load_dotenv(BASE_DIR / ".env")


def train():
    model_dir = BASE_DIR / get_param("artifacts", "model_dir")
    metrics_dir = BASE_DIR / get_param("artifacts", "metrics_dir")

    model_path = model_dir / get_param("artifacts", "model_filename")
    metrics_path = metrics_dir / get_param("artifacts", "metrics_filename")

    model_params = get_param("model", "random_forest", default={})
    training_params = get_param("training", default={})

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "default_experiment")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test = prepare_data()

    with mlflow.start_run():
        mlflow.log_param("target_column", training_params.get("target_column"))
        mlflow.log_param("test_size", training_params.get("test_size"))
        mlflow.log_param("random_state", training_params.get("random_state"))

        for param_name, param_value in model_params.items():
            mlflow.log_param(param_name, param_value)

        print("Training Random Forest model...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        print("Model training completed.")

        y_pred = model.predict(X_test)

        results = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
            "f1_score": round(f1_score(y_test, y_pred, average="weighted"), 4),
        }
        print("Model evaluation completed.")

        mlflow.log_metrics(results)

        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model saved at: {model_path}")

        metrics_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        print(f"Metrics saved at: {metrics_path}")

        mlflow.log_artifact(str(metrics_path))
        mlflow.sklearn.log_model(model, name="model")

        print("MLflow logging completed.")


if __name__ == "__main__":
    train()