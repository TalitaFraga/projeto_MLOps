import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.Train.preprocess import prepare_data
from src.config import BASE_DIR, get_param


def train():
    model_dir = BASE_DIR / get_param("artifacts", "model_dir")
    metrics_dir = BASE_DIR / get_param("artifacts", "metrics_dir")

    model_path = model_dir / get_param("artifacts", "model_filename")
    metrics_path = metrics_dir / get_param("artifacts", "metrics_filename")

    model_params = get_param("model", "random_forest", default={})

    X_train, X_test, y_train, y_test = prepare_data()

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

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    print(f"Metrics saved at: {metrics_path}")


if __name__ == "__main__":
    train()