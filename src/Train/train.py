import json
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.Train.preprocess import prepare_data


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR")
METRICS_DIR = BASE_DIR / os.getenv("METRICS_DIR")

MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"
METRICS_PATH = METRICS_DIR / "RF_results.json"


def train():
    random_state = int(os.getenv("RANDOM_STATE"))

    X_train, X_test, y_train, y_test = prepare_data()

    print("Training Random Forest model...")
    model = RandomForestClassifier(random_state=random_state)
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

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    print(f"Metrics saved at: {METRICS_PATH}")


if __name__ == "__main__":
    train()