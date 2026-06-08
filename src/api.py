import os
from typing import Any

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import BASE_DIR


load_dotenv(BASE_DIR / ".env")


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "genz_burnout_prediction")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/champion-model/latest")

FRONTEND_DIR = BASE_DIR / "frontend"


if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


app = FastAPI(
    title="Gen Z Burnout Prediction API",
    description="API para predição de risco de burnout usando modelo carregado pelo MLflow.",
    version="1.0.0",
)


if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static",
    )


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Dicionário com as features esperadas pelo modelo.",
    )


class PredictResponse(BaseModel):
    prediction: Any
    model_uri: str

    model_config = {
        "protected_namespaces": ()
    }


_model = None


def get_model():
    global _model

    if _model is None:
        try:
            _model = mlflow.pyfunc.load_model(MODEL_URI)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Erro ao carregar modelo pelo MLflow. "
                    f"MODEL_URI usado: {MODEL_URI}. "
                    f"Erro: {str(exc)}"
                ),
            )

    return _model


@app.get("/")
def home():
    index_path = FRONTEND_DIR / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "message": "API de predição rodando.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
    }

FEATURE_COLUMNS = [
    "Age",
    "Gender_Female",
    "Gender_Male",
    "Gender_Non-binary",
    "Student_Working_Status_Both",
    "Student_Working_Status_Student",
    "Student_Working_Status_Working",
    "Daily_Social_Media_Hours",
    "Screen_Time_Hours",
    "Night_Scrolling_Frequency",
    "Online_Gaming_Hours",
    "Content_Type_Preference_Educational",
    "Content_Type_Preference_Entertainment",
    "Content_Type_Preference_Gaming",
    "Content_Type_Preference_Lifestyle",
    "Content_Type_Preference_News",
    "Exercise_Frequency_per_Week",
    "Daily_Sleep_Hours",
    "Caffeine_Intake_Cups",
    "Study_Work_Hours_per_Day",
    "Overthinking_Score",
    "Anxiety_Score",
    "Mood_Stability_Score",
    "Social_Comparison_Index",
    "Sleep_Quality_Score",
    "Motivation_Level",
    "Emotional_Fatigue_Score",
    "Wellbeing_Index",
]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = get_model()

    try:
        features = request.features.copy()

        for column in FEATURE_COLUMNS:
            if column not in features:
                features[column] = False

        input_df = pd.DataFrame([features])
        input_df = input_df[FEATURE_COLUMNS]

        prediction = model.predict(input_df)

        if hasattr(prediction, "tolist"):
            prediction = prediction.tolist()

        if isinstance(prediction, list) and len(prediction) == 1:
            prediction = prediction[0]

        return PredictResponse(
            prediction=prediction,
            model_uri=MODEL_URI,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao realizar predição: {str(exc)}",
        )