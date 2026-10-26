from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["predictor"] = joblib.load("model.pkl")
    yield
    ml_model.clear()


app = FastAPI(title="Student Predictor", lifespan=lifespan)


class StudentFeatures(BaseModel):
    study_hours: float
    prev_score: float


class PredictionResult(BaseModel):
    prediction: int
    result_label: str
    confidence: float


@app.get("/")
def health():
    return {"status": "running"}


@app.post("/predict", response_model=PredictionResult)
def predict(features: StudentFeatures):
    model = ml_model["predictor"]
    X = np.array([[features.study_hours, features.prev_score]])
    pred = int(model.predict(X)[0])
    conf = round(float(model.predict_proba(X)[0][pred]), 3)
    return PredictionResult(
        prediction=pred,
        result_label="Pass" if pred == 1 else "Fail",
        confidence=conf,
    )