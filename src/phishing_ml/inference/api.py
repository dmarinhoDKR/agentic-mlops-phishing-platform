import os

from fastapi import FastAPI
from pydantic import BaseModel, Field

from phishing_ml.inference.predictor import PhishingPredictor


app = FastAPI(title="Agentic MLOps Phishing API", version="0.1.0")

artifacts_dir = os.getenv("PHISHING_ARTIFACTS_DIR", "artifacts/baseline")
predictor = PhishingPredictor(artifacts_dir=artifacts_dir)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predictor.predict(text=request.text, threshold=request.threshold)