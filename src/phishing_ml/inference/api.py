import os

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from phishing_ml.inference.predictor import PhishingPredictor
from phishing_ml.agents.workflow import run_incident_workflow


app = FastAPI(title="Agentic MLOps Phishing API", version="0.1.0")

artifacts_dir = os.getenv("PHISHING_ARTIFACTS_DIR", "artifacts/baseline")
predictor = PhishingPredictor(artifacts_dir=artifacts_dir)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predictor.predict(text=request.text, threshold=request.threshold)


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    return run_incident_workflow(
        message=request.text,
        threshold=request.threshold,
        artifacts_dir=artifacts_dir,
    )
