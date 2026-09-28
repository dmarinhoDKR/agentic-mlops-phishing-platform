import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from phishing_ml.agents.workflow import run_incident_workflow
from phishing_ml.inference.predictor import PhishingPredictor


app = FastAPI(title="Agentic MLOps Phishing API", version="0.1.0")
frontend_dir = Path(__file__).resolve().parents[3] / "frontend"
app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="ui")

artifacts_dir = os.getenv("PHISHING_ARTIFACTS_DIR", "artifacts/baseline")
predictor = PhishingPredictor(artifacts_dir=artifacts_dir)


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(..., min_length=1, max_length=10_000)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    tool_name: Literal["analyze_incident"]
    outcome: Literal["blocked", "legitimate", "phishing"]
    reason: str
    model_status: dict[str, Any]
    classification: dict[str, Any] | None
    guidance: dict[str, Any] | None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predictor.predict(text=request.text, threshold=request.threshold)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> dict:
    return run_incident_workflow(
        message=request.text,
        threshold=request.threshold,
        artifacts_dir=artifacts_dir,
    )
