# Agentic MLOps Phishing Platform

A production-style MLOps platform for phishing and incident detection.

## Goals

- Train and evaluate phishing detection models with PyTorch.
- Track experiments with MLflow.
- Serve predictions through a FastAPI inference service.
- Containerize the system with Docker.
- Deploy training and inference workloads on Kubernetes.
- Add an agentic AI assistant to analyze model metrics, logs, and regressions.
- Extend the platform to distributed training with Ray and cloud infrastructure on AWS.

## Planned Architecture

data -> preprocessing -> training -> evaluation -> model registry -> inference API -> monitoring -> agentic analysis

## Running Locally

Train the baseline model:

```bash
python -m phishing_ml.training.train_baseline
```

Run the API locally:

```bash
uvicorn phishing_ml.inference.api:app --reload
```

Test the API:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Security alert: validate your credentials within 24 hours."}'
```

## Running With Docker Compose

Build and run the inference API:

```bash
docker compose up --build api
```

The API expects model artifacts at:

```text
artifacts/baseline/model.pt
artifacts/baseline/vectorizer.pkl
```

## Roadmap

- [x] Project scaffold
- [x] Synthetic phishing dataset
- [x] Training pipeline
- [x] Evaluation pipeline
- [x] FastAPI inference service
- [x] MLflow experiment tracking
- [x] Docker Compose local environment
- [ ] Kubernetes manifests
- [ ] Agentic ML Ops Copilot
- [ ] Ray distributed training
- [ ] AWS deployment