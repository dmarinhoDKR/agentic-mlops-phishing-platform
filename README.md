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

## Roadmap

- [x] Project scaffold
- [ ] Synthetic phishing dataset
- [ ] Training pipeline
- [ ] Evaluation pipeline
- [ ] FastAPI inference service
- [ ] MLflow experiment tracking
- [ ] Docker Compose local environment
- [ ] Kubernetes manifests
- [ ] Agentic ML Ops Copilot
- [ ] Ray distributed training
- [ ] AWS deployment