from fastapi.testclient import TestClient

from phishing_ml.inference.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_endpoint_returns_phishing_prediction():
    response = client.post(
        "/predict",
        json={"text": "Security alert: validate your credentials within 24 hours."},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["label"] in [0, 1]
    assert payload["class_name"] in ["phishing", "legitimate"]
    assert 0.0 <= payload["phishing_probability"] <= 1.0
    assert payload["threshold"] == 0.5


def test_predict_endpoint_rejects_oversized_text():
    response = client.post("/predict", json={"text": "x" * 10_001})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "string_too_long"


def test_analyze_endpoint_returns_phishing_guidance():
    response = client.post(
        "/analyze",
        json={"text": "Urgent: verify your password immediately."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "phishing"
    assert payload["model_status"]["status"] == "approved"
    assert payload["classification"]["class_name"] == "phishing"
    assert any(
        result.get("citation")
        for result in payload["guidance"]["results"]
    )


def test_analyze_endpoint_rejects_blank_text():
    response = client.post("/analyze", json={"text": "   "})

    assert response.status_code == 422


def test_analyze_endpoint_blocks_unapproved_model(monkeypatch):
    monkeypatch.setattr(
        "phishing_ml.agents.workflow.build_model_status",
        lambda config_path: {"status": "rejected"},
    )

    response = client.post("/analyze", json={"text": "Suspicious message"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "blocked"
    assert payload["classification"] is None
    assert payload["guidance"] is None
