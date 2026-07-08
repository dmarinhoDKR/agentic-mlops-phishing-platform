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