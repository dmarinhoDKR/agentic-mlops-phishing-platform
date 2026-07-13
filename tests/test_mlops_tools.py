import json
from pathlib import Path

import pytest
import yaml

from phishing_ml.agents.mlops_tools import (
    build_model_status,
    classify_message,
    format_model_status,
    load_metrics_report,
)


def _write_metrics_report(
    tmp_path: Path,
    recall: float = 0.89,
) -> Path:
    report_path = tmp_path / "metrics.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.90,
                    "recall": recall,
                    "f1": 0.88,
                    "threshold": 0.5,
                },
                "model": {
                    "type": "tfidf_pytorch_linear",
                },
                "evaluation_split": {
                    "rows": 20,
                },
            }
        ),
        encoding="utf-8",
    )

    return report_path


def _write_quality_gate_config(
    tmp_path: Path,
    report_path: Path,
) -> Path:
    config_path = tmp_path / "quality_gate.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "metrics_report_path": str(report_path),
                "minimum_accuracy": 0.85,
                "minimum_precision": 0.80,
                "minimum_recall": 0.80,
                "minimum_f1": 0.80,
            }
        ),
        encoding="utf-8",
    )

    return config_path


def test_load_metrics_report(tmp_path):
    report_path = _write_metrics_report(tmp_path)

    report = load_metrics_report(report_path)

    assert report["metrics"]["accuracy"] == 0.91
    assert report["model"]["type"] == "tfidf_pytorch_linear"
    assert report["evaluation_split"]["rows"] == 20


def test_load_metrics_report_requires_expected_sections(tmp_path):
    report_path = tmp_path / "invalid_metrics.json"
    report_path.write_text(
        json.dumps({"metrics": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required report sections"):
        load_metrics_report(report_path)


def test_build_model_status_approves_model(tmp_path):
    report_path = _write_metrics_report(tmp_path)
    config_path = _write_quality_gate_config(tmp_path, report_path)

    status = build_model_status(config_path)
    summary = format_model_status(status)

    assert status["status"] == "approved"
    assert status["failed_metrics"] == []
    assert status["quality_gate"]["passed"] is True
    assert "Model status: APPROVED" in summary
    assert "All configured quality checks passed." in summary


def test_build_model_status_rejects_model(tmp_path):
    report_path = _write_metrics_report(tmp_path, recall=0.70)
    config_path = _write_quality_gate_config(tmp_path, report_path)

    status = build_model_status(config_path)
    summary = format_model_status(status)

    assert status["status"] == "rejected"
    assert status["failed_metrics"] == ["recall"]
    assert status["quality_gate"]["passed"] is False
    assert "Model status: REJECTED" in summary
    assert "Failed quality checks: recall" in summary


def test_classify_message_returns_structured_prediction(
    monkeypatch,
    tmp_path,
):
    calls = {}

    class FakePredictor:
        def __init__(self, artifacts_dir):
            calls["artifacts_dir"] = artifacts_dir

        def predict(self, text, threshold):
            calls["text"] = text
            calls["threshold"] = threshold

            return {
                "label": 1,
                "class_name": "phishing",
                "phishing_probability": 0.92,
                "threshold": threshold,
            }

    monkeypatch.setattr(
        "phishing_ml.agents.mlops_tools.PhishingPredictor",
        FakePredictor,
    )

    result = classify_message(
        text="  Urgent password reset required.  ",
        threshold=0.6,
        artifacts_dir=tmp_path,
    )

    assert calls == {
        "artifacts_dir": tmp_path,
        "text": "Urgent password reset required.",
        "threshold": 0.6,
    }
    assert result == {
        "tool_name": "classify_message",
        "input_text": "Urgent password reset required.",
        "label": 1,
        "class_name": "phishing",
        "phishing_probability": 0.92,
        "threshold": 0.6,
    }


def test_classify_message_rejects_empty_text():
    with pytest.raises(ValueError, match="Message text must not be empty"):
        classify_message("   ")


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_classify_message_rejects_invalid_threshold(threshold):
    with pytest.raises(
        ValueError,
        match="Threshold must be between 0.0 and 1.0",
    ):
        classify_message("Example message", threshold=threshold)
