import json
from pathlib import Path

import pytest
import yaml

from phishing_ml.agents.mlops_tools import (
    build_model_status,
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
