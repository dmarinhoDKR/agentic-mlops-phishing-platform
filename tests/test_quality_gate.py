import json

import pytest
import yaml

from phishing_ml.evaluation.quality_gate import evaluate_quality_gate, load_quality_gate_config


def test_load_quality_gate_config(tmp_path):
    report_path = tmp_path / "metrics.json"
    config_path = tmp_path / "gate.yaml"
    config_path.write_text(
        f"""
metrics_report_path: {report_path}
minimum_accuracy: 0.85
minimum_precision: 0.80
minimum_recall: 0.80
minimum_f1: 0.80
""",
        encoding="utf-8",
    )

    config = load_quality_gate_config(config_path)

    assert config.metrics_report_path == report_path
    assert config.minimum_accuracy == 0.85
    assert config.minimum_precision == 0.80
    assert config.minimum_recall == 0.80
    assert config.minimum_f1 == 0.80


def test_evaluate_quality_gate_passes_when_metrics_meet_thresholds(tmp_path):
    report_path = tmp_path / "metrics.json"
    config_path = tmp_path / "gate.yaml"

    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.90,
                    "recall": 0.89,
                    "f1": 0.88,
                }
            }
        ),
        encoding="utf-8",
    )
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

    result = evaluate_quality_gate(config_path)

    assert result["passed"] is True
    assert all(check["passed"] for check in result["checks"])


def test_evaluate_quality_gate_fails_when_metric_is_below_threshold(tmp_path):
    report_path = tmp_path / "metrics.json"
    config_path = tmp_path / "gate.yaml"

    report_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "accuracy": 0.91,
                    "precision": 0.90,
                    "recall": 0.70,
                    "f1": 0.88,
                }
            }
        ),
        encoding="utf-8",
    )
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

    result = evaluate_quality_gate(config_path)

    assert result["passed"] is False

    failed_checks = [check for check in result["checks"] if not check["passed"]]
    assert failed_checks == [
        {
            "metric": "recall",
            "actual": 0.70,
            "minimum": 0.80,
            "passed": False,
        },
    ]


def test_load_quality_gate_config_requires_all_keys(tmp_path):
    config_path = tmp_path / "invalid_gate.yaml"
    config_path.write_text("minimum_accuracy: 0.85\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required quality gate config keys"):
        load_quality_gate_config(config_path)