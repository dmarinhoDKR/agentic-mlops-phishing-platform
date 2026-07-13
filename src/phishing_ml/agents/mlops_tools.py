import json
from pathlib import Path
from typing import Any

from phishing_ml.evaluation.quality_gate import (
    DEFAULT_GATE_CONFIG_PATH,
    evaluate_quality_gate,
    load_quality_gate_config,
)

from phishing_ml.inference.predictor import PhishingPredictor


DEFAULT_ARTIFACTS_DIR = Path("artifacts/baseline")


REQUIRED_REPORT_SECTIONS = {
    "metrics",
    "model",
    "evaluation_split",
}

REQUIRED_METRIC_NAMES = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "threshold",
}


def load_metrics_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)

    if not report_path.exists():
        raise FileNotFoundError(f"Metrics report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as file:
        report: dict[str, Any] = json.load(file)

    missing_sections = REQUIRED_REPORT_SECTIONS.difference(report)

    if missing_sections:
        raise ValueError(
            f"Missing required report sections: {sorted(missing_sections)}"
        )

    for section_name in REQUIRED_REPORT_SECTIONS:
        if not isinstance(report[section_name], dict):
            raise ValueError(
                f"Report section must be an object: {section_name}"
            )

    metrics: dict[str, Any] = report["metrics"]
    missing_metrics = REQUIRED_METRIC_NAMES.difference(metrics)

    if missing_metrics:
        raise ValueError(
            f"Missing required metrics: {sorted(missing_metrics)}"
        )

    return report


def build_model_status(
    config_path: str | Path = DEFAULT_GATE_CONFIG_PATH,
) -> dict[str, Any]:
    gate_config = load_quality_gate_config(config_path)
    report = load_metrics_report(gate_config.metrics_report_path)
    gate_result = evaluate_quality_gate(config_path)

    metrics: dict[str, Any] = report["metrics"]
    model: dict[str, Any] = report["model"]
    evaluation_split: dict[str, Any] = report["evaluation_split"]

    failed_metrics = [
        str(check["metric"])
        for check in gate_result["checks"]
        if not check["passed"]
    ]

    return {
        "status": "approved" if gate_result["passed"] else "rejected",
        "metrics_report_path": str(gate_config.metrics_report_path),
        "model_type": str(model.get("type", "unknown")),
        "evaluation_rows": int(evaluation_split.get("rows", 0)),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "threshold": float(metrics["threshold"]),
        },
        "failed_metrics": failed_metrics,
        "quality_gate": gate_result,
    }


def format_model_status(status: dict[str, Any]) -> str:
    metrics = status["metrics"]

    lines = [
        f"Model status: {status['status'].upper()}",
        f"Model type: {status['model_type']}",
        f"Evaluation rows: {status['evaluation_rows']}",
        (
            "Metrics: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}"
        ),
    ]

    if status["failed_metrics"]:
        lines.append(
            "Failed quality checks: " + ", ".join(status["failed_metrics"])
        )
    else:
        lines.append("All configured quality checks passed.")

    return "\n".join(lines)


def classify_message(
    text: str,
    threshold: float = 0.5,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    normalized_text = text.strip()

    if not normalized_text:
        raise ValueError("Message text must not be empty")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0")

    predictor = PhishingPredictor(artifacts_dir=artifacts_dir)
    prediction = predictor.predict(
        text=normalized_text,
        threshold=threshold,
    )

    return {
        "tool_name": "classify_message",
        "input_text": normalized_text,
        "label": int(prediction["label"]),
        "class_name": str(prediction["class_name"]),
        "phishing_probability": float(
            prediction["phishing_probability"]
        ),
        "threshold": float(prediction["threshold"]),
    }
