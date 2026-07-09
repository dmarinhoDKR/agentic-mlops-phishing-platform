import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_GATE_CONFIG_PATH = Path("configs/model_quality_gate.yaml")


@dataclass(frozen=True)
class ModelQualityGateConfig:
    metrics_report_path: Path
    minimum_accuracy: float
    minimum_precision: float
    minimum_recall: float
    minimum_f1: float


REQUIRED_GATE_CONFIG_KEYS = {
    "metrics_report_path",
    "minimum_accuracy",
    "minimum_precision",
    "minimum_recall",
    "minimum_f1",
}


def load_quality_gate_config(
    path: str | Path = DEFAULT_GATE_CONFIG_PATH,
) -> ModelQualityGateConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Quality gate config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    missing_keys = REQUIRED_GATE_CONFIG_KEYS.difference(raw_config)

    if missing_keys:
        raise ValueError(f"Missing required quality gate config keys: {sorted(missing_keys)}")

    return ModelQualityGateConfig(
        metrics_report_path=Path(raw_config["metrics_report_path"]),
        minimum_accuracy=float(raw_config["minimum_accuracy"]),
        minimum_precision=float(raw_config["minimum_precision"]),
        minimum_recall=float(raw_config["minimum_recall"]),
        minimum_f1=float(raw_config["minimum_f1"]),
    )


def evaluate_quality_gate(
    config_path: str | Path = DEFAULT_GATE_CONFIG_PATH,
) -> dict[str, Any]:
    config = load_quality_gate_config(config_path)

    if not config.metrics_report_path.exists():
        raise FileNotFoundError(f"Metrics report not found: {config.metrics_report_path}")

    with open(config.metrics_report_path, "r", encoding="utf-8") as file:
        report: dict[str, Any] = json.load(file)

    metrics = report["metrics"]

    checks = [
        _build_check("accuracy", metrics["accuracy"], config.minimum_accuracy),
        _build_check("precision", metrics["precision"], config.minimum_precision),
        _build_check("recall", metrics["recall"], config.minimum_recall),
        _build_check("f1", metrics["f1"], config.minimum_f1),
    ]

    passed = all(check["passed"] for check in checks)

    return {
        "passed": passed,
        "metrics_report_path": str(config.metrics_report_path),
        "checks": checks,
    }


def _build_check(metric_name: str, actual_value: float, minimum_value: float) -> dict[str, Any]:
    return {
        "metric": metric_name,
        "actual": float(actual_value),
        "minimum": float(minimum_value),
        "passed": actual_value >= minimum_value,
    }


def main() -> None:
    result = evaluate_quality_gate()

    for check in result["checks"]:
        status = "passed" if check["passed"] else "failed"
        print(
            f"{check['metric']}={check['actual']:.4f} "
            f"minimum={check['minimum']:.4f} status={status}"
        )

    if not result["passed"]:
        raise SystemExit("Model quality gate failed")

    print("model_quality_gate=passed")


if __name__ == "__main__":
    main()