from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BaselineTrainingConfig:
    experiment_name: str
    run_name: str
    dataset_path: Path
    artifacts_dir: Path
    model_type: str
    max_features: int
    learning_rate: float
    epochs: int
    test_size: float
    random_state: int
    threshold: float


REQUIRED_BASELINE_CONFIG_KEYS = {
    "experiment_name",
    "run_name",
    "dataset_path",
    "artifacts_dir",
    "model_type",
    "max_features",
    "learning_rate",
    "epochs",
    "test_size",
    "random_state",
    "threshold",
}


def load_baseline_training_config(
    path: str | Path = "configs/baseline.yaml",
) -> BaselineTrainingConfig:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    missing_keys = REQUIRED_BASELINE_CONFIG_KEYS.difference(raw_config)

    if missing_keys:
        raise ValueError(f"Missing required config keys: {sorted(missing_keys)}")

    return BaselineTrainingConfig(
        experiment_name=str(raw_config["experiment_name"]),
        run_name=str(raw_config["run_name"]),
        dataset_path=Path(raw_config["dataset_path"]),
        artifacts_dir=Path(raw_config["artifacts_dir"]),
        model_type=str(raw_config["model_type"]),
        max_features=int(raw_config["max_features"]),
        learning_rate=float(raw_config["learning_rate"]),
        epochs=int(raw_config["epochs"]),
        test_size=float(raw_config["test_size"]),
        random_state=int(raw_config["random_state"]),
        threshold=float(raw_config["threshold"]),
    )