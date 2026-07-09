import json
from pathlib import Path
import pickle
from typing import Any

import torch
from sklearn.model_selection import train_test_split

from phishing_ml.config import load_baseline_training_config
from phishing_ml.data.load_dataset import load_phishing_dataset
from phishing_ml.evaluation.metrics import compute_binary_classification_metrics
from phishing_ml.training.model import PhishingClassifier

DEFAULT_CONFIG_PATH = Path("configs/baseline.yaml")
DEFAULT_REPORT_PATH = Path("reports/baseline_metrics.json")


def _to_dense_array(matrix: Any) -> Any:
    return matrix.toarray()


def evaluate(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path)
    resolved_report_path = Path(report_path)
    config = load_baseline_training_config(resolved_config_path)

    data = load_phishing_dataset(config.dataset_path)
    texts = data["text"].tolist()
    labels = [int(label) for label in data["label"].tolist()]

    _, test_texts, _, test_labels = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=labels,
    )

    vectorizer_path = config.artifacts_dir / "vectorizer.pkl"
    model_path = config.artifacts_dir / "model.pt"

    with open(vectorizer_path, "rb") as file:
        vectorizer = pickle.load(file)

    features = _to_dense_array(vectorizer.transform(test_texts))
    features_tensor = torch.tensor(features, dtype=torch.float32)

    model = PhishingClassifier(input_dim=features_tensor.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(features_tensor)
        probabilities = [float(value) for value in torch.sigmoid(logits).tolist()]

    metrics = compute_binary_classification_metrics(
        labels=test_labels,
        probabilities=probabilities,
        threshold=config.threshold,
    )

    report = {
        "config_path": str(resolved_config_path),
        "dataset": {
            "path": str(config.dataset_path),
            "rows": len(data),
            "positive_labels": int(sum(labels)),
            "negative_labels": int(len(labels) - sum(labels)),
        },
        "evaluation_split": {
            "rows": len(test_labels),
            "positive_labels": int(sum(test_labels)),
            "negative_labels": int(len(test_labels) - sum(test_labels)),
            "test_size": config.test_size,
            "random_state": config.random_state,
        },
        "model": {
            "type": config.model_type,
            "artifacts_dir": str(config.artifacts_dir),
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
        },
        "metrics": metrics,
    }

    resolved_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(resolved_report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)

    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"precision={metrics['precision']:.4f}")
    print(f"recall={metrics['recall']:.4f}")
    print(f"f1={metrics['f1']:.4f}")
    print(f"saved_report={resolved_report_path}")

    return report


if __name__ == "__main__":
    evaluate()