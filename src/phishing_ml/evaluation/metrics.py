from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_binary_classification_metrics(
    labels: list[int],
    probabilities: list[float],
    threshold: float,
) -> dict[str, Any]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }