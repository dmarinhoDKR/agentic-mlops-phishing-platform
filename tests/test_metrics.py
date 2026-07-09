from phishing_ml.evaluation.metrics import compute_binary_classification_metrics


def test_compute_binary_classification_metrics():
    metrics = compute_binary_classification_metrics(
        labels=[0, 1, 1, 0],
        probabilities=[0.1, 0.9, 0.4, 0.8],
        threshold=0.5,
    )

    assert metrics["threshold"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["confusion_matrix"] == {
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_positive": 1,
    }