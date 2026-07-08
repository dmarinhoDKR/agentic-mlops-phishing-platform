from pathlib import Path
import pickle
from typing import Any

import mlflow
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn

from phishing_ml.config import BaselineTrainingConfig, load_baseline_training_config
from phishing_ml.data.load_dataset import load_phishing_dataset
from phishing_ml.training.model import PhishingClassifier


DEFAULT_CONFIG_PATH = Path("configs/baseline.yaml")


def _to_dense_array(matrix: Any) -> Any:
    return matrix.toarray()


def train(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    resolved_config_path = Path(config_path)
    config = load_baseline_training_config(resolved_config_path)

    torch.manual_seed(config.random_state)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    data = load_phishing_dataset(config.dataset_path)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=data["label"].tolist(),
    )

    vectorizer = TfidfVectorizer(max_features=config.max_features)
    x_train = _to_dense_array(vectorizer.fit_transform(train_texts))
    x_test = _to_dense_array(vectorizer.transform(test_texts))

    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    model = PhishingClassifier(input_dim=x_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        _log_training_params(
            config=config,
            data_rows=len(data),
            train_rows=len(train_texts),
            test_rows=len(test_texts),
        )
        mlflow.log_artifact(str(resolved_config_path), artifact_path="config")

        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()

            logits = model(x_train_tensor)
            loss = loss_fn(logits, y_train)

            loss.backward()
            optimizer.step()

            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test_tensor)
            probabilities = torch.sigmoid(test_logits)
            predictions = (probabilities >= config.threshold).float()
            accuracy = (predictions == y_test).float().mean().item()

        mlflow.log_metric("test_accuracy", accuracy)

        model_path = config.artifacts_dir / "model.pt"
        vectorizer_path = config.artifacts_dir / "vectorizer.pkl"

        torch.save(model.state_dict(), model_path)

        with open(vectorizer_path, "wb") as file:
            pickle.dump(vectorizer, file)

        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(vectorizer_path), artifact_path="model")

    print(f"test_accuracy={accuracy:.4f}")
    print(f"saved_artifacts={config.artifacts_dir}")


def _log_training_params(
    config: BaselineTrainingConfig,
    data_rows: int,
    train_rows: int,
    test_rows: int,
) -> None:
    mlflow.log_params(
        {
            "dataset_path": str(config.dataset_path),
            "artifacts_dir": str(config.artifacts_dir),
            "model_type": config.model_type,
            "max_features": config.max_features,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "test_size": config.test_size,
            "random_state": config.random_state,
            "threshold": config.threshold,
            "data_rows": data_rows,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
    )


if __name__ == "__main__":
    train()