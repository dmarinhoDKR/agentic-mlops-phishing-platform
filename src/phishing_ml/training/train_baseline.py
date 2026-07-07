from pathlib import Path
import pickle

import mlflow
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch import nn

from phishing_ml.data.load_dataset import load_phishing_dataset
from phishing_ml.training.model import PhishingClassifier

DATASET_PATH = Path("data/raw/phishing_samples.csv")
ARTIFACTS_DIR = Path("artifacts/baseline")

def train() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_phishing_dataset(DATASET_PATH)
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"].tolist(),
        data["label"].tolist(),
        test_size=0.25,
        random_state=42,
        stratify=data["label"].tolist(),
    )

    vectorizer = TfidfVectorizer(max_features=128)
    x_train = vectorizer.fit_transform(train_texts).toarray()
    x_test = vectorizer.transform(test_texts).toarray()

    y_train = torch.tensor(train_labels, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.float32)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    model = PhishingClassifier(input_dim=x_train_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()

    mlflow.set_experiment("phishing-baseline")

    with mlflow.start_run(run_name="tfidf-pytorch-linear"):
        for epoch in range(50):
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
            predictions = (torch.sigmoid(test_logits) >= 0.5).float()
            accuracy = (predictions == y_test).float().mean().item()

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_param("model_type", "tfidf_pytorch_linear")
        mlflow.log_param("max_features", 128)

        torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pt")

        with open(ARTIFACTS_DIR / "vectorizer.pkl", "wb") as file:
            pickle.dump(vectorizer, file)

    print(f"test_accuracy={accuracy:.4f}")
    print(f"saved_artifacts={ARTIFACTS_DIR}")

if __name__ == "__main__":
    train()