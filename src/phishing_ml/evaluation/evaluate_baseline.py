from pathlib import Path
import pickle

import torch

from phishing_ml.data.load_dataset import load_phishing_dataset
from phishing_ml.training.model import PhishingClassifier

DATASET_PATH = Path("data/raw/phishing_samples.csv")
ARTIFACTS_DIR = Path("artifacts/baseline")

def evaluate() -> None:
    data = load_phishing_dataset(DATASET_PATH)

    with open(ARTIFACTS_DIR / "vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    features = vectorizer.transform(data["text"].tolist()).toarray()
    labels = torch.tensor(data["label"].tolist(), dtype=torch.float32)
    features_tensor = torch.tensor(features, dtype=torch.float32)

    model = PhishingClassifier(input_dim=features_tensor.shape[1])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "model.pt"))
    model.eval()

    with torch.no_grad():
        logits = model(features_tensor)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        accuracy = (predictions == labels).float().mean().item()

    print(f"accuracy={accuracy:.4f}")

if __name__ == "__main__":
    evaluate()