from pathlib import Path
from typing import cast

from scipy.sparse import csr_matrix
import skops.io as sio
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from phishing_ml.training.model import PhishingClassifier


class PhishingPredictor:
    def __init__(self, artifacts_dir: str | Path = "artifacts/baseline") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.vectorizer = self._load_vectorizer()
        self.model = self._load_model()

    def _load_vectorizer(self) -> TfidfVectorizer:
        vectorizer_path = self.artifacts_dir / "vectorizer.skops"

        if not vectorizer_path.exists():
            raise FileNotFoundError(f"Vectorizer not found: {vectorizer_path}")

        vectorizer = sio.load(vectorizer_path)
        if not isinstance(vectorizer, TfidfVectorizer):
            raise TypeError("Unexpected vectorizer type")

        return vectorizer

    def _load_model(self) -> PhishingClassifier:
        model_path = self.artifacts_dir / "model.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        input_dim = len(self.vectorizer.get_feature_names_out())
        model = PhishingClassifier(input_dim=input_dim)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        features = cast(csr_matrix, self.vectorizer.transform([text])).toarray()
        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(features_tensor)
            probability = torch.sigmoid(logits).item()

        label = int(probability >= threshold)

        return {
            "label": label,
            "class_name": "phishing" if label == 1 else "legitimate",
            "phishing_probability": probability,
            "threshold": threshold,
        }