import pytest
import skops.io as sio
from sklearn.feature_extraction.text import CountVectorizer

from phishing_ml.inference.predictor import PhishingPredictor


def test_predictor_rejects_unexpected_vectorizer(tmp_path):
    sio.dump(CountVectorizer(), tmp_path / "vectorizer.skops")

    with pytest.raises(TypeError, match="Unexpected vectorizer type"):
        PhishingPredictor(artifacts_dir=tmp_path)
