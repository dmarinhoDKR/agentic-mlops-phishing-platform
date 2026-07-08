import pandas as pd
import pytest

from phishing_ml.data.load_dataset import load_phishing_dataset


def test_load_phishing_dataset_valid_file(tmp_path):
    dataset_path = tmp_path / "samples.csv"
    pd.DataFrame(
        {
            "text": ["Urgent password reset", "Team meeting tomorrow"],
            "label": [1, 0],
        }
    ).to_csv(dataset_path, index=False)

    data = load_phishing_dataset(dataset_path)

    assert list(data.columns) == ["text", "label"]
    assert len(data) == 2
    assert data["label"].tolist() == [1, 0]


def test_load_phishing_dataset_requires_text_and_label_columns(tmp_path):
    dataset_path = tmp_path / "invalid.csv"
    pd.DataFrame({"message": ["hello"]}).to_csv(dataset_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_phishing_dataset(dataset_path)