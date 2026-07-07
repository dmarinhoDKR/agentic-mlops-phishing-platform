from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"text", "label"}

def load_phishing_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    data = pd.read_csv(dataset_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    
    data = data.dropna(subset=["text", "label"]).copy()
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(int)

    return data