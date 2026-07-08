import pytest

from phishing_ml.config import load_baseline_training_config


def test_load_baseline_training_config(tmp_path):
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        """
experiment_name: phishing-baseline
run_name: test-run
dataset_path: data/raw/phishing_samples.csv
artifacts_dir: artifacts/baseline
model_type: tfidf_pytorch_linear
max_features: 128
learning_rate: 0.05
epochs: 50
test_size: 0.25
random_state: 42
threshold: 0.5
""",
        encoding="utf-8",
    )

    config = load_baseline_training_config(config_path)

    assert config.experiment_name == "phishing-baseline"
    assert config.run_name == "test-run"
    assert config.max_features == 128
    assert config.learning_rate == 0.05
    assert config.epochs == 50


def test_load_baseline_training_config_requires_all_keys(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("experiment_name: phishing-baseline\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required config keys"):
        load_baseline_training_config(config_path)