from app.config import Settings


def test_hdbscan_min_samples_blank_env_var_falls_back_to_default(monkeypatch):
    """A blank HDBSCAN_MIN_SAMPLES (e.g. `KEY=` left in .env) must not crash
    Settings validation; it should fall back to the documented default of
    None, same as if the variable were absent entirely."""
    monkeypatch.setenv("HDBSCAN_MIN_SAMPLES", "")

    settings = Settings()

    assert settings.hdbscan_min_samples is None


def test_hdbscan_min_samples_valid_value_still_parses(monkeypatch):
    monkeypatch.setenv("HDBSCAN_MIN_SAMPLES", "5")

    settings = Settings()

    assert settings.hdbscan_min_samples == 5


def test_hdbscan_min_samples_unset_defaults_to_none():
    settings = Settings(hdbscan_min_cluster_size=10)

    assert settings.hdbscan_min_samples is None


def test_blank_bedrock_labeling_temperature_means_omit(monkeypatch):
    """A blank BEDROCK_LABELING_TEMPERATURE (e.g. `KEY=` in .env) must not

    crash Settings validation; it should fall back to None, the value that
    tells the labeler to omit the parameter entirely."""
    monkeypatch.setenv("BEDROCK_LABELING_TEMPERATURE", "")

    settings = Settings()

    assert settings.bedrock_labeling_temperature is None


def test_bedrock_labeling_temperature_valid_value_still_parses(monkeypatch):
    monkeypatch.setenv("BEDROCK_LABELING_TEMPERATURE", "0.7")

    settings = Settings()

    assert settings.bedrock_labeling_temperature == 0.7
