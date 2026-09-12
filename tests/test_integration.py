"""
Week 6 integration tests — verifies the model comparison, the model
interface contract, and end-to-end reproducibility of the integrated
pipeline. Complements tests/test_pipeline.py (Week 5 unit tests, still in
force and still passing unchanged).

Run with: pytest tests/test_integration.py -v
"""

import glob
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import get_config
from src.data.validate import load_data, validate_feature_matrix_contract
from src.data.clean import clean_appointments
from src.features.build_features import build_target, build_feature_matrix
from src.training.train import MODEL_FACTORY, time_aware_split, validate_training_inputs
from src.inference.score import (
    latest_recommended_model_path, score_batch, validate_scores, risk_tier,
)

DATA_PATH = "data/raw/HealthConnect_Appointment_Data.csv"


@pytest.fixture(scope="module")
def prepared_data():
    raw = load_data(DATA_PATH)
    cleaned = clean_appointments(raw)
    targeted = build_target(cleaned)
    return raw, targeted


# ---------- config.py ----------

def test_config_loads_and_has_required_sections():
    cfg = get_config()
    for section in ("data", "features", "split", "model", "risk_tiers"):
        assert section in cfg


# ---------- feature refinement (Week 6) ----------

def test_redundant_history_columns_dropped_from_features(prepared_data):
    _, targeted = prepared_data
    features = build_feature_matrix(targeted)
    assert "previous_appointments" not in features.columns
    assert "previous_no_shows" not in features.columns


def test_no_show_rate_history_still_present_after_refinement(prepared_data):
    _, targeted = prepared_data
    features = build_feature_matrix(targeted)
    assert "no_show_rate_history" in features.columns


# ---------- model registry (Week 6) ----------

def test_model_factory_has_multiple_candidates():
    assert len(MODEL_FACTORY) >= 2
    assert "baseline_logreg" in MODEL_FACTORY


def test_all_registered_models_are_fittable_and_predict_probabilities(prepared_data):
    _, targeted = prepared_data
    train_raw, test_raw, _ = time_aware_split(targeted, "appointment_date")
    train_feat = build_feature_matrix(train_raw).drop(columns=["is_no_show"])
    test_feat = build_feature_matrix(test_raw).drop(columns=["is_no_show"])
    train_feat, test_feat = train_feat.align(test_feat, join="left", axis=1, fill_value=0)
    y_train = train_raw["is_no_show"].values

    for name, factory in MODEL_FACTORY.items():
        model = factory()
        model.fit(train_feat, y_train)
        probs = model.predict_proba(test_feat)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all(), f"{name} produced out-of-range probabilities"


def test_validate_training_inputs_rejects_mismatched_columns():
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    X_test = pd.DataFrame({"a": [1], "c": [3]})
    y_train = [0, 1]
    y_test = [0]
    with pytest.raises(ValueError, match="do not match"):
        validate_training_inputs(X_train, X_test, y_train, y_test)


def test_validate_training_inputs_rejects_bad_target():
    X = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="0/1"):
        validate_training_inputs(X, X, [0, 2], [0, 1])


# ---------- inference contract + output validation (Week 6) ----------

def test_feature_matrix_contract_detects_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    issues = validate_feature_matrix_contract(df, expected_columns=["a", "b"])
    assert any("missing" in i.lower() for i in issues)


def test_feature_matrix_contract_passes_on_match():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    issues = validate_feature_matrix_contract(df, expected_columns=["a", "b"])
    assert issues == []


def test_validate_scores_rejects_out_of_range_probability():
    bad = pd.DataFrame({
        "appointment_id": ["A1", "A2"],
        "no_show_probability": [0.5, 1.5],
        "risk_tier": ["Low", "High"],
    })
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_scores(bad, expected_rows=2)


def test_validate_scores_rejects_duplicate_appointment_id():
    bad = pd.DataFrame({
        "appointment_id": ["A1", "A1"],
        "no_show_probability": [0.5, 0.6],
        "risk_tier": ["Low", "Medium"],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        validate_scores(bad, expected_rows=2)


def test_risk_tier_thresholds_match_config():
    cfg = get_config()
    assert risk_tier(0.1, cfg) == "Low"
    assert risk_tier(0.5, cfg) == "Medium"
    assert risk_tier(0.9, cfg) == "High"


# ---------- end-to-end integration (requires a trained model in models/) ----------

def test_latest_recommended_model_path_resolves():
    # Skips gracefully if no model has been trained yet in this environment.
    if not glob.glob("models/candidate_recommended_*.joblib") and not glob.glob("models/baseline_logreg_*.joblib"):
        pytest.skip("No trained model artefact present — run `python -m src.training.train` first.")
    path = latest_recommended_model_path()
    assert path.exists()


def test_score_batch_end_to_end(prepared_data):
    if not glob.glob("models/candidate_recommended_*.joblib") and not glob.glob("models/baseline_logreg_*.joblib"):
        pytest.skip("No trained model artefact present — run `python -m src.training.train` first.")

    raw, targeted = prepared_data
    cfg = get_config()
    model_path = latest_recommended_model_path()
    model = joblib.load(model_path)
    feature_columns = list(build_feature_matrix(targeted).drop(columns=["is_no_show"]).columns)

    batch = raw.sort_values("appointment_date").tail(50)
    result = score_batch(batch, model, feature_columns, cfg)

    assert len(result) <= len(batch)  # Cancelled rows dropped
    assert result["no_show_probability"].between(0, 1).all()
    assert result["risk_tier"].isin(["Low", "Medium", "High"]).all()


def test_pipeline_reproducible_via_documented_commands():
    """
    Week 6: verifies the exact commands in README.md actually work when run
    fresh, rather than trusting the documentation matches the code.
    """
    commands = [
        [sys.executable, "-m", "src.data.validate"],
        [sys.executable, "-m", "src.data.clean"],
        [sys.executable, "-m", "src.features.build_features"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        assert result.returncode == 0, f"{' '.join(cmd)} failed:\n{result.stderr}"
