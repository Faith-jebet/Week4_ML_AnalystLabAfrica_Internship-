"""
<<<<<<< HEAD
Batch inference module (Week 6: integrated + validated).

Changes from Week 5:
- Loads the latest RECOMMENDED candidate model (tagged by train.py's
  selection step), not just "the latest file alphabetically" — Week 5's
  version picked up any file matching a fixed prefix, which would have
  silently picked an unapproved/losing candidate once multiple models
  existed in the registry.
- Adds validate_scores(): explicit output validation (probability range,
  no NaNs, row-count integrity, appointment_id preserved) rather than
  trusting the model's output blindly.
- Uses shared logging instead of print().
"""

import glob
import json
=======
Batch inference module.

Responsibility:
- Load the latest model artefact from the registry
- Apply the SAME cleaning + feature pipeline used in training
- Score a batch of appointments
- Output: appointment_id, no_show_probability, risk_tier, model_version

Week 5 scope: this scores the held-out slice of the existing dataset as a
stand-in for "upcoming appointments", to prove the batch-scoring path
works end-to-end. Wiring this to a live daily appointment feed is Week 6+
work, per the Week 4 design.
"""

import glob
>>>>>>> origin/main
from pathlib import Path

import joblib
import pandas as pd

<<<<<<< HEAD
from src.config import get_config, get_logger
from src.data.clean import clean_appointments
from src.data.validate import load_data, validate_feature_matrix_contract
from src.features.build_features import build_target, build_feature_matrix

log = get_logger("inference")

MODELS_DIR = Path("models")


def latest_recommended_model_path() -> Path:
    """
    Find the most recent model tagged 'candidate_recommended' by train.py.
    Falls back to the Week 5 baseline artefact if no Week 6 candidate has
    been trained yet, so the pipeline degrades gracefully rather than
    breaking for anyone who hasn't re-run training.
    """
    recommended = sorted(glob.glob(str(MODELS_DIR / "candidate_recommended_*.joblib")))
    if recommended:
        return Path(recommended[-1])
    legacy_baseline = sorted(glob.glob(str(MODELS_DIR / "baseline_logreg_*.joblib")))
    if legacy_baseline:
        log.warning("No Week 6 recommended candidate found — falling back to Week 5 baseline artefact.")
        return Path(legacy_baseline[-1])
    raise FileNotFoundError(
        "No model artefact found in models/. Run `python -m src.training.train` first."
    )


def risk_tier(prob: float, cfg: dict) -> str:
    thresholds = cfg["risk_tiers"]
    if prob < thresholds["low_max"]:
        return "Low"
    if prob < thresholds["medium_max"]:
=======
from src.data.clean import clean_appointments
from src.data.validate import load_data
from src.features.build_features import build_target, build_feature_matrix

MODELS_DIR = Path("models")

RISK_THRESHOLDS = {"low": 0.33, "medium": 0.66}  # score < low -> Low, < medium -> Medium, else High


def latest_model_path() -> Path:
    candidates = sorted(glob.glob(str(MODELS_DIR / "baseline_logreg_*.joblib")))
    if not candidates:
        raise FileNotFoundError(
            "No model artefact found in models/. Run `python -m src.training.train` first."
        )
    return Path(candidates[-1])


def risk_tier(prob: float) -> str:
    if prob < RISK_THRESHOLDS["low"]:
        return "Low"
    if prob < RISK_THRESHOLDS["medium"]:
>>>>>>> origin/main
        return "Medium"
    return "High"


<<<<<<< HEAD
def validate_scores(result: pd.DataFrame, expected_rows: int) -> None:
    """
    Week 6: explicit output validation. Raises if the batch output is not
    safe to hand to a downstream consumer (dashboard/reminder workflow).
    """
    issues = []
    if len(result) != expected_rows:
        issues.append(f"Row count mismatch: expected {expected_rows}, got {len(result)}")
    if result["no_show_probability"].isna().any():
        issues.append("NaN values present in no_show_probability")
    if not result["no_show_probability"].between(0, 1).all():
        issues.append("no_show_probability contains values outside [0, 1]")
    if result["appointment_id"].isna().any():
        issues.append("Missing appointment_id in scored output")
    if result["appointment_id"].duplicated().any():
        issues.append("Duplicate appointment_id in scored output")
    if not result["risk_tier"].isin(["Low", "Medium", "High"]).all():
        issues.append("risk_tier contains an unexpected value")
    if issues:
        raise ValueError("Output validation failed: " + "; ".join(issues))
    log.info("Output validation passed for %d scored rows.", len(result))


def score_batch(df_raw: pd.DataFrame, model, feature_columns, cfg: dict) -> pd.DataFrame:
=======
def score_batch(df_raw: pd.DataFrame, model, feature_columns) -> pd.DataFrame:
>>>>>>> origin/main
    cleaned = clean_appointments(df_raw)
    targeted = build_target(cleaned)  # drops Cancelled — same rule as training
    features = build_feature_matrix(targeted).drop(columns=["is_no_show"], errors="ignore")
    features = features.reindex(columns=feature_columns, fill_value=0)

<<<<<<< HEAD
    contract_issues = validate_feature_matrix_contract(features, feature_columns)
    if contract_issues:
        raise ValueError("Model input contract violated: " + "; ".join(contract_issues))

=======
>>>>>>> origin/main
    probs = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({
        "appointment_id": targeted["appointment_id"].values,
        "no_show_probability": probs.round(4),
    })
<<<<<<< HEAD
    result["risk_tier"] = result["no_show_probability"].apply(lambda p: risk_tier(p, cfg))

    validate_scores(result, expected_rows=len(targeted))
=======
    result["risk_tier"] = result["no_show_probability"].apply(risk_tier)
>>>>>>> origin/main
    return result


if __name__ == "__main__":
<<<<<<< HEAD
    cfg = get_config()
    model_path = latest_recommended_model_path()
    log.info("Loading model: %s", model_path)
    model = joblib.load(model_path)

    raw = load_data(cfg["data"]["raw_path"])
    raw_sorted = raw.sort_values("appointment_date")
    batch = raw_sorted.tail(int(len(raw_sorted) * 0.1))
    log.info("Scoring batch of %d appointments (most recent 10%% by date)", len(batch))

    feature_columns = list(
        build_feature_matrix(build_target(clean_appointments(raw)))
        .drop(columns=["is_no_show"], errors="ignore").columns
    )

    scored = score_batch(batch, model, feature_columns, cfg)
    scored["model_version"] = model_path.stem

    log.info("Risk tier distribution: %s", scored["risk_tier"].value_counts().to_dict())

    out_path = Path(cfg["data"]["processed_dir"]) / "scored_batch_sample.csv"
    scored.to_csv(out_path, index=False)
    log.info("Saved scored batch to: %s", out_path)
=======
    model_path = latest_model_path()
    print(f"Loading model: {model_path}")
    model = joblib.load(model_path)

    raw = load_data("data/raw/HealthConnect_Appointment_Data.csv")
    # Use the most recent 10% of appointments by date as a stand-in "upcoming batch"
    raw_sorted = raw.sort_values("appointment_date")
    batch = raw_sorted.tail(int(len(raw_sorted) * 0.1))

    feature_columns = [c for c in build_feature_matrix(build_target(clean_appointments(raw)))
                        .drop(columns=["is_no_show"], errors="ignore").columns]

    scored = score_batch(batch, model, feature_columns)
    scored["model_version"] = model_path.stem.replace("baseline_logreg_", "")

    print(f"\nScored {len(scored)} appointments.")
    print("\nRisk tier distribution:")
    print(scored["risk_tier"].value_counts())
    print("\nSample output:")
    print(scored.head(10).to_string(index=False))

    out_path = Path("data/processed") / "scored_batch_sample.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nSaved scored batch to: {out_path}")
>>>>>>> origin/main
