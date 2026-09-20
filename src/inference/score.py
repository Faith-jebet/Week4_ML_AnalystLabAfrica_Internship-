"""
Batch inference module (Week 6: integrated + validated. Week 7: fixed
feature encoding for small/skewed batches).

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

Week 7 changes (see docs/ISSUE_LOG.md #11/#12 and docs/PIPELINE.md §11):
- feature_columns is now read from the model itself (model.feature_names_in_)
  instead of being recomputed by re-running the feature pipeline over the
  full raw dataset on every inference run — that was both wasteful and,
  worse, itself vulnerable to the same batch-dependent encoding bug this
  update fixes.
- feature_params (fitted category levels + long_lead_time threshold) is
  now loaded from models/feature_params.json and passed through to
  build_feature_matrix, so a single-row batch encodes identically to a
  5,000-row batch.
"""

import glob
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import get_config, get_logger
from src.data.clean import clean_appointments
from src.data.validate import load_data, validate_feature_matrix_contract
from src.features.build_features import build_target, build_feature_matrix

log = get_logger("inference")

MODELS_DIR = Path("models")
FEATURE_PARAMS_PATH = MODELS_DIR / "feature_params.json"


def load_feature_params() -> dict:
    if not FEATURE_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"{FEATURE_PARAMS_PATH} not found. Run `python -m src.training.train` "
            "first — inference requires fitted feature parameters from training."
        )
    with open(FEATURE_PARAMS_PATH) as f:
        return json.load(f)


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
        return "Medium"
    return "High"


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


def score_batch(df_raw: pd.DataFrame, model, feature_columns, cfg: dict, feature_params: dict) -> pd.DataFrame:
    cleaned = clean_appointments(df_raw)
    targeted = build_target(cleaned)  # drops Cancelled — same rule as training
    features = build_feature_matrix(targeted, feature_params).drop(columns=["is_no_show"], errors="ignore")
    features = features.reindex(columns=feature_columns, fill_value=0)

    contract_issues = validate_feature_matrix_contract(features, feature_columns)
    if contract_issues:
        raise ValueError("Model input contract violated: " + "; ".join(contract_issues))

    if len(features) == 0:
        log.info("Empty batch after preprocessing (0 rows) — returning empty scored result.")
        return pd.DataFrame(columns=["appointment_id", "no_show_probability", "risk_tier"])

    probs = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({
        "appointment_id": targeted["appointment_id"].values,
        "no_show_probability": probs.round(4),
    })
    result["risk_tier"] = result["no_show_probability"].apply(lambda p: risk_tier(p, cfg))

    validate_scores(result, expected_rows=len(targeted))
    return result


if __name__ == "__main__":
    cfg = get_config()
    model_path = latest_recommended_model_path()
    log.info("Loading model: %s", model_path)
    model = joblib.load(model_path)
    feature_params = load_feature_params()

    raw = load_data(cfg["data"]["raw_path"])
    raw_sorted = raw.sort_values("appointment_date")
    batch = raw_sorted.tail(int(len(raw_sorted) * 0.1))
    log.info("Scoring batch of %d appointments (most recent 10%% by date)", len(batch))

    feature_columns = (
        list(model.feature_names_in_) if hasattr(model, "feature_names_in_")
        else list(build_feature_matrix(build_target(clean_appointments(raw)), feature_params)
                   .drop(columns=["is_no_show"], errors="ignore").columns)
    )

    scored = score_batch(batch, model, feature_columns, cfg, feature_params)
    scored["model_version"] = model_path.stem

    log.info("Risk tier distribution: %s", scored["risk_tier"].value_counts().to_dict())

    out_path = Path(cfg["data"]["processed_dir"]) / "scored_batch_sample.csv"
    scored.to_csv(out_path, index=False)
    log.info("Saved scored batch to: %s", out_path)
