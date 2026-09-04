"""
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
from pathlib import Path

import joblib
import pandas as pd

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
        return "Medium"
    return "High"


def score_batch(df_raw: pd.DataFrame, model, feature_columns) -> pd.DataFrame:
    cleaned = clean_appointments(df_raw)
    targeted = build_target(cleaned)  # drops Cancelled — same rule as training
    features = build_feature_matrix(targeted).drop(columns=["is_no_show"], errors="ignore")
    features = features.reindex(columns=feature_columns, fill_value=0)

    probs = model.predict_proba(features)[:, 1]
    result = pd.DataFrame({
        "appointment_id": targeted["appointment_id"].values,
        "no_show_probability": probs.round(4),
    })
    result["risk_tier"] = result["no_show_probability"].apply(risk_tier)
    return result


if __name__ == "__main__":
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
