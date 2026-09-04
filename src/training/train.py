"""
Initial model integration / workflow scaffold.

Responsibility (Week 5 scope — NOT final model development, that belongs to
the Data Science track):
- Prove the ML Engineering pipeline runs end-to-end: raw data -> validation
  -> cleaning -> features -> time-aware split -> a model -> a saved,
  versioned artefact -> a metrics log.
- Provide the integration point the Data Science track's chosen model will
  eventually slot into, without pre-empting their algorithm choice.

A plain Logistic Regression baseline is used here deliberately — it is not
a proposal for the final model. Its only job is to validate that the
pipeline scaffolding (split strategy, feature matrix, model registry
pattern) actually works against real data before Week 6.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from src.data.clean import clean_appointments
from src.data.validate import load_data
from src.features.build_features import build_target, build_feature_matrix

MODELS_DIR = Path("models")
REGISTRY_LOG = MODELS_DIR / "model_registry_log.csv"


def time_aware_split(df: pd.DataFrame, date_col: str, test_frac: float = 0.2):
    """
    Split chronologically on appointment_date rather than randomly.

    Rationale: in production, the model will always be scoring appointments
    that are chronologically AFTER the data it was trained on. A random
    split would let the model "see the future" relative to some of its own
    training data and overstate how well it will generalise once deployed.
    """
    df_sorted = df.sort_values(date_col)
    cutoff_idx = int(len(df_sorted) * (1 - test_frac))
    cutoff_date = df_sorted.iloc[cutoff_idx][date_col]
    train = df_sorted[df_sorted[date_col] < cutoff_date]
    test = df_sorted[df_sorted[date_col] >= cutoff_date]
    return train, test, cutoff_date


def run():
    raw = load_data("data/raw/HealthConnect_Appointment_Data.csv")
    cleaned = clean_appointments(raw)
    targeted = build_target(cleaned)

    train_raw, test_raw, cutoff = time_aware_split(targeted, "appointment_date")
    print(f"Time-aware split cutoff (appointment_date): {cutoff.date()}")
    print(f"Train rows: {len(train_raw)}  |  Test rows: {len(test_raw)}")

    # Build features on train and test separately using the SAME function,
    # then align columns (test may lack a rare category present in train).
    train_feat = build_feature_matrix(train_raw)
    test_feat = build_feature_matrix(test_raw)
    y_train = train_raw["is_no_show"].values
    y_test = test_raw["is_no_show"].values
    X_train = train_feat.drop(columns=["is_no_show"], errors="ignore")
    X_test = test_feat.drop(columns=["is_no_show"], errors="ignore")
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }
    cm = confusion_matrix(y_test, preds)

    print("\nBaseline metrics (smoke test — not a final model):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # --- Lightweight model registry: versioned artefact + metadata log ---
    MODELS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"baseline_logreg_{version}.joblib"
    try:
        import joblib
        joblib.dump(model, model_path)
        saved = True
    except ImportError:
        saved = False

    log_row = {
        "model_version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "algorithm": "LogisticRegression (baseline smoke-test)",
        "feature_list_ref": f"{list(X_train.columns)}",
        "train_rows": len(X_train),
        "val_metric_recall": metrics["recall"],
        "val_metric_f1": metrics["f1"],
        "val_metric_roc_auc": metrics["roc_auc"],
        "approved_by": "not approved — Week 5 pipeline smoke test only",
        "notes": "Time-aware split; excludes waiting_time_minutes/appointment_id/patient_id. "
                 "Final model selection is the Data Science track's responsibility.",
    }
    log_df = pd.DataFrame([log_row])
    if REGISTRY_LOG.exists():
        log_df.to_csv(REGISTRY_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(REGISTRY_LOG, index=False)

    print(f"\nModel artefact saved: {model_path if saved else '(joblib unavailable — skipped)'}")
    print(f"Registry log updated: {REGISTRY_LOG}")

    return metrics


if __name__ == "__main__":
    run()
