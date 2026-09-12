"""
Model integration / workflow (Week 6: pipeline integration & validation).

Week 5 shipped a single hardcoded Logistic Regression as a pipeline smoke
test. Week 6 replaces that with:

1. A pluggable model registry (MODEL_FACTORY) so a new candidate can be
   added/swapped without touching the pipeline logic around it — this is
   the "model interface" the ML Engineering track owns, and is the
   integration point the Data Science track's chosen algorithm plugs into.
2. Error analysis of the Week 5 baseline (see docs/PIPELINE.md §9) that
   directly motivated two concrete changes tested here:
     - Dropping previous_appointments/previous_no_shows (multicollinear
       with the engineered no_show_rate_history — see build_features.py)
     - Trying a RandomForestClassifier alongside a refined Logistic
       Regression, since the baseline's errors were not concentrated in
       any single obvious segment (see error analysis), suggesting a
       model capable of interactions might help more than more features.
3. A candidate-selection step that compares models on the SAME time-aware
   split and only promotes a new "candidate" model if it beats the
   baseline on the agreed metric (ROC-AUC, per config.yaml).

This is still not a final production model — see docs/PIPELINE.md for the
Data Science track dependency this remains subject to.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

from src.config import get_config, get_logger
from src.data.clean import clean_appointments
from src.data.validate import load_data
from src.features.build_features import build_target, build_feature_matrix

log = get_logger("training")

MODELS_DIR = Path("models")
REGISTRY_LOG = MODELS_DIR / "model_registry_log.csv"

MODEL_FACTORY = {
    "baseline_logreg": lambda: LogisticRegression(max_iter=1000),
    "refined_logreg": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42
    ),
}


def time_aware_split(df: pd.DataFrame, date_col: str, test_frac: float = 0.2):
    """Chronological split on date_col — see module docstring in Week 5 version for rationale."""
    df_sorted = df.sort_values(date_col)
    cutoff_idx = int(len(df_sorted) * (1 - test_frac))
    cutoff_date = df_sorted.iloc[cutoff_idx][date_col]
    train = df_sorted[df_sorted[date_col] < cutoff_date]
    test = df_sorted[df_sorted[date_col] >= cutoff_date]
    return train, test, cutoff_date


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
    }, confusion_matrix(y_test, preds)


def validate_training_inputs(X_train, X_test, y_train, y_test):
    """
    Week 6: explicit input validation before training — previously the
    pipeline would fail with an opaque sklearn error if columns mismatched.
    """
    issues = []
    if list(X_train.columns) != list(X_test.columns):
        issues.append("Train/test feature columns do not match after alignment.")
    if X_train.isna().any().any() or X_test.isna().any().any():
        issues.append("NaNs present in feature matrix after preprocessing.")
    if set(pd.unique(np.asarray(y_train))) - {0, 1} or set(pd.unique(np.asarray(y_test))) - {0, 1}:
        issues.append("Target contains values other than 0/1.")
    if len(X_train) == 0 or len(X_test) == 0:
        issues.append("Empty train or test split.")
    if issues:
        raise ValueError("Training input validation failed: " + "; ".join(issues))
    log.info("Input validation passed: %d train rows, %d test rows, %d features",
              len(X_train), len(X_test), X_train.shape[1])


def run():
    cfg = get_config()
    log.info("Loading and preparing data...")
    raw = load_data(cfg["data"]["raw_path"])
    cleaned = clean_appointments(raw)
    targeted = build_target(cleaned)

    train_raw, test_raw, cutoff = time_aware_split(
        targeted, cfg["split"]["date_column"], cfg["split"]["test_fraction"]
    )
    log.info("Time-aware split cutoff: %s | train=%d test=%d",
              cutoff.date(), len(train_raw), len(test_raw))

    train_feat = build_feature_matrix(train_raw)
    test_feat = build_feature_matrix(test_raw)
    y_train = train_raw["is_no_show"].values
    y_test = test_raw["is_no_show"].values
    X_train = train_feat.drop(columns=["is_no_show"], errors="ignore")
    X_test = test_feat.drop(columns=["is_no_show"], errors="ignore")
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    validate_training_inputs(X_train, X_test, y_train, y_test)

    results = {}
    trained_models = {}
    for name, factory in MODEL_FACTORY.items():
        log.info("Training %s...", name)
        model = factory()
        model.fit(X_train, y_train)
        metrics, cm = evaluate(model, X_test, y_test)
        results[name] = {"metrics": metrics, "confusion_matrix": cm.tolist()}
        trained_models[name] = model
        log.info("%s -> %s", name, metrics)

    selection_metric = cfg["model"]["selection_metric"]
    baseline_score = results["baseline_logreg"]["metrics"][selection_metric]
    candidate_name = max(
        (n for n in results if n != "baseline_logreg"),
        key=lambda n: results[n]["metrics"][selection_metric],
    )
    candidate_score = results[candidate_name]["metrics"][selection_metric]
    promote_candidate = candidate_score > baseline_score

    recommended = candidate_name if promote_candidate else "baseline_logreg"
    log.info("Baseline %s=%.4f | Best candidate (%s) %s=%.4f | Recommended: %s",
              selection_metric, baseline_score, candidate_name, selection_metric,
              candidate_score, recommended)

    MODELS_DIR.mkdir(exist_ok=True)
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    log_rows = []
    for name, model in trained_models.items():
        is_recommended = (name == recommended)
        tag = "candidate_recommended" if is_recommended else "candidate_evaluated"
        model_path = MODELS_DIR / f"{tag}_{name}_{version}.joblib"
        joblib.dump(model, model_path)
        m = results[name]["metrics"]
        log_rows.append({
            "model_version": f"{tag}_{name}_{version}",
            "training_date": datetime.now(timezone.utc).isoformat(),
            "algorithm": name,
            "feature_list_ref": f"{list(X_train.columns)}",
            "train_rows": len(X_train),
            "val_metric_recall": m["recall"],
            "val_metric_f1": m["f1"],
            "val_metric_roc_auc": m["roc_auc"],
            "approved_by": (
                "Week 6 integration — recommended candidate for Data Science review"
                if is_recommended else "Week 6 integration — evaluated, not selected"
            ),
            "notes": f"Week 6 comparison run {version}; selection_metric={selection_metric}",
        })

    log_df = pd.DataFrame(log_rows)
    if REGISTRY_LOG.exists():
        log_df.to_csv(REGISTRY_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(REGISTRY_LOG, index=False)
    log.info("Registry updated with %d model(s) at %s", len(log_rows), REGISTRY_LOG)

    comparison_path = MODELS_DIR / f"model_comparison_{version}.json"
    with open(comparison_path, "w") as f:
        json.dump({
            "version": version,
            "selection_metric": selection_metric,
            "recommended_model": recommended,
            "results": {n: r["metrics"] for n, r in results.items()},
        }, f, indent=2)
    log.info("Comparison summary saved to %s", comparison_path)

    return results, recommended


if __name__ == "__main__":
    run()
