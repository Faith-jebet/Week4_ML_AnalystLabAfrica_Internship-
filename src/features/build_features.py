"""
Feature engineering module.

Responsibility:
- Turn a cleaned appointment DataFrame into a model-ready feature matrix
- Explicitly exclude fields that are not available at prediction time
- Encode categoricals, derive behavioural features
- Called identically by the training pipeline and the batch inference
  pipeline, so features can never drift between the two

Target definition (confirmed with the Data Science track — see
docs/PIPELINE.md, Cross-Track Collaboration):
- Binary target: is_no_show = 1 if appointment_outcome == "No-Show", else 0
- Cancelled appointments are DROPPED from the modelling set, not encoded
  as 0 — a cancellation is a different event (patient proactively frees
  the slot) from a genuine no-show, and blending them would teach the
  model a misleading distinction. This matches the Week 4 design.

Week 6 update: previous_appointments and previous_no_shows are now
EXCLUDED from the final feature matrix (they remain available upstream
for other uses). Error analysis on the Week 5 baseline found these two
raw counts correlate 0.79 with the engineered no_show_rate_history and
produced contradictory coefficient signs (previous_no_shows +0.55 vs.
no_show_rate_history -0.55) — a multicollinearity artefact that made the
model harder to interpret without improving its predictions. The
engineered ratio (no_show_rate_history) plus is_first_time_patient
already carry this information in a cleaner form. See
docs/PIPELINE.md §9 (Week 6) for the full analysis.

Week 7 update: build_feature_matrix now accepts an optional
`feature_params` dict (from fit_feature_params()). Two real bugs were
found via edge-case testing this week and fixed by this change — see
docs/PIPELINE.md §11 and docs/ISSUE_LOG.md #11/#12:

1. Categorical columns were one-hot encoded with pd.get_dummies directly
   on whatever rows were passed in. A single-row (or otherwise small)
   batch would only contain SOME category values, so pd.get_dummies
   silently produced a completely different, smaller set of columns than
   training — e.g. a real single-appointment batch produced 9 columns
   instead of the expected ~30. The downstream `reindex()` in
   src/inference/score.py papered over the shape mismatch by zero-filling
   missing columns, but that means a real category could be silently
   scored as if it were the reference/baseline category. Fix: fit fixed
   category levels ONCE from the full training set (fit_feature_params),
   and apply them via pd.Categorical before get_dummies, so column shape
   is now identical regardless of batch size or composition.

2. `long_lead_time`'s threshold (`booking_lead_days.quantile(0.75)`) was
   recomputed from whatever DataFrame was passed in. For a single-row
   batch, a row's own 75th percentile is itself — so long_lead_time was
   ALWAYS 1 for any single-appointment batch, regardless of the appointment's
   actual booking lead time. Fix: the threshold is now fit once from the
   training set and reused at inference time, exactly like a real
   fitted preprocessing parameter (mean/std, encoder categories, etc.
   normally would be).
"""

import pandas as pd

from src.config import get_logger

log = get_logger("features")

# Never available at prediction time — a real appointment hasn't happened yet.
# waiting_time_minutes is excluded even though this synthetic file happens to
# populate it for No-Show/Cancelled rows too (see docs/PIPELINE.md) — the
# exclusion is a production-availability rule, not a data-quality patch.
EXCLUDED_COLUMNS = ["waiting_time_minutes", "appointment_id", "patient_id", "appointment_outcome"]

# Week 6: redundant raw counts, superseded by engineered features (see module docstring)
REDUNDANT_HISTORY_COLUMNS = ["previous_appointments", "previous_no_shows"]

CATEGORICAL_COLUMNS = [
    "gender", "age_group", "appointment_type", "appointment_day",
    "appointment_time", "reminder_sent", "reminder_channel",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Cancelled rows and add a binary is_no_show target."""
    out = df[df["appointment_outcome"] != "Cancelled"].copy()
    out["is_no_show"] = (out["appointment_outcome"] == "No-Show").astype(int)
    return out


def fit_feature_params(df: pd.DataFrame) -> dict:
    """
    Week 7: compute fixed preprocessing parameters ONCE from a reference
    (training) dataset, to be reused unchanged at inference time — the
    same idea as fitting a scaler or encoder in scikit-learn, just done
    explicitly here since the pipeline doesn't use sklearn transformers
    for this part.

    Must be called on the full/training dataset, never on a small batch —
    that defeats the purpose (see module docstring, Week 7 update).
    """
    return {
        "category_levels": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in CATEGORICAL_COLUMNS
        },
        "long_lead_time_threshold": float(df["booking_lead_days"].quantile(0.75)),
    }


def build_feature_matrix(df: pd.DataFrame, feature_params: dict = None) -> pd.DataFrame:
    """
    Derive model-ready features from a cleaned DataFrame.

    Args:
        feature_params: output of fit_feature_params(), computed once from
            the training set. When provided, category encoding and the
            long_lead_time threshold are FIXED regardless of what this
            particular df contains — required for correct behaviour on
            small or skewed batches (see module docstring, Week 7 update).
            When None, parameters are derived from `df` itself — only
            correct when `df` IS the reference/training set (e.g. the
            __main__ inspection block below, or when calling
            fit_feature_params() separately beforehand).

    Engineered features:
    - no_show_rate_history: previous_no_shows / previous_appointments,
      0 for patients with no history (first-time patients). Captures each
      patient's personal track record, which is one of the strongest
      plausible predictors of a future no-show.
    - is_first_time_patient: 1 if previous_appointments == 0. Kept as its
      own flag because a rate of 0 means something different for a
      first-timer (no history at all) than for a patient with 10 clean
      visits — collapsing both to "0" would blur that distinction.
    - is_weekend_appointment: 1 if appointment_day is Saturday/Sunday —
      tests whether weekend scheduling relates to attendance.
    - long_lead_time: 1 if booking_lead_days is at/above the FITTED 75th
      percentile threshold (see fit_feature_params) — long gaps between
      booking and the appointment are a plausible no-show driver.
    - distance_was_missing: carried through from the cleaning step.
    """
    out = df.copy()

    out["no_show_rate_history"] = (
        out["previous_no_shows"] / out["previous_appointments"].replace(0, pd.NA)
    ).fillna(0)
    out["is_first_time_patient"] = (out["previous_appointments"] == 0).astype(int)
    out["is_weekend_appointment"] = out["appointment_day"].isin(["Saturday", "Sunday"]).astype(int)

    threshold = (
        feature_params["long_lead_time_threshold"] if feature_params is not None
        else df["booking_lead_days"].quantile(0.75)
    )
    out["long_lead_time"] = (out["booking_lead_days"] >= threshold).astype(int)

    if feature_params is not None:
        for col in CATEGORICAL_COLUMNS:
            allowed = feature_params["category_levels"][col]
            str_values = out[col].astype(str)
            is_unseen = out[col].notna() & ~str_values.isin(allowed)
            unseen = int(is_unseen.sum())
            if unseen > 0:
                log.warning("%d row(s) had a value in '%s' not seen during fitting — "
                            "scored as the reference category (see docs/ISSUE_LOG.md #11).",
                            unseen, col)
            # Mask unseen values to NaN BEFORE constructing the Categorical — passing
            # out-of-vocabulary values directly to pd.Categorical(categories=...) is
            # deprecated in pandas and will raise in a future version.
            str_values = str_values.where(~is_unseen, other=None)
            out[col] = pd.Categorical(str_values, categories=allowed)

    out = pd.get_dummies(out, columns=CATEGORICAL_COLUMNS, drop_first=True)

    drop_cols = [c for c in EXCLUDED_COLUMNS if c in out.columns]
    drop_cols += [c for c in REDUNDANT_HISTORY_COLUMNS if c in out.columns]  # Week 6 refinement
    drop_cols += ["booking_date", "appointment_date"]  # raw dates not used directly as features
    out = out.drop(columns=drop_cols)

    return out


if __name__ == "__main__":
    from src.data.clean import clean_appointments
    from src.data.validate import load_data

    raw = load_data("data/raw/HealthConnect_Appointment_Data.csv")
    cleaned = clean_appointments(raw)

    targeted = build_target(cleaned)
    print(f"Rows after dropping Cancelled: {len(targeted)} (removed {len(cleaned) - len(targeted)})")
    print(f"is_no_show distribution:\n{targeted['is_no_show'].value_counts(normalize=True).round(3)}")

    features = build_feature_matrix(targeted)
    print(f"\nFeature matrix shape: {features.shape}")
    print("Excluded columns check — none of these should appear:", EXCLUDED_COLUMNS)
    leaked = [c for c in EXCLUDED_COLUMNS if c in features.columns]
    print("Leaked columns found:", leaked if leaked else "None — clean.")
    print("\nSample engineered feature stats:")
    print(features[["no_show_rate_history", "is_first_time_patient",
                     "is_weekend_appointment", "long_lead_time"]].describe())
