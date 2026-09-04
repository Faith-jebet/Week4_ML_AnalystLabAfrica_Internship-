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
"""

import pandas as pd

# Never available at prediction time — a real appointment hasn't happened yet.
# waiting_time_minutes is excluded even though this synthetic file happens to
# populate it for No-Show/Cancelled rows too (see docs/PIPELINE.md) — the
# exclusion is a production-availability rule, not a data-quality patch.
EXCLUDED_COLUMNS = ["waiting_time_minutes", "appointment_id", "patient_id", "appointment_outcome"]

CATEGORICAL_COLUMNS = [
    "gender", "age_group", "appointment_type", "appointment_day",
    "appointment_time", "reminder_sent", "reminder_channel",
]


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Cancelled rows and add a binary is_no_show target."""
    out = df[df["appointment_outcome"] != "Cancelled"].copy()
    out["is_no_show"] = (out["appointment_outcome"] == "No-Show").astype(int)
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive model-ready features from a cleaned DataFrame.

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
    - long_lead_time: 1 if booking_lead_days is in the top quartile —
      long gaps between booking and the appointment are a plausible
      no-show driver worth isolating as its own signal.
    - distance_was_missing: carried through from the cleaning step.
    """
    out = df.copy()

    out["no_show_rate_history"] = (
        out["previous_no_shows"] / out["previous_appointments"].replace(0, pd.NA)
    ).fillna(0)
    out["is_first_time_patient"] = (out["previous_appointments"] == 0).astype(int)
    out["is_weekend_appointment"] = out["appointment_day"].isin(["Saturday", "Sunday"]).astype(int)
    lead_q3 = df["booking_lead_days"].quantile(0.75)
    out["long_lead_time"] = (out["booking_lead_days"] >= lead_q3).astype(int)

    out = pd.get_dummies(out, columns=CATEGORICAL_COLUMNS, drop_first=True)

    drop_cols = [c for c in EXCLUDED_COLUMNS if c in out.columns]
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
