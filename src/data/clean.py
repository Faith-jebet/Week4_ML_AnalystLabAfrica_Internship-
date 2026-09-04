"""
Cleaning / preprocessing module.

Responsibility:
- Apply reproducible cleaning transforms to a validated appointment DataFrame
- Never touch data/raw — always returns/saves a new processed copy
- Keep decisions explicit and documented rather than silently applied

Called by both the training pipeline (on historical data) and the batch
inference pipeline (on new appointment records), so cleaning logic never
drifts between the two.
"""

import pandas as pd
from src.data.validate import load_data, NULLABLE_COLUMNS


def clean_appointments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the Week 5 cleaning decisions:

    1. distance_to_clinic_km (1.8% missing): impute with the median for the
       same age_group, falling back to the overall median. Distance is
       reasonably stable within an age/demographic cohort and this avoids
       distorting the distribution with a single global constant.
    2. reminder_channel (27.3% missing): NOT a data-quality problem — every
       missing value corresponds exactly to reminder_sent == "No" (verified
       in validate.check_consistency). Filled with the explicit category
       "None" rather than imputed, since "no reminder was sent" is itself
       informative.
    3. waiting_time_minutes (1.2% missing): left as-is. This column is
       EXCLUDED from modelling entirely in the feature pipeline (see
       src/features/build_features.py) because it is only known after a
       patient checks in — using it (or even its missingness) as a feature
       would leak information that does not exist at prediction time. It is
       kept in the processed file only for potential downstream analytics
       use (e.g. the Data Analytics track), clearly excluded before features
       are built.
    4. No duplicate rows, no duplicate appointment_id, no logically
       inconsistent records were found (see validate.check_consistency) —
       no row-level dropping was necessary.
    """
    out = df.copy()

    median_by_group = out.groupby("age_group")["distance_to_clinic_km"].transform("median")
    overall_median = out["distance_to_clinic_km"].median()
    out["distance_to_clinic_km"] = out["distance_to_clinic_km"].fillna(median_by_group).fillna(overall_median)
    out["distance_was_missing"] = df["distance_to_clinic_km"].isna().astype(int)

    out["reminder_channel"] = out["reminder_channel"].fillna("None")

    for col in ["gender", "age_group", "appointment_type", "appointment_day",
                "appointment_time", "reminder_sent", "reminder_channel", "appointment_outcome"]:
        out[col] = out[col].astype("category")

    return out


def save_processed(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw = load_data("data/raw/HealthConnect_Appointment_Data.csv")
    cleaned = clean_appointments(raw)

    print(f"Rows in: {len(raw)}  |  Rows out: {len(cleaned)}")
    print(f"distance_to_clinic_km missing before: {raw['distance_to_clinic_km'].isna().sum()}, "
          f"after: {cleaned['distance_to_clinic_km'].isna().sum()}")
    print(f"reminder_channel missing before: {raw['reminder_channel'].isna().sum()}, "
          f"after: {cleaned['reminder_channel'].isna().sum()}")
    print(f"distance_was_missing flag count: {cleaned['distance_was_missing'].sum()}")

    out_path = "data/processed/appointments_cleaned.csv"
    save_processed(cleaned, out_path)
    print(f"\nSaved processed file to: {out_path}")
