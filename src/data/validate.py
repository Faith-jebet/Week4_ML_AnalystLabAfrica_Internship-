"""
Data validation module.

Responsibility:
- Load the HealthConnect appointment dataset
- Validate columns against the HealthConnect Data Dictionary
- Check for duplicates, inconsistent records, and out-of-range values
- Produce a data-quality report (missing values, distributions, anomalies)

Shared by both the training pipeline and the batch inference pipeline so
validation logic never drifts between the two.

Week 5 note: the Data Dictionary states booking_date/appointment_date are
"ISO format". The actual file uses M/D/YYYY (US format), not ISO 8601
(YYYY-MM-DD). This is a real discrepancy between documentation and data,
found while implementing this module — see docs/PIPELINE.md.
"""

import pandas as pd

EXPECTED_COLUMNS = [
    "appointment_id", "patient_id", "gender", "age", "age_group",
    "appointment_type", "booking_date", "appointment_date", "appointment_day",
    "appointment_time", "booking_lead_days", "previous_appointments",
    "previous_no_shows", "reminder_sent", "reminder_channel",
    "distance_to_clinic_km", "waiting_time_minutes", "appointment_outcome",
]

# Structurally-missing-by-design columns (per Data Dictionary / observed pattern)
NULLABLE_COLUMNS = ["distance_to_clinic_km", "waiting_time_minutes", "reminder_channel"]

DATE_FORMAT = "%m/%d/%Y"  # observed format — NOT ISO, despite the Data Dictionary


def load_data(path: str) -> pd.DataFrame:
    """Load an appointment data CSV and parse date columns with the observed format."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find appointment data at '{path}'. "
            "Check the path, or place HealthConnect_Appointment_Data.csv in data/raw/."
        ) from e
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"File at '{path}' is empty or not a valid CSV.") from e

    missing_required = [c for c in ("booking_date", "appointment_date") if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required date column(s) missing from input file: {missing_required}")

    df["booking_date"] = pd.to_datetime(df["booking_date"], format=DATE_FORMAT, errors="coerce")
    df["appointment_date"] = pd.to_datetime(df["appointment_date"], format=DATE_FORMAT, errors="coerce")
    return df


def validate_schema(df: pd.DataFrame) -> list:
    """Return a list of schema issues (empty list = schema OK)."""
    issues = []
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")
    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        issues.append(f"Unexpected columns present: {extra_cols}")
    unparsed_dates = df["booking_date"].isna().sum() if "booking_date" in df else None
    if unparsed_dates:
        issues.append(f"{unparsed_dates} booking_date values failed to parse")
    return issues


def check_consistency(df: pd.DataFrame) -> dict:
    """Cross-field logical checks. Returns a dict of check_name -> failure count."""
    results = {}
    results["duplicate_rows"] = int(df.duplicated().sum())
    results["duplicate_appointment_id"] = int(df["appointment_id"].duplicated().sum())
    results["booking_after_appointment"] = int((df["booking_date"] > df["appointment_date"]).sum())
    results["no_shows_exceed_previous_appointments"] = int(
        (df["previous_no_shows"] > df["previous_appointments"]).sum()
    )
    results["negative_booking_lead_days"] = int((df["booking_lead_days"] < 0).sum())
    results["negative_waiting_time"] = int((df["waiting_time_minutes"] < 0).sum())
    calc_lead = (df["appointment_date"] - df["booking_date"]).dt.days
    results["lead_days_mismatch_vs_dates"] = int((calc_lead != df["booking_lead_days"]).sum())
    results["reminder_channel_set_when_not_sent"] = int(
        ((df["reminder_sent"] == "No") & df["reminder_channel"].notna()).sum()
    )
    results["waiting_time_present_for_non_attended"] = int(
        (df["appointment_outcome"].isin(["No-Show", "Cancelled"]) & df["waiting_time_minutes"].notna()).sum()
    )
    return results


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a simple per-column missing-value summary."""
    report = df.isna().sum().reset_index()
    report.columns = ["column", "missing_count"]
    report["missing_pct"] = (report["missing_count"] / len(df) * 100).round(2)
    report["allowed_missing"] = report["column"].isin(NULLABLE_COLUMNS)
    return report.sort_values("missing_count", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    path = "data/raw/HealthConnect_Appointment_Data.csv"
    print(f"Loading: {path}")
    df = load_data(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.\n")

    issues = validate_schema(df)
    print("Schema issues:", issues if issues else "None — matches Data Dictionary column set.")

    print("\nConsistency checks (failure counts — 0 is good):")
    for check, count in check_consistency(df).items():
        flag = "OK" if count == 0 else "REVIEW"
        print(f"  [{flag}] {check}: {count}")

    print("\nMissing-value report:")
    print(quality_report(df).to_string(index=False))

    print("\nappointment_outcome distribution:")
    print(df["appointment_outcome"].value_counts())
    print(df["appointment_outcome"].value_counts(normalize=True).round(3))
