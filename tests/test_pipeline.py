"""
Basic tests for the HealthConnect ML Engineering pipeline components.

Week 5 scope: these are smoke/unit tests confirming each component behaves
as designed against real (and synthetic edge-case) data — not exhaustive
coverage. Run with: pytest tests/ -v
"""

import pandas as pd
import pytest

from src.data.validate import load_data, validate_schema, check_consistency, EXPECTED_COLUMNS
from src.data.clean import clean_appointments
from src.features.build_features import build_target, build_feature_matrix, EXCLUDED_COLUMNS

DATA_PATH = "data/raw/HealthConnect_Appointment_Data.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    return clean_appointments(raw_df)


@pytest.fixture(scope="module")
def targeted_df(cleaned_df):
    return build_target(cleaned_df)


# ---------- validate.py ----------

def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("data/raw/does_not_exist.csv")


def test_schema_matches_data_dictionary(raw_df):
    assert validate_schema(raw_df) == []


def test_all_expected_columns_present(raw_df):
    assert set(EXPECTED_COLUMNS).issubset(set(raw_df.columns))


def test_no_duplicate_appointment_ids(raw_df):
    checks = check_consistency(raw_df)
    assert checks["duplicate_appointment_id"] == 0


def test_no_shows_never_exceed_previous_appointments(raw_df):
    checks = check_consistency(raw_df)
    assert checks["no_shows_exceed_previous_appointments"] == 0


# ---------- clean.py ----------

def test_cleaning_removes_all_missing_distance(cleaned_df):
    assert cleaned_df["distance_to_clinic_km"].isna().sum() == 0


def test_cleaning_removes_all_missing_reminder_channel(cleaned_df):
    assert cleaned_df["reminder_channel"].isna().sum() == 0


def test_cleaning_preserves_row_count(raw_df, cleaned_df):
    assert len(raw_df) == len(cleaned_df)


def test_distance_was_missing_flag_matches_original_nulls(raw_df, cleaned_df):
    assert cleaned_df["distance_was_missing"].sum() == raw_df["distance_to_clinic_km"].isna().sum()


# ---------- build_features.py ----------

def test_cancelled_rows_dropped_from_target(cleaned_df, targeted_df):
    n_cancelled = (cleaned_df["appointment_outcome"] == "Cancelled").sum()
    assert len(targeted_df) == len(cleaned_df) - n_cancelled


def test_is_no_show_target_is_binary(targeted_df):
    assert set(targeted_df["is_no_show"].unique()).issubset({0, 1})


def test_excluded_columns_never_leak_into_features(targeted_df):
    features = build_feature_matrix(targeted_df)
    leaked = [c for c in EXCLUDED_COLUMNS if c in features.columns]
    assert leaked == [], f"Excluded columns leaked into feature matrix: {leaked}"


def test_no_show_rate_history_bounded_0_to_1(targeted_df):
    features = build_feature_matrix(targeted_df)
    assert features["no_show_rate_history"].between(0, 1).all()


def test_first_time_patient_has_zero_history_rate(targeted_df):
    features = build_feature_matrix(targeted_df)
    first_timers = features[features["is_first_time_patient"] == 1]
    assert (first_timers["no_show_rate_history"] == 0).all()


def test_feature_matrix_has_no_missing_values(targeted_df):
    features = build_feature_matrix(targeted_df)
    assert features.isna().sum().sum() == 0
