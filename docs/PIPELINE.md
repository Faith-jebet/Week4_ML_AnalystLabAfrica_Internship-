# HealthConnect ML Pipeline — Documentation

**Track:** Machine Learning Engineering
**Status:** Week 5 — initial implementation (data processing + feature pipeline +
baseline model integration, run end-to-end against the real dataset)

## 1. Pipeline Stages

| Stage | Module | What it does |
|---|---|---|
| Validation | `src/data/validate.py` | Loads the CSV, checks schema against the Data Dictionary, runs cross-field consistency checks, produces a missing-value report |
| Cleaning | `src/data/clean.py` | Imputes/fills missing values with documented, reproducible rules; never edits `data/raw` |
| Feature engineering | `src/features/build_features.py` | Builds the binary target, drops `Cancelled` rows, derives behavioural features, one-hot encodes categoricals, excludes non-production-available columns |
| Training (smoke test) | `src/training/train.py` | Time-aware train/test split, fits a baseline Logistic Regression, evaluates it, saves a versioned model artefact + registry log row |
| Inference | `src/inference/score.py` | Loads the latest model, scores a batch through the same cleaning/feature path, outputs probability + risk tier |
| Tests | `tests/test_pipeline.py` | 15 tests covering validation, cleaning, and feature-engineering behaviour |

Run order for a full pipeline pass:
```bash
python -m src.data.validate      # inspect data quality
python -m src.data.clean         # produce data/processed/appointments_cleaned.csv
python -m src.features.build_features   # sanity-check feature matrix
python -m src.training.train     # fit + register the baseline model
python -m src.inference.score    # score a batch with the saved model
python -m pytest tests/ -v       # run the test suite
```

## 2. Data Quality Findings (real dataset, 5,000 rows)

Run via `python -m src.data.validate`:

- **No duplicate rows, no duplicate `appointment_id`** — primary key integrity holds.
- **Missing values** are limited to three columns and all are structurally
  explainable, not random data-entry errors:
  - `reminder_channel`: 1,366 missing (27.3%) — **every** missing value
    corresponds exactly to `reminder_sent == "No"` (0 mismatches found).
    This is expected, not a quality problem.
  - `distance_to_clinic_km`: 90 missing (1.8%).
  - `waiting_time_minutes`: 60 missing (1.2%).
- **`booking_lead_days` is fully consistent** with `appointment_date − booking_date`
  for all 5,000 rows (0 mismatches) — the derived field can be trusted.
- **`previous_no_shows` never exceeds `previous_appointments`** — history fields are internally consistent.
- **Discrepancy found vs. the Data Dictionary:** the dictionary states
  `booking_date`/`appointment_date` are in "ISO format". The actual file uses
  `M/D/YYYY` (US format, e.g. `2/6/2025`), not ISO 8601 (`YYYY-MM-DD`).
  `validate.py` parses against the observed format, not the documented one.
- **Notable data-generation artefact:** `waiting_time_minutes` is populated for
  2,647 appointments whose outcome is `No-Show` or `Cancelled` — logically odd,
  since a patient who didn't show up shouldn't have a recorded waiting time.
  Its distribution is also statistically indistinguishable across `Attended`
  (mean 24.3), `No-Show` (mean 24.2), and `Cancelled` (mean 23.2) — consistent
  with it being generated independently of outcome in this synthetic dataset.
  This reinforces, rather than weakens, the Week 4 decision to exclude it: it
  carries no real predictive signal here, and in a real system it would not be
  available before the appointment happens regardless.
- **Class balance for the modelling target:** after dropping `Cancelled`,
  `No-Show` is 51.2% and `Attended` is 48.8% of the remaining 4,737 rows — close
  to balanced. (Note: this is balance in the *cleaned modelling set*; the raw
  three-way outcome split is No-Show 48.5% / Attended 46.3% / Cancelled 5.3%.)

## 3. Feature Engineering Decisions

| Feature | Logic | Why |
|---|---|---|
| `no_show_rate_history` | `previous_no_shows / previous_appointments`, 0 for no history | Patient's own track record — plausible strongest signal |
| `is_first_time_patient` | 1 if `previous_appointments == 0` | Distinguishes "no history" from "clean history" (both would otherwise show rate 0) |
| `is_weekend_appointment` | 1 if `appointment_day` is Sat/Sun | Tests a schedule-based hypothesis |
| `long_lead_time` | 1 if `booking_lead_days` ≥ 75th percentile | Long booking-to-appointment gaps as a candidate no-show driver |
| `distance_was_missing` | carried from cleaning | Preserves the "was this imputed" signal after filling the median |

**Excluded from the feature matrix (never eligible as model inputs):**
`waiting_time_minutes`, `appointment_id`, `patient_id`, `appointment_outcome`
(replaced by the derived `is_no_show` target), plus raw date columns (used only
for the time-aware split, not as direct features in this baseline).

## 4. Train/Test Strategy

Chronological (time-aware) split on `appointment_date`, not random — 80% earliest
appointments for training, most recent 20% for testing. This matches how the
system will actually be used: always predicting appointments that come after
whatever it was trained on. A random split would let the model implicitly see
"future" patterns during training and overstate real-world performance.

## 5. Baseline Model (smoke test, not a final model)

A plain Logistic Regression was fit purely to prove the pipeline runs end-to-end.
**Model selection is the Data Science track's responsibility** — this is not a
proposed production model.

Latest run (`python -m src.training.train`):

| Metric | Value |
|---|---|
| Accuracy | 0.616 |
| Precision | 0.622 |
| Recall | 0.627 |
| F1 | 0.625 |
| ROC-AUC | 0.677 |

These numbers exist to confirm the pipeline produces a working, evaluable model
— not to claim this is a good no-show classifier. Given the near-balanced
classes, accuracy is a reasonably fair headline metric here, but recall/F1/ROC-AUC
are tracked because that will matter more once the Data Science track works with
the true (imbalanced, three-way) outcome distribution.

## 6. Testing Evidence

`python -m pytest tests/ -v` → **15 passed, 0 failed** (see `tests/test_pipeline.py`).
Covers: missing-file error handling, schema validation, consistency checks,
cleaning completeness (zero missing values post-clean), row-count preservation,
target construction (Cancelled correctly dropped, binary target), and feature
leakage prevention (excluded columns verified absent from the output matrix).

## 7. Implementation Issues & Next Steps

- **Config file (`config.yaml`) is not yet wired into the modules** — paths and
  thresholds are still hard-coded in each script for Week 5. Loading from
  `config.yaml` is a Week 6 cleanup task.
- **Model registry is intentionally lightweight** (versioned `.joblib` file +
  CSV metadata log) rather than a dedicated tool — appropriate for this stage,
  revisit if the project scales.
- **Batch inference currently scores a held-out slice of historical data**
  as a stand-in for "upcoming appointments" — wiring to a real daily booking
  feed is out of scope until later weeks.
- **Feature list alignment** between train and test (and eventually between
  train and live inference) is handled with `DataFrame.align`/`reindex` — works
  for this dataset's category set, but a rare category appearing only at
  inference time would still need a more robust encoding strategy (e.g. a
  fitted `OneHotEncoder` object saved alongside the model) before production use.
- **Dependency on the Data Science track:** this pipeline currently makes its
  own target/feature decisions (documented above) to stay unblocked. If the
  Data Science track's Week 5 output changes any of these (e.g. a different
  treatment of `Cancelled`, or additional features), the shared modules in
  `src/features/build_features.py` will need to be updated to match — see
  Cross-Track Collaboration in the Week 5 report.
