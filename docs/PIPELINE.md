# HealthConnect ML Pipeline — Documentation

**Track:** Machine Learning Engineering
<<<<<<< HEAD
**Status:** Week 6 — integrated pipeline (pluggable model registry, input/output
validation contracts, config wired in, 30 passing tests). Week 5 section below
is kept as history; see §9 for what changed.

## 1. Pipeline Stages (current, Week 6)

| Stage | Module | What it does |
|---|---|---|
| Validation | `src/data/validate.py` | Loads the CSV, checks schema against the Data Dictionary, runs cross-field consistency checks, produces a missing-value report, exposes `validate_feature_matrix_contract()` for inference-time checks |
| Cleaning | `src/data/clean.py` | Imputes/fills missing values with documented, reproducible rules; never edits `data/raw` |
| Feature engineering | `src/features/build_features.py` | Builds the binary target, drops `Cancelled` rows, derives behavioural features, one-hot encodes categoricals, excludes non-production-available AND redundant columns (Week 6) |
| Config | `src/config.py` | `get_config()` loads `config.yaml` once; `get_logger()` gives every stage consistent logging |
| Training | `src/training/train.py` | Time-aware split, trains **3 candidate models** via a pluggable `MODEL_FACTORY`, validates inputs, compares on ROC-AUC, promotes a recommended candidate, logs everything to the model registry |
| Inference | `src/inference/score.py` | Loads the latest **recommended** model (not just latest file), validates the input feature contract, scores a batch, validates output before saving |
| Unit tests | `tests/test_pipeline.py` | 15 tests — validation, cleaning, feature-engineering (Week 5, still passing unchanged) |
| Integration tests | `tests/test_integration.py` | 15 tests — config, model registry, input/output contracts, end-to-end scoring, and documented-command reproducibility (Week 6, new) |
=======
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
>>>>>>> origin/main

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
<<<<<<< HEAD

## 9. Week 6 — Integration & Validation

### 9.1 What Changed From Week 5

| Area | Week 5 | Week 6 |
|---|---|---|
| Model | 1 hardcoded Logistic Regression | 3 candidates via pluggable `MODEL_FACTORY`, compared and one promoted |
| Config | `config.yaml` written but unused | `src/config.py` loads it; `train.py`/`score.py` read from it |
| Logging | `print()` only | Shared `get_logger()`, structured log lines with timestamps |
| Input validation | None before training/scoring | `validate_training_inputs()`, `validate_feature_matrix_contract()` |
| Output validation | None | `validate_scores()` — range, nulls, duplicates, valid tiers |
| Model selection | N/A (only one model) | Explicit comparison on `selection_metric` (ROC-AUC), only promotes if the candidate beats baseline |
| Tests | 15 unit tests | +15 integration tests (30 total) |

### 9.2 Error Analysis of the Week 5 Baseline

Run against the held-out test set (948 rows): 584 correct, 184 false
positives, 180 false negatives.

- **Error rate by patient history:** first-time patients had a 44.4% error
  rate vs. 38.1% for returning patients — the model struggles more without
  history, unsurprising since `no_show_rate_history` (the strongest feature)
  is 0 by construction for first-timers.
- **Error rate by appointment type:** Diagnostic Test appointments had the
  highest false-negative rate (26.2%) vs. 16.8–18.6% for other types —
  worth flagging to the Data Science/Data Analytics tracks as a segment
  that might benefit from a type-specific feature.
- **Feature importance (baseline coefficients):** `previous_no_shows`
  (+0.55) and `no_show_rate_history` (−0.55) had opposing signs of nearly
  equal magnitude — a textbook multicollinearity signature, not a genuine
  finding about no-show behaviour. This directly motivated dropping the
  two raw history columns (see `docs/ISSUE_LOG.md` #1).
- **False-negative probabilities cluster near the decision boundary**
  (mean 0.387, 75th percentile 0.451) — these are genuinely uncertain
  cases close to 0.5, not confidently-wrong predictions, which is a
  reasonable failure mode for a baseline linear model.

### 9.3 Model Comparison Results

Three models trained on the same time-aware split (3,789 train / 948 test,
cutoff 2026-03-13), after dropping the two redundant history columns:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| `baseline_logreg` (Week 5 config, refined features) | 0.6203 | 0.6268 | 0.6294 | 0.6281 | 0.6733 |
| `refined_logreg` (+ `class_weight="balanced"`) | 0.6234 | 0.6376 | 0.6046 | 0.6206 | **0.6734** |
| `random_forest` (200 trees, max_depth=8) | 0.6224 | 0.6305 | 0.6253 | 0.6279 | 0.6724 |

**Recommended candidate: `refined_logreg`** — technically the highest
ROC-AUC, but the margin (0.6734 vs 0.6733) is not meaningfully different
from the baseline; the real justification is interpretability (no more
multicollinearity, see §9.2) at zero performance cost, not a modelling
breakthrough. `random_forest` did not outperform either linear model.

**Honest interpretation:** neither feature refinement nor a
higher-capacity model meaningfully moved performance. This suggests the
current feature set may be near its ceiling for this target with these
algorithms — a candidate for the Data Science track to investigate further
(richer features, different algorithms, or hyperparameter search) rather
than something this track's remaining scope could resolve alone. This is
logged as an open item in `docs/ISSUE_LOG.md` (#8).

### 9.4 Validation Evidence

- `python -m pytest tests/ -v` → **30 passed, 0 failed** (15 unit + 15 integration).
- `test_pipeline_reproducible_via_documented_commands` actually shells out
  and runs the exact commands in this file's §1 run order — not just an
  assertion that the code "should" work.
- `test_all_registered_models_are_fittable_and_predict_probabilities`
  fits every entry in `MODEL_FACTORY` fresh and checks output range —
  catches a broken model registration before it reaches training.
- Manual end-to-end run confirmed: `train.py` → 3 models trained, registry
  updated with 3 rows, comparison JSON saved; `score.py` → recommended
  model loaded, 471 of 500 appointments scored (29 were `Cancelled` and
  correctly excluded), output validation passed, risk tiers: 76 Low / 293
  Medium / 102 High.

### 9.5 Full Issue Log

See `docs/ISSUE_LOG.md` for all 10 issues found this week (7 resolved, 3 open).

### 9.6 Week 7 Testing Plan

- Broader test coverage: edge cases in `clean.py` (e.g. all-null distance
  for an entire age group), and a larger synthetic adversarial batch for
  `score.py` (unseen categories, empty batches).
- Once the Data Science track finalises a model, integrate it into
  `MODEL_FACTORY` alongside the existing candidates and re-run the
  comparison rather than replacing it outright, so there's a documented
  before/after.
- Begin implementing `src/monitoring/monitor.py` (still a design stub) —
  at minimum, a data-drift check comparing a new batch's `appointment_type`/
  `age_group` distribution against the training distribution.
- Fairness/bias testing on `gender` and `age` (carried-forward open item).
=======
>>>>>>> origin/main
