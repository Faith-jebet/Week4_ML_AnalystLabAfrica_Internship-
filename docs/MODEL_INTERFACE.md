# Data / Model Interface Documentation — Week 6

This is the contract other tracks (primarily Data Science) need to know
about to plug into this pipeline, and the contract this pipeline enforces
on itself before trusting a model's output.

## 1. Feature Matrix Contract (input to any model)

Any model registered in `src/training/train.py`'s `MODEL_FACTORY` receives
the output of `src/features/build_features.build_feature_matrix()`:

- **Shape:** one row per appointment, one column per feature (30 columns
  as of Week 6 — see `docs/PIPELINE.md` for the current list).
- **Dtype:** entirely numeric (categoricals are one-hot encoded upstream);
  no NaNs (enforced by `validate_training_inputs` at training time and
  `validate_feature_matrix_contract` at inference time).
- **Excluded by design, never present:** `waiting_time_minutes`,
  `appointment_id`, `patient_id`, `appointment_outcome`,
  `previous_appointments`, `previous_no_shows` (Week 6 — see
  `ISSUE_LOG.md` #1), raw `booking_date`/`appointment_date`.
- **Target column:** `is_no_show` (0/1) is present in the training-time
  matrix and must be dropped before calling `model.predict()` — both
  `train.py` and `score.py` do this explicitly.

Any candidate model the Data Science track proposes must be trainable on
exactly this matrix. If a new feature is needed, it belongs in
`build_features.py`, not bolted on separately, so both training and
inference stay in sync (this is the whole point of the shared pipeline).

## 2. Model Interface Contract (what a registered model must support)

Every entry in `MODEL_FACTORY` is a zero-argument callable returning an
unfitted scikit-learn-compatible estimator implementing:

- `.fit(X, y)`
- `.predict(X)`
- `.predict_proba(X)` → returns an array with a column for the positive
  class (`is_no_show == 1`) at index `[:, 1]`

This is deliberately the standard scikit-learn estimator interface, so
any model the Data Science track produces (as long as it exposes these
three methods — true for every sklearn classifier, and for XGBoost/
LightGBM wrappers too) can be added to `MODEL_FACTORY` with one line,
with no other pipeline code changing.

## 3. Model Output Contract (what inference produces)

`src/inference/score.py`'s `score_batch()` returns a DataFrame with:

| Column | Type | Guarantee |
|---|---|---|
| `appointment_id` | string | Present, non-null, unique (enforced by `validate_scores`) |
| `no_show_probability` | float | In `[0, 1]`, non-null (enforced by `validate_scores`) |
| `risk_tier` | string | One of `Low` / `Medium` / `High`, thresholds from `config.yaml` `risk_tiers` |
| `model_version` | string | Added by the `__main__` block — identifies which registry artefact produced the score |

This is the contract the Data Analytics track's dashboard and any future
reminder-workflow consumer should build against.

## 4. Model Registry Contract

Every training run appends one row per candidate model to
`models/model_registry_log.csv` with: `model_version`, `training_date`,
`algorithm`, `feature_list_ref`, `train_rows`, three metrics
(`val_metric_recall`, `val_metric_f1`, `val_metric_roc_auc`),
`approved_by`, and `notes`. The `approved_by` field is the only signal of
promotion status — `"Week 6 integration — recommended candidate for Data
Science review"` versus `"...evaluated, not selected"` versus the Week 5
`"not approved — Week 5 pipeline smoke test only"`. **No model in this
registry has been approved for production** — that decision sits with
the Data Science track and, ultimately, whoever owns the HealthConnect
project sign-off, not with this pipeline.

## 5. What Changed Since Week 5

- Week 5: one hardcoded model, no formal interface, no output validation.
- Week 6: pluggable `MODEL_FACTORY`, explicit input/output contracts
  enforced with exceptions (not silent failures), and a documented,
  versioned handoff point (`approved_by` field) for the Data Science
  track to review and eventually override with their own candidate.
