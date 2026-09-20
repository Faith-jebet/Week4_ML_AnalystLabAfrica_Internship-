# HealthConnect Clinic — ML Engineering Track

**AnalystLab Africa — Experience Lab Internship Programme**
**Project:** HealthConnect Clinic Experience Lab — Improving Patient Appointment Attendance and Healthcare Support Using Data and AI
**Track:** Machine Learning Engineering

## Central Project Question

How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

This repository holds the Machine Learning Engineering track's contribution: system design (Week 4), a working pipeline (Week 5), integration with a pluggable model registry (Week 6), and — as of Week 7 — that pipeline tested against adversarial/edge-case inputs, with two real bugs found and fixed.

## Project Status

| Week | Status | Output |
|---|---|---|
| Week 4 | ✅ Complete | System design |
| Week 5 | ✅ Complete | Working pipeline, 15 passing tests |
| Week 6 | ✅ Complete | Integrated pipeline: pluggable model comparison, I/O validation, config, logging, 30 passing tests |
| Week 7 | ✅ Complete | Pipeline tested against edge cases; 2 real train/serve skew bugs found and fixed; 37 passing tests |
| Week 8 | ⏳ Planned | See `docs/PIPELINE.md` §10.8 |

## Week 7 Deliverables

- [`docs/ML_Pipeline_Testing_Reliability_Report.docx`](docs/ML_Pipeline_Testing_Reliability_Report.docx) — full Week 7 write-up (Week 6 review, test plan, test results table, root cause analysis, fix implementation, retest evidence, HC-POD cross-track testing, updated risk register)
- [`docs/Week7_Project_Summary.docx`](docs/Week7_Project_Summary.docx) — concise summary and Week 8 focus
- [`docs/PIPELINE.md`](docs/PIPELINE.md) §10 — full technical detail on the two bugs found and fixed this week
- [`docs/ISSUE_LOG.md`](docs/ISSUE_LOG.md) — updated with 3 new Week 7 issues (#11, #12 resolved; #13 open)
- `tests/test_integration.py` — 7 new edge-case regression tests (single-row batch, skewed batch, empty batch, all-Cancelled batch, unseen category, missing column, fixed threshold)
- [`notebooks/HealthConnect_ML_Pipeline_Walkthrough.ipynb`](notebooks/HealthConnect_ML_Pipeline_Walkthrough.ipynb) — a supporting, fully-executed notebook that runs the entire pipeline end-to-end (load → validate → clean → engineer features → train/compare 3 models → score a batch → reproduce and fix the two Week 7 bugs live → run the test suite), so a reviewer can see it all work in one place without opening five separate files

## Earlier Deliverables (still current)

- Week 6: [`docs/ML_Integrated_Pipeline_Report.docx`](docs/ML_Integrated_Pipeline_Report.docx), [`docs/Week6_Project_Summary.docx`](docs/Week6_Project_Summary.docx), [`docs/MODEL_INTERFACE.md`](docs/MODEL_INTERFACE.md)
- Week 5: [`docs/ML_Pipeline_Implementation_Report.docx`](docs/ML_Pipeline_Implementation_Report.docx), [`docs/Week5_Project_Summary.docx`](docs/Week5_Project_Summary.docx)
- Week 4: [`docs/ML_System_Design_Document.docx`](docs/ML_System_Design_Document.docx), [`docs/Week4_Project_Summary.docx`](docs/Week4_Project_Summary.docx), [`docs/architecture.png`](docs/architecture.png)

## Repository Structure

```
healthconnect-ml-engineering/
├── data/
│   ├── raw/               # HealthConnect_Appointment_Data.csv — never edited in place, not committed to git
│   └── processed/          # appointments_cleaned.csv, scored_batch_sample.csv — tracked as evidence
├── notebooks/               # HealthConnect_ML_Pipeline_Walkthrough.ipynb — supporting artefact, run end-to-end
├── src/
│   ├── config.py             # get_config() / get_logger()
│   ├── data/                 # validate.py, clean.py
│   ├── features/              # build_features.py — Week 7: fit_feature_params() fixes train/serve skew (see docs/ISSUE_LOG.md #11/#12)
│   ├── training/               # train.py — fits and saves feature_params.json alongside each model
│   ├── inference/               # score.py — loads feature_params.json + model.feature_names_in_, handles empty batches explicitly
│   └── monitoring/             # monitor.py (still a design stub — Week 8)
├── tests/
│   ├── test_pipeline.py         # 15 Week 5 unit tests
│   └── test_integration.py      # 22 tests: 15 Week 6 integration + 7 Week 7 edge-case/regression
├── models/                       # Versioned model artefacts, model_registry_log.csv, feature_params.json, model_comparison_*.json
├── docs/                          # Design + weekly reports, diagrams, pipeline/model/issue documentation
├── config.yaml
└── requirements.txt
```

## Running the Pipeline

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.validate       # data-quality report against the real dataset
python -m src.data.clean          # writes data/processed/appointments_cleaned.csv
python -m src.features.build_features   # sanity-checks the feature matrix
python -m src.training.train      # trains 3 candidates, fits+saves feature_params.json, registers the recommended model
python -m src.inference.score     # loads the recommended model + feature_params.json, validates I/O, batch-scores
python -m pytest tests/ -v        # 37 tests should pass
```

Full technical detail — data-quality findings, error analysis, model
comparison, and the Week 7 edge-case testing that found and fixed two real
bugs — is in [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Key Decisions

- **Target:** binary — `is_no_show` (1 = No-Show, 0 = Attended). `Cancelled`
  rows dropped from modelling entirely.
- **Excluded features:** `waiting_time_minutes`, plus `previous_appointments`/
  `previous_no_shows` (redundant with `no_show_rate_history`).
- **Model:** 3 candidates compared; `refined_logreg` recommended, mainly for
  interpretability at equal performance — **not yet approved for production**.
- **Feature encoding (Week 7):** category levels and the `long_lead_time`
  threshold are now fit once on the training set and reused unchanged at
  inference, fixing two real bugs where small/skewed batches were encoded
  inconsistently with how the model was trained. See `docs/ISSUE_LOG.md` #11/#12.

## Data Sources

- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymised appointment
  records (provided by AnalystLab Africa). Kept in `data/raw/` locally, not
  committed to git — uploaded to Google Drive per submission requirements.
- `HealthConnect_Data_Dictionary` — variable definitions.

## Testing

`pytest tests/ -v` → **37 passed** (15 unit + 15 Week 6 integration + 7 Week 7
edge-case). New in Week 7: tests that deliberately probe single-row batches,
skewed batches, empty batches, all-Cancelled batches, unseen categories, and
missing columns — the kind of inputs a real deployment would actually see,
which full-dataset-only testing in Weeks 5–6 never exercised. See
`docs/PIPELINE.md` §10.2 for the full before/after results table.

## Cross-Track Testing (Week 7)

Per the HC-POD testing matrix (Data Science ↔ ML Engineering: "test whether
the integrated pipeline correctly handles model requirements and expected
inputs/outputs"), this week's testing directly validated that claim: every
registered model (including any future Data Science contribution added to
`MODEL_FACTORY`) now receives a feature matrix that is provably identical in
shape regardless of batch size, via `model.feature_names_in_` and
`fit_feature_params()`. Since this internship is being completed
independently, both sides of this test were completed within this track and
documented transparently rather than presented as a separate contribution —
see `docs/ML_Pipeline_Testing_Reliability_Report.docx` §6 for the full
Test → Finding → Action → Retest record.

## Next Steps (Week 8)

- Final integration with whichever model the Data Science track settles on.
- Address Issue #13 (unclear `KeyError` on a missing required column) if time allows.
- Begin fairness/bias testing on `gender` and `age` (carried forward, still open).
- Prepare final presentation materials.

---
*Tagging #AnalystLabAfrica*
