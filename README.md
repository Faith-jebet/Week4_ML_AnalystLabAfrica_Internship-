# HealthConnect Clinic — ML Engineering Track

**AnalystLab Africa — Experience Lab Internship Programme**
**Project:** HealthConnect Clinic Experience Lab — Improving Patient Appointment Attendance and Healthcare Support Using Data and AI
**Track:** Machine Learning Engineering

## Central Project Question

How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

This repository holds the Machine Learning Engineering track's contribution: the system design, and the working pipeline, needed to turn a no-show prediction model into a reproducible, versioned, production-style service.

## Project Status

| Week | Status | Output |
|---|---|---|
| Week 4 | ✅ Complete | System design — problem framing, input/output definition, architecture, workflow, dependencies, reproducibility plan |
| Week 5 | ✅ Complete | Working pipeline run end-to-end against the real dataset: validation, cleaning, feature engineering, a baseline model integration, batch inference, and a passing test suite |
| Week 6 | ⏳ Planned | See "Next Steps" below |

## Week 5 Deliverables

- [`docs/ML_Pipeline_Implementation_Report.docx`](docs/ML_Pipeline_Implementation_Report.docx) — full Week 5 write-up (Week 4 review, data-processing pipeline, feature engineering, baseline model integration, testing evidence, cross-track collaboration, updated risk register)
- [`docs/Week5_Project_Summary.docx`](docs/Week5_Project_Summary.docx) — concise Week 5 summary and Week 6 focus
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — technical pipeline documentation, data-quality findings, and testing evidence
- `src/`, `tests/`, `data/processed/`, `models/` — the actual working code and its output (see below)

## Week 4 Deliverables (still current — design foundation)

- [`docs/ML_System_Design_Document.docx`](docs/ML_System_Design_Document.docx)
- [`docs/Week4_Project_Summary.docx`](docs/Week4_Project_Summary.docx)
- [`docs/architecture.png`](docs/architecture.png)

## Repository Structure

```
healthconnect-ml-engineering/
├── data/
│   ├── raw/               # HealthConnect_Appointment_Data.csv — never edited in place, not committed to git
│   └── processed/          # appointments_cleaned.csv, scored_batch_sample.csv — tracked as evidence
├── notebooks/               # Reserved for future EDA/experimentation notebooks
├── src/
│   ├── data/                # validate.py (schema + consistency checks), clean.py (imputation)
│   ├── features/             # build_features.py (target + engineered features, shared by train & inference)
│   ├── training/              # train.py (time-aware split, baseline model, registry logging)
│   ├── inference/             # score.py (batch scoring using the same feature pipeline)
│   └── monitoring/            # monitor.py (design stub — Week 6+)
├── tests/                      # test_pipeline.py — 15 tests, run with pytest
├── models/                     # Versioned model artefacts (.joblib) + model_registry_log.csv
├── docs/                       # Design + Week 5 reports, diagrams, pipeline documentation
├── config.yaml                 # Central settings reference (not yet wired into modules — see docs/PIPELINE.md)
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
python -m src.training.train      # fits + registers a baseline model (smoke test)
python -m src.inference.score     # batch-scores appointments with the saved model
python -m pytest tests/ -v        # 15 tests should pass
```

Full explanation of each stage, real data-quality findings (including a
Data Dictionary discrepancy discovered while implementing this), feature
rationale, baseline metrics, and testing evidence: see
[`docs/PIPELINE.md`](docs/PIPELINE.md).

## System Overview

The system separates an **offline training pipeline** (validation → cleaning →
feature engineering → model training → evaluation → model registry) from an
**online batch inference pipeline** (load latest model → same feature pipeline
→ score → risk tier output). See `docs/ML_System_Design_Document.docx` for the
full Week 4 architecture and `docs/PIPELINE.md` for how it was implemented.

## Key Decisions

- **Target:** binary — `is_no_show` (1 = No-Show, 0 = Attended). `Cancelled`
  rows are dropped from the modelling set entirely (5.3% of raw data), not
  encoded as 0 — confirmed as the right call after Week 5 data exploration
  (see Cross-Track Collaboration in the Week 5 report).
- **Excluded feature:** `waiting_time_minutes` — never available before an
  appointment happens. Week 5 found it's even populated for No-Show/Cancelled
  rows in this dataset with a near-identical distribution across outcomes,
  reinforcing that it carries no real signal and must stay excluded.
- **Split:** time-aware (chronological on `appointment_date`), not random.
- **Cadence:** daily batch scoring remains the working assumption.
- **Model management:** versioned `.joblib` files + a CSV metadata log.

## Data Sources

- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymised appointment
  records (provided by AnalystLab Africa). Kept in `data/raw/` locally, **not
  committed to git** (see `.gitignore`) — uploaded to Google Drive per the
  submission requirements instead.
- `HealthConnect_Data_Dictionary` — variable definitions (provided by AnalystLab Africa).

Original resources are never overwritten; all cleaned/processed/scored data is
saved separately under `data/processed/`.

## Testing

`pytest tests/ -v` → **15 passed**. Covers schema validation, consistency
checks, cleaning completeness, target construction, and feature-leakage
prevention. See `docs/PIPELINE.md` §6 for details.

## Next Steps (Week 6)

- Wire `config.yaml` into the modules instead of hard-coded paths/settings.
- Replace the baseline Logistic Regression with whatever model the Data
  Science track finalises; keep the registry/inference scaffolding as-is.
- Save a fitted encoder object alongside the model so inference-time category
  handling is robust to categories unseen in training.
- Begin wiring batch inference to a simulated "daily new appointments" feed
  rather than a held-out historical slice.

---

