# HealthConnect Clinic — ML Engineering Track

**AnalystLab Africa — Experience Lab Internship Programme**
**Project:** HealthConnect Clinic Experience Lab — Improving Patient Appointment Attendance and Healthcare Support Using Data and AI
**Track:** Machine Learning Engineering

## Central Project Question

How can HealthConnect Clinic use data and AI to reduce missed appointments and improve the patient support experience?

This repository holds the Machine Learning Engineering track's contribution: the system design, the working pipeline, and — as of Week 6 — an integrated and validated version of that pipeline with a pluggable model registry and explicit input/output contracts.

## Project Status

| Week | Status | Output |
|---|---|---|
| Week 4 | ✅ Complete | System design — problem framing, input/output definition, architecture, workflow, dependencies, reproducibility plan |
| Week 5 | ✅ Complete | Working pipeline run end-to-end against the real dataset: validation, cleaning, feature engineering, a baseline model, batch inference, 15 passing tests |
| Week 6 | ✅ Complete | Integrated pipeline: pluggable model comparison (3 candidates), input/output validation contracts, config wired in, logging added, 30 passing tests (15 unit + 15 integration) |
| Week 7 | ⏳ Planned | See `docs/PIPELINE.md` §9.6 |

## Week 6 Deliverables

- [`docs/ML_Integrated_Pipeline_Report.docx`](docs/ML_Integrated_Pipeline_Report.docx) — full Week 6 write-up (Week 5 review, integration gaps, error analysis, model comparison, validation evidence, cross-track integration, issue log)
- [`docs/Week6_Project_Summary.docx`](docs/Week6_Project_Summary.docx) — concise Week 6 summary and Week 7 focus
- [`docs/PIPELINE.md`](docs/PIPELINE.md) §9 — technical detail on everything integrated/changed this week
- [`docs/MODEL_INTERFACE.md`](docs/MODEL_INTERFACE.md) — the data/model contract other tracks build against
- [`docs/ISSUE_LOG.md`](docs/ISSUE_LOG.md) — 10 issues found this week (7 resolved, 3 open)
- `src/config.py`, updated `src/training/train.py`, `src/inference/score.py`, `tests/test_integration.py` — the actual working code

## Earlier Deliverables (still current)

- Week 5: [`docs/ML_Pipeline_Implementation_Report.docx`](docs/ML_Pipeline_Implementation_Report.docx), [`docs/Week5_Project_Summary.docx`](docs/Week5_Project_Summary.docx)
- Week 4: [`docs/ML_System_Design_Document.docx`](docs/ML_System_Design_Document.docx), [`docs/Week4_Project_Summary.docx`](docs/Week4_Project_Summary.docx), [`docs/architecture.png`](docs/architecture.png)

## Repository Structure

```
healthconnect-ml-engineering/
├── data/
│   ├── raw/               # HealthConnect_Appointment_Data.csv — never edited in place, not committed to git
│   └── processed/          # appointments_cleaned.csv, scored_batch_sample.csv — tracked as evidence
├── notebooks/               # Reserved for future EDA/experimentation notebooks
├── src/
│   ├── config.py             # Week 6: get_config() / get_logger() — wires config.yaml + logging into every module
│   ├── data/                 # validate.py (schema, consistency, + Week 6 feature-contract check), clean.py
│   ├── features/              # build_features.py (target + engineered features; Week 6: drops redundant raw history columns)
│   ├── training/               # train.py (Week 6: pluggable MODEL_FACTORY, comparison, candidate promotion)
│   ├── inference/               # score.py (Week 6: loads recommended candidate, validates output before saving)
│   └── monitoring/             # monitor.py (still a design stub — Week 7)
├── tests/
│   ├── test_pipeline.py         # 15 Week 5 unit tests
│   └── test_integration.py      # 15 Week 6 integration tests (config, model registry, contracts, reproducibility)
├── models/                       # Versioned model artefacts (.joblib), model_registry_log.csv, model_comparison_*.json
├── docs/                          # Design + weekly reports, diagrams, pipeline/model/issue documentation
├── config.yaml                    # Now actually loaded — see src/config.py
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
python -m src.training.train      # trains 3 candidate models, compares, registers the recommended one
python -m src.inference.score     # loads the recommended model, validates I/O, batch-scores appointments
python -m pytest tests/ -v        # 30 tests should pass
```

Full explanation of every stage, real data-quality findings, error analysis
of the Week 5 baseline, the model comparison results, and testing evidence:
see [`docs/PIPELINE.md`](docs/PIPELINE.md).

## System Overview

The system separates an **offline training pipeline** (validation → cleaning →
feature engineering → model comparison → model registry) from an **online
batch inference pipeline** (load the recommended model → validate the input
contract → score → validate the output → risk tier). See
`docs/ML_System_Design_Document.docx` for the original Week 4 architecture,
`docs/MODEL_INTERFACE.md` for the exact contracts, and `docs/PIPELINE.md` for
implementation detail.

## Key Decisions

- **Target:** binary — `is_no_show` (1 = No-Show, 0 = Attended). `Cancelled`
  rows are dropped from the modelling set entirely (5.3% of raw data).
- **Excluded features:** `waiting_time_minutes` (never available before an
  appointment happens), plus (Week 6) `previous_appointments`/
  `previous_no_shows` — redundant with the engineered `no_show_rate_history`
  and shown to cause multicollinearity in the baseline (see `ISSUE_LOG.md`).
- **Model:** 3 candidates compared on the same time-aware split;
  `refined_logreg` recommended, mainly for interpretability at equal
  performance — **not yet approved for production**, pending Data Science
  track review (see `MODEL_INTERFACE.md` §4).
- **Split:** time-aware (chronological on `appointment_date`), not random.
- **Cadence:** daily batch scoring remains the working assumption.

## Data Sources

- `HealthConnect_Appointment_Data.csv` — 5,000 fictional, anonymised appointment
  records (provided by AnalystLab Africa). Kept in `data/raw/` locally, **not
  committed to git** (see `.gitignore`) — uploaded to Google Drive per the
  submission requirements instead.
- `HealthConnect_Data_Dictionary` — variable definitions (provided by AnalystLab Africa).

Original resources are never overwritten; all cleaned/processed/scored data is
saved separately under `data/processed/`.

## Testing

`pytest tests/ -v` → **30 passed** (15 unit + 15 integration). New in Week 6:
model-registry tests, input/output contract tests, and a reproducibility test
that actually shells out and runs the documented commands above rather than
just asserting the code "should" work. See `docs/PIPELINE.md` §9.4.

## Cross-Track Integration (Week 6)

This track's Week 6 work is built around the dependency flagged in Week 5:
**Data Science → ML Engineering, candidate model informs pipeline
integration.** Since this internship is being completed independently, the
Data Science side of that exchange (error analysis + an improved model) was
completed within this track to produce a concrete artefact to integrate —
kept clearly labelled as such throughout (`docs/ISSUE_LOG.md`, `MODEL_INTERFACE.md`
§4) rather than presented as a separate team's output. The pipeline's
`MODEL_FACTORY` interface (§`MODEL_INTERFACE.md`) is what a genuinely
separate Data Science contribution would plug into.

## Next Steps (Week 7)

- Integrate a genuinely separate Data Science candidate model if/when
  available, alongside the existing comparison (not a replacement).
- Begin implementing `src/monitoring/monitor.py` — data-drift checks first.
- Fairness/bias testing on `gender` and `age` (carried forward, still open).
- Broader edge-case testing (see `docs/PIPELINE.md` §9.6).

