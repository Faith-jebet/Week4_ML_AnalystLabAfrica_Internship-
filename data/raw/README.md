# Raw Data

`HealthConnect_Appointment_Data.csv` (the original, provided dataset) is kept
here locally but is **not committed to Git** — see `.gitignore`. It is:

- Uploaded to the project's Google Drive folder per the submission requirements.
- Never edited in place. All cleaning happens in `src/data/clean.py` and writes
  to `data/processed/`, which IS tracked.

If you're setting this repo up fresh, place `HealthConnect_Appointment_Data.csv`
in this folder before running any pipeline script.
