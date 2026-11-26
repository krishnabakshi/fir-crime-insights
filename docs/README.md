# Docs

* `CSL236_MidTerm_Project_Guidelines.pdf` and `CSL236_EndTerm_Project_Guidelines.pdf`: the course guidelines this
  project follows. They are kept locally but **git-ignored** (`docs/*.pdf` in `.gitignore`) until the instructor
  confirms they may be shared publicly. To publish them, delete that line from `.gitignore` and commit the PDFs.
* `screenshots/`: images used in the main README (UI, `/docs`, MLflow, pytest, CI).

## How each guideline requirement is covered

| Guideline | Where |
|---|---|
| Mid-term A-H preprocessing stages, each justified | `notebooks/FIR_Crime_Insights.ipynb` sections 2-4 and 9.2 (mapping table in the first cell) |
| Split before fitting any imputer / encoder / scaler | `src/model_pipeline.py` (all learned steps inside one Pipeline), notebook 4.6 |
| Team names and roll numbers in the first cell | notebook first cell (fill in) |
| >= 5 features, >= 1,000 rows | 15 features, 2,139 labelled rows of 2,400 FIRs |
| Data contract enforced by Pydantic, 422 on bad input | `app.py` `CaseInput`, `tests/test_app.py` |
| MLflow, 3+ runs, 2+ algorithms, CLI hyper-parameters | `train.py --sweep` (6 runs, 4 algorithms) |
| Model loaded once at start-up | `app.py` module-level `MODEL = joblib.load(...)` |
| FastAPI `/health`, `/predict`, UI at `/` with tips | `app.py`, `static/index.html` |
| 4+ pytest tests incl. domain logic and 422 | `tests/test_app.py` (17 tests) + `tests/test_preprocessing.py` (20 tests) |
| Docker from `python:3.11-slim` with serve-only requirements | `Dockerfile`, `requirements-serve.txt` |
| CI on push to main: install, train, pytest | `.github/workflows/ci.yml` |
| Prediction inputs logged for drift | `logs/predictions.jsonl`, `logs/sample_predictions.jsonl` |
| Bonus: drift check, explainability | `scripts/drift_check.py`; permutation importance in `/model-info` and the UI |
