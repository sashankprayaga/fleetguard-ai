# FleetGuard AI — Project Status

## Current Development Phase
- **Phase 2:** Dataset Exploration, Schema Validation & Leakage Audit (Starting)

## Completed Work
- **Phase 1 (Complete):**
  - Created modular Python package structure (`src/fleetguard/`) with `pyproject.toml` editable install.
  - Built centralized path configuration (`src/fleetguard/config/settings.py`).
  - Built geospatial + powertrain-aware synthetic telemetry generator (`src/fleetguard/data/synthetic_generator.py`) covering 6 truck classes across 12 German Autobahn corridors (`250,000` rows in `data/raw/synthetic_fleet_history.parquet`).
  - Verified generator schema and German GPS bounds via `pytest` (`tests/test_synthetic_generator.py`).

## Files Created
- `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `PROJECT_STATUS.md`
- `src/fleetguard/config/settings.py`
- `src/fleetguard/data/synthetic_generator.py`
- `tests/test_synthetic_generator.py`

## Dependencies Installed
- `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `joblib`, `plotly`, `streamlit`, `pydantic`, `python-dotenv`, `pytest` (Python 3.12+ `.venv`)

## Tests Performed
- `tests/test_synthetic_generator.py` — Passed (`pytest -v`).
- Generated and verified `data/raw/synthetic_fleet_history.parquet` and `data/raw/fleet_registry.parquet`.

## Known Issues
- None.

## Remaining Work
- Phase 2: Pydantic schema validation, EDA profiling, and feature leakage/exclusion audit.
- Phases 3–12 per roadmap.

## Next Recommended Task
- Implement `src/fleetguard/data/validator.py` and `src/fleetguard/data/eda.py`, then run validation & exploratory audit on `synthetic_fleet_history.parquet`.

# FleetGuard AI — Project Status

## Current Development Phase
- **Phase 3:** Data Preprocessing & ML Baseline Models (Starting)

## Completed Work
- **Phase 1 (Complete):** Project skeleton, `Paths` config, 12-corridor German Autobahn + 6-class truck synthetic generator (`250,000` rows), and unit tests.
- **Phase 2 (Complete):**
  - Built Pydantic v2 + vectorized Pandas data contract validator (`src/fleetguard/data/validator.py`).
  - Built EDA & feature governance profiler (`src/fleetguard/data/eda.py`) separating ML features from IDs, timestamps, and GPS coordinates.
  - Audited 250,000 rows (`validation_passed: true`, `1.91%` positive class rate, `4,767` maintenance events across 6 vehicle classes).

## Files Created
- `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `PROJECT_STATUS.md`
- `src/fleetguard/config/settings.py`
- `src/fleetguard/data/synthetic_generator.py`
- `src/fleetguard/data/validator.py`
- `src/fleetguard/data/eda.py`
- `data/processed/eda_audit_report.json`
- `tests/test_synthetic_generator.py`

## Dependencies Installed
- `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `joblib`, `plotly`, `streamlit`, `pydantic`, `python-dotenv`, `pytest` (Python 3.12.4 `.venv`)

## Tests Performed
- `pytest -v` — 2 passed (`test_synthetic_generator.py`).
- `python -m fleetguard.data.eda` — Validated 250,000 rows and saved `eda_audit_report.json`.

## Known Issues
- Minor Pandas 3+ deprecation warning on `inplace=True` in `synthetic_generator.py` (easy cleanup).

## Remaining Work
- Phase 3: Temporal train/val/test split, Scikit-learn `ColumnTransformer` preprocessing pipeline, and baseline classifiers.
- Phases 4–12 per roadmap.

## Next Recommended Task
- Implement `src/fleetguard/features/preprocessing.py` and `src/fleetguard/models/baseline.py`, train baseline Logistic Regression and Random Forest models, and evaluate PR-AUC / ROC-AUC on held-out future telemetry.