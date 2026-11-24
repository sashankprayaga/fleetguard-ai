from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT_DIR
    raw_data: Path = ROOT_DIR / "data" / "raw"
    processed_data: Path = ROOT_DIR / "data" / "processed"
    database_dir: Path = ROOT_DIR / "data" / "database"
    sqlite_db: Path = ROOT_DIR / "data" / "database" / "fleetguard.db"
    mlflow_db: Path = ROOT_DIR / "data" / "database" / "mlflow.db"
    models_dir: Path = ROOT_DIR / "models"
    mlruns_dir: Path = ROOT_DIR / "models" / "mlruns"
    manuals_dir: Path = ROOT_DIR / "data" / "manuals"


PATHS = Paths()