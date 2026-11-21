from dataclasses import dataclass
from pathlib import Path

# Resolves to the root 'fleetguard-ai/' directory regardless of current working directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    database_dir: Path = PROJECT_ROOT / "data" / "database"
    models_dir: Path = PROJECT_ROOT / "models"
    manuals_dir: Path = PROJECT_ROOT / "docs" / "manuals"


PATHS = Paths()