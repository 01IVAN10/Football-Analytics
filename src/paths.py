"""Project paths in one place.

Kept separate from data_loader.py, which imports statsbombpy: the web app and the
charts only read local files and should not depend on the API client.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW = PROJECT_ROOT / "data" / "raw"              # StatsBomb downloads (not in git)
PROCESSED = PROJECT_ROOT / "data" / "processed"  # metric tables (not in git)
APP_DATA = PROJECT_ROOT / "data" / "app"         # slim data read by the app (in git)
EXPORT = PROJECT_ROOT / "data" / "export"        # open dataset, CSV (in git)
FIGURES = PROJECT_ROOT / "reports" / "figures"   # PNGs from the CLI (not in git)
ASSETS = PROJECT_ROOT / "assets"                 # static files (StatsBomb logo)
