"""Central paths. Code, raw data and the score cache live in this repo (outside OneDrive);
results/ and writing live in the OneDrive project folder next to the thesis documents."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
CACHE = REPO / "cache" / "scores"
FROZEN_CONFIG = REPO / "frozen_config.json"

PROJECT = Path(os.environ.get(
    "THESIS_PROJECT_DIR", r"C:\Users\rifat\OneDrive\Documents\thesis-docs"))
RESULTS = PROJECT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
