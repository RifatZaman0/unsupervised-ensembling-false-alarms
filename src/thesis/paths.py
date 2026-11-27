"""Central paths. Raw data, the score cache and results/ live in this repository.

Set THESIS_PROJECT_DIR to read and write results/ (and tdk/main.tex) in another folder."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_RAW = REPO / "data" / "raw"
CACHE = REPO / "cache" / "scores"
FROZEN_CONFIG = REPO / "frozen_config.json"

PROJECT = Path(os.environ.get("THESIS_PROJECT_DIR", REPO))
RESULTS = PROJECT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
