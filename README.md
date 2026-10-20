# Unsupervised Model Ensembling for False Alarm Reduction in Anomaly Detection

BSc thesis code (BME VIK). Planning documents, `results/` and writing live in the OneDrive project folder
(`C:\Users\rifat\OneDrive\Documents\thesis-docs`, see `src/thesis/paths.py`).

## Setup

```
"C:\Program Files\Python311\python.exe" -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pytest
```

Python 3.11. pandas stays below 3 because adtk 0.6.2 does not run on pandas 3.

## Layout

- `src/thesis/`: loaders, detectors (`score`), fusion (`fuse`), metrics, splits
- `tests/`: leakage rules as tests
- `data/raw/`, `cache/scores/`: not in git; regenerable
- `frozen_config.json`: written and tagged `freeze-v1` before any fraud-data result; never edited afterwards
