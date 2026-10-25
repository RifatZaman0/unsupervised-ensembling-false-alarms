"""Export the dataset table (TDK §3.4.1) to results/tables/datasets/.

    python -m thesis.export_datasets

Descriptive statistics only (rows, features, label counts per split); no detector is run.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from thesis import data
from thesis.paths import REPO, TABLES

OUT = TABLES / "datasets"


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before exporting results.")
    OUT.mkdir(parents=True, exist_ok=True)

    cat = data.adbench_catalogue()
    cat["selected"] = cat.dataset.isin(data.select_adbench(cat))
    cat.to_csv(OUT / "adbench_catalogue.csv", index=False)

    rows = []
    for name in sorted(data.ADBENCH_15, key=str.lower):
        X, y = data.load_adbench(name)
        s = data.split_adbench(name, data.ADBENCH_SEEDS[0], X, y)
        rows.append({"dataset": name, "source": "ADBench", "split": "stratified 60/40, 5 seeds",
                     "n_rows": len(y), "n_features": X.shape[1], "n_anomalies": int(y.sum()),
                     "anomaly_rate": y.mean(), "n_train": len(s.y_train), "n_test": len(s.y_test),
                     "anomalies_train": int(s.y_train.sum()), "anomalies_test": int(s.y_test.sum())})
    X, y, t = data.load_ulb()
    s = data.split_ulb(X, y, t)
    rows.append({"dataset": "ULB", "source": "Kaggle mlg-ulb/creditcardfraud",
                 "split": "day 1 (Time < 86400 s) / day 2", "n_rows": len(y),
                 "n_features": X.shape[1], "n_anomalies": int(y.sum()), "anomaly_rate": y.mean(),
                 "n_train": len(s.y_train), "n_test": len(s.y_test),
                 "anomalies_train": int(s.y_train.sum()), "anomalies_test": int(s.y_test.sum())})
    pd.DataFrame(rows).to_csv(OUT / "datasets.csv", index=False)

    (OUT / "datasets.meta.json").write_text(json.dumps({
        **state, "command": "python -m thesis.export_datasets",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": ["adbench_catalogue.csv", "datasets.csv"],
        "note": "n_train/n_test for ADBench are for seed 0; all seeds have the same sizes",
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
