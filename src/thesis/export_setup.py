"""Experimental Setup tables (TDK §3.4) to results/tables/setup/, as CSV and booktabs LaTeX.

    python -m thesis.export_setup        (after python -m thesis.zero_variance)

- datasets:        15 ADBench sets + ULB: rows, features, anomaly rate, split
- hyperparameters: PyOD 3.6.7 defaults as used, and every deviation with its reason
- versions:        Python and the libraries the numbers depend on
- runtimes:        fit + score seconds per detector (ADBench: from the score cache; ULB: day-1
                   fit, training scores only, from zero_variance.csv) and the machine they ran on
No detector is fitted here.
"""
from __future__ import annotations

import ctypes
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version

import pandas as pd
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA

from thesis import config, data, detectors, scorecache
from thesis.export_datasets import dataset_rows, git_state
from thesis.paths import TABLES

OUT = TABLES / "setup"
PACKAGES = ("pyod", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib")
PYOD_CLASSES = {"IForest": IForest, "PCA": PCA, "HBOS": HBOS, "KNN": KNN, "LOF": LOF}

# Everything that is not a plain PyOD default, with the reason (README § Detectors).
DEVIATIONS = [
    ("all", "input scaling", "z-score, training statistics", "none",
     "KNN and LOF are distance-based and need comparable feature scales; fitted on the "
     "training split only (the other four are invariant to per-feature affine scaling)"),
    ("IForest", "random_state", "seeds 0-4", "None",
     "IForest is the only randomised detector; 5 seeds (ADBench: seed = split seed)"),
    ("PCA", "min_variance_ratio", "1e-10", "(not in PyOD)",
     "PyOD's weighted PCA divides by each component's explained-variance ratio, so a "
     "zero-variance direction gives infinite scores (e.g. cardio); such components are dropped"),
    ("ECOD", "implementation", "train-only ECDFs", "PyOD ECOD",
     "PyOD 3.6.7 ECOD builds the ECDFs from the training rows plus the rows being scored, so a "
     "score depends on the batch; here the ECDFs are fixed at fit time"),
    ("KNN, LOF", "training scores", "leave-one-out", "same",
     "PyOD's training scores exclude the row itself, as for a new row; stated because "
     "calibrators and thresholds are fitted on them"),
    ("KNN, LOF", "fit sample", f"{config.KNN_LOF_MAX_FIT_ROWS:,} training rows",
     "all rows",
     f"fit cost on fraud data; seed = split seed, or {config.KNN_LOF_SAMPLE_SEED} without one "
     "(ULB); rows outside the sample are scored like new rows; never applies to ADBench "
     "(largest training split 29,720 rows)"),
]


def booktabs(df: pd.DataFrame, caption: str, label: str, commit: str, source: str,
             align: str | None = None) -> str:
    def esc(v) -> str:
        s = str(v)
        for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                     ("#", "\\#")):
            s = s.replace(a, b)
        return s
    align = align or "l" * len(df.columns)
    lines = [f"% commit {commit}; source {source}", "\\begin{table}[t]", "\\centering",
             "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(esc(v) for v in r) + " \\\\" for r in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def datasets_table() -> pd.DataFrame:
    d = pd.DataFrame(dataset_rows())
    return pd.DataFrame({
        "dataset": d.dataset, "source": d.source, "n": d.n_rows, "d": d.n_features,
        "anomalies": d.n_anomalies, "anomaly_rate_pct": (100 * d.anomaly_rate).round(2),
        "split": d.split, "n_train": d.n_train, "n_test": d.n_test})


def hyperparameter_table() -> pd.DataFrame:
    rows = []
    for det in detectors.DETECTORS:
        cls = PYOD_CLASSES.get(det)
        defaults = ({k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()}
                    if cls else {})
        for k, v in detectors.PARAMS[det].items():
            if k in defaults:
                if defaults[k] != v:
                    raise SystemExit(f"{det}.{k} = {v!r} is not the PyOD default "
                                     f"{defaults[k]!r}; add it to DEVIATIONS")
                rows.append((det, k, v, defaults[k], "PyOD default", ""))
    rows += [(det, k, v, d, "stated (PyOD behaviour)" if d == "same" else "deviation", why)
             for det, k, v, d, why in DEVIATIONS]
    return pd.DataFrame(rows, columns=["detector", "setting", "value", "pyod_default", "kind",
                                       "reason"]).astype(str)


def versions_table() -> pd.DataFrame:
    rows = [("Python", platform.python_version())] + [(p, version(p)) for p in PACKAGES]
    return pd.DataFrame(rows, columns=["package", "version"])


def hardware() -> dict:
    cpu = platform.processor()
    ram_gb = None
    if sys.platform == "win32":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()

        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MemStatus()
        m.dwLength = ctypes.sizeof(MemStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.ullTotalPhys / 2**30, 1)
    release = platform.release()
    if platform.system() == "Windows" and int(platform.version().split(".")[-1]) >= 22000:
        release = "11"                      # platform.release() reports Windows 11 as "10"
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram_gb,
            "os": f"{platform.system()} {release} ({platform.version()})",
            "n_jobs": 1}


def runtime_table(zv: pd.DataFrame) -> pd.DataFrame:
    ad = zv[zv["data"] == "ADBench"]
    ulb = zv[zv["data"] == "ULB day 1"]
    rows = []
    for det in detectors.DETECTORS:
        a, u = ad[ad.detector == det], ulb[ulb.detector == det]
        rows.append({"detector": det,
                     "adbench_median_s": round(a.runtime_s.median(), 3),
                     "adbench_max_s": round(a.runtime_s.max(), 2),
                     "adbench_max_dataset": a.loc[a.runtime_s.idxmax(), "dataset"],
                     "adbench_total_s": round(a.runtime_s.sum(), 1),
                     "ulb_day1_fit_s": round(u.runtime_s.mean(), 1),
                     "ulb_runs": len(u)})
    return pd.DataFrame(rows)


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before exporting results.")
    zv_path = OUT / "zero_variance.csv"
    if not zv_path.exists():
        raise SystemExit("run python -m thesis.zero_variance first (ULB fit times)")
    zv_meta = json.loads((OUT / "zero_variance.meta.json").read_text(encoding="utf-8"))
    zv = pd.read_csv(zv_path)
    commit = state["commit"][:7]
    hw = hardware()

    ds = datasets_table()
    hp = hyperparameter_table()
    ver = versions_table()
    rt = runtime_table(zv)
    n_ad = int(len(ds) - 1)
    tex = {
        "datasets": booktabs(
            ds.drop(columns=["source", "anomalies"]).rename(columns={
                "anomaly_rate_pct": "anomalies (%)", "n_train": "train", "n_test": "test"}),
            f"Datasets: {n_ad} ADBench sets (files as distributed, duplicates kept) and ULB "
            "credit-card fraud. ADBench: stratified 60/40 split, 5 seeds (sizes for seed 0, "
            "equal for all seeds); ULB: day 1 for fitting, day 2 for testing.",
            "tab:setup-datasets", commit, "datasets (thesis.export_setup)", "lrrrlrr"),
        "hyperparameters": booktabs(
            hp.drop(columns=["pyod_default"]),
            "Detector settings: PyOD 3.6.7 defaults as used, and every deviation with its "
            "reason. contamination is not listed: no detector threshold is used.",
            "tab:setup-hyperparameters", commit, "hyperparameters (thesis.export_setup)",
            "lllp{0.45\\linewidth}"),
        "versions": booktabs(ver, "Software versions.", "tab:setup-versions", commit,
                             "versions (thesis.export_setup)"),
        "runtimes": booktabs(
            rt.rename(columns={"adbench_median_s": "ADBench median (s)",
                               "adbench_max_s": "ADBench max (s)",
                               "adbench_max_dataset": "max on",
                               "adbench_total_s": "ADBench total (s)",
                               "ulb_day1_fit_s": "ULB day 1 (s)"}).drop(columns="ulb_runs"),
            f"Detector runtimes in seconds, single thread, on {hw['cpu']} "
            f"({hw['logical_cores']} logical cores, {hw['ram_gb']} GB RAM). ADBench: fit + "
            f"scoring of one split ({n_ad} sets $\\times$ 5 seeds), recorded when the score "
            "cache was filled. ULB: fit on day 1 and training scores only (IForest: mean of "
            "5 seeds); KNN and LOF on the 30,000-row sample.",
            "tab:setup-runtimes", commit, "runtimes (thesis.export_setup)", "lrrlrr"),
    }
    frames = {"datasets": ds, "hyperparameters": hp, "versions": ver, "runtimes": rt,
              "hardware": pd.DataFrame([hw])}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    for name, t in tex.items():
        (OUT / f"{name}.tex").write_text(t, encoding="utf-8")
    (OUT / "setup.meta.json").write_text(json.dumps({
        **state, "command": "python -m thesis.export_setup",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cache_key": scorecache.cache_key(), "runtimes_from": "zero_variance.csv",
        "zero_variance_commit": zv_meta["commit"],
        "files": sorted([f"{n}.csv" for n in frames] + [f"{n}.tex" for n in tex]),
        "note": "ADBench runtimes come from the score cache (one fit + score per detector, "
                "dataset and seed, n_jobs 1); ULB runtimes are day-1 fits with training scores "
                "only, from thesis.zero_variance. ULB day 2 is not scored.",
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
