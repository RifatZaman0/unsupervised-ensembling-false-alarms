"""Experimental Setup tables (TDK §3.4) to results/tables/setup/, as CSV and booktabs LaTeX.

    python -m thesis.export_setup        (after python -m thesis.zero_variance)

- datasets:        15 ADBench sets + ULB: rows, features, anomaly rate, split
- hyperparameters: PyOD 3.6.7 defaults as used, and every deviation with its reason
- versions:        Python and the libraries the numbers depend on
- runtimes:        fit + score seconds per detector (ADBench: from the score cache; ULB: day-1
                   fit, training scores only, from zero_variance.csv) and the machine they ran on
- duplicates:      LaTeX only, from results/tables/datasets/duplicates.csv (supervisor's B01b)
No detector is fitted here. Every .tex fits the 16 cm TDK text block (tests/test_tdk_exports.py).
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
             align: str | None = None, colsep: str | None = None,
             size: str = "\\small") -> str:
    """booktabs table; `colsep` narrows \\tabcolsep (LaTeX default 6pt) so that a wide table
    fits the 16 cm text block of the TDK paper without changing any cell; `size` sets a long
    table in a smaller font so that it fits the page height."""
    def esc(v) -> str:
        s = str(v)
        for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"),
                     ("#", "\\#")):
            s = s.replace(a, b)
        return s
    align = align or "l" * len(df.columns)
    lines = [f"% commit {commit}; source {source}", "\\begin{table}[t]", "\\centering",
             size] + ([f"\\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []) + [
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
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


def duplicates_table(dup: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The supervisor's duplicate table (B01b) from results/tables/datasets/duplicates.csv:
    one row per set with exact (features, label) duplicates, and the names of the sets
    without any. Nothing is recounted here."""
    has = dup[dup.rows_removed > 0]
    none = list(dup.loc[dup.rows_removed == 0, "dataset"])
    table = pd.DataFrame({
        "dataset": has.dataset,
        "rows": has.n_rows.map(lambda v: f"{v:,}"),
        "in groups": [f"{n:,} ({100 * s:.1f}%)" for n, s in
                      zip(has.rows_in_dup_groups, has.share_rows_in_dup_groups)],
        "removed": has.rows_removed.map(lambda v: f"{v:,}"),
        "of them anomalies": has.anomalies_removed.astype(int),
        "rate (%)": (100 * has.anomaly_rate).map(lambda v: f"{v:.2f}"),
        "rate after (%)": (100 * has.anomaly_rate_dedup).map(lambda v: f"{v:.2f}")})
    return table.reset_index(drop=True), none


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


def tex_tables(ds: pd.DataFrame, hp: pd.DataFrame, ver: pd.DataFrame, rt: pd.DataFrame,
               hw: dict, dup: pd.DataFrame, dup_none: list[str], dup_meta: dict,
               commit: str) -> dict[str, str]:
    """The LaTeX tables of the Experimental Setup, from the frames written as CSV."""
    n_ad = int(len(ds) - 1)
    return {
        "datasets": booktabs(
            ds.drop(columns=["source", "anomalies"]).rename(columns={
                "anomaly_rate_pct": "anomalies (%)", "n_train": "train", "n_test": "test"}),
            f"Datasets: {n_ad} ADBench sets (files as distributed, duplicates kept) and ULB "
            "credit-card fraud. ADBench: stratified 60/40 split, 5 seeds (sizes for seed 0, "
            "equal for all seeds); ULB: day 1 for fitting, day 2 for testing.",
            "tab:setup-datasets", commit, "datasets (thesis.export_setup)", "lrrrlrr",
            colsep="5pt"),
        "hyperparameters": booktabs(
            hp.drop(columns=["pyod_default"]),
            "Detector settings: PyOD 3.6.7 defaults as used, and every deviation with its "
            "reason. contamination is not listed: no detector threshold is used.",
            "tab:setup-hyperparameters", commit, "hyperparameters (thesis.export_setup)",
            "llp{0.17\\linewidth}p{0.12\\linewidth}p{0.32\\linewidth}", colsep="4pt",
            size="\\footnotesize"),             # 16 rows: \small is 43.6 pt taller than the page
        "versions": booktabs(ver, "Software versions.", "tab:setup-versions", commit,
                             "versions (thesis.export_setup)"),
        "runtimes": booktabs(
            rt.rename(columns={"adbench_median_s": "ADBench median (s)",
                               "adbench_max_s": "ADBench max (s)",
                               "adbench_max_dataset": "max on",
                               "adbench_total_s": "ADBench total (s)",
                               "ulb_day1_fit_s": "ULB day 1 (s)"}).drop(columns="ulb_runs"),
            f"Detector runtimes in seconds, n\\_jobs = 1, on {hw['cpu']} "
            f"({hw['logical_cores']} logical cores, {hw['ram_gb']} GB RAM). ADBench: fit + "
            f"scoring of one split ({n_ad} sets $\\times$ 5 seeds), recorded when the score "
            "cache was filled. ULB: fit on day 1 and training scores only (IForest: mean of "
            "5 seeds); KNN and LOF on the 30,000-row sample.",
            "tab:setup-runtimes", commit, "runtimes (thesis.export_setup)", "lrrlrr",
            colsep="5pt"),
        "duplicates": booktabs(
            dup,
            f"Exact duplicates (features and label) in the distributed ADBench files, for the "
            f"{len(dup)} sets that contain any; the other {len(dup_none)} "
            f"({', '.join(dup_none)}) contain none. Removed: what the duplicate robustness "
            "check drops (the first occurrence of every (features, label) row is kept; rows "
            "with equal features but different labels are all kept). In groups: rows whose "
            "(features, label) occurs more than once. Rate: anomaly rate as distributed and "
            "after removal.",
            "tab:setup-duplicates", commit,
            f"duplicates.csv in tables/datasets @ {dup_meta['commit'][:7]} "
            "(thesis.export_setup)", "lrrrrrr", colsep="5pt"),
    }


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before exporting results.")
    zv_path = OUT / "zero_variance.csv"
    if not zv_path.exists():
        raise SystemExit("run python -m thesis.zero_variance first (ULB fit times)")
    zv_meta = json.loads((OUT / "zero_variance.meta.json").read_text(encoding="utf-8"))
    zv = pd.read_csv(zv_path)
    dup_dir = TABLES / "datasets"
    dup_meta = json.loads((dup_dir / "datasets.meta.json").read_text(encoding="utf-8"))
    dup, dup_none = duplicates_table(pd.read_csv(dup_dir / "duplicates.csv"))
    commit = state["commit"][:7]
    hw = hardware()

    ds = datasets_table()
    hp = hyperparameter_table()
    ver = versions_table()
    rt = runtime_table(zv)
    tex = tex_tables(ds, hp, ver, rt, hw, dup, dup_none, dup_meta, commit)
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
        "duplicates_from": "../datasets/duplicates.csv",
        "duplicates_commit": dup_meta["commit"],
        "files": sorted([f"{n}.csv" for n in frames] + [f"{n}.tex" for n in tex]),
        "note": "ADBench runtimes come from the score cache (one fit + score per detector, "
                "dataset and seed, n_jobs 1); ULB runtimes are day-1 fits with training scores "
                "only, from thesis.zero_variance. ULB day 2 is not scored.",
    }, indent=1) + "\n", encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
