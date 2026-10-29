"""Setup tables: every parameter is a checked PyOD default or a listed deviation, LaTeX is
escaped, and the runtime table summarises per detector."""
import pandas as pd

from thesis import detectors, export_setup


def test_hyperparameters_are_defaults_or_listed_deviations():
    hp = export_setup.hyperparameter_table()
    defaults = hp[hp.kind == "PyOD default"]
    for det in ("IForest", "PCA", "HBOS", "KNN", "LOF"):
        for k in detectors.PARAMS[det]:
            if k != "min_variance_ratio":
                assert ((defaults.detector == det) & (defaults.setting == k)).sum() == 1
    dev = hp[hp.kind == "deviation"]
    assert {"ECOD", "PCA", "IForest", "KNN, LOF", "all"} <= set(dev.detector)
    assert (dev.reason.str.len() > 20).all()
    stated = hp[hp.kind.str.startswith("stated")]
    assert list(stated.setting) == ["training scores"]          # leave-one-out is PyOD's own


def test_booktabs_escapes_special_characters():
    df = pd.DataFrame({"a_b": ["50%", "x & y"], "n": [1, 2]})
    tex = export_setup.booktabs(df, "cap", "tab:x", "abc1234", "src")
    assert "a\\_b & n \\\\" in tex and "50\\% & 1 \\\\" in tex and "x \\& y & 2" in tex
    assert tex.startswith("% commit abc1234") and "\\toprule" in tex and "\\bottomrule" in tex


def test_runtime_table_one_row_per_detector():
    rows = [{"data": "ADBench", "dataset": ds, "seed": s, "detector": d, "runtime_s": 1.0 + s}
            for ds in ("a", "b") for s in range(5) for d in detectors.DETECTORS]
    rows += [{"data": "ULB day 1", "dataset": "ULB", "seed": None, "detector": d,
              "runtime_s": 10.0} for d in detectors.DETECTORS]
    rt = export_setup.runtime_table(pd.DataFrame(rows))
    assert list(rt.detector) == list(detectors.DETECTORS)
    assert (rt.adbench_total_s == 2 * sum(1.0 + s for s in range(5))).all()
    assert (rt.ulb_day1_fit_s == 10.0).all() and (rt.adbench_median_s == 3.0).all()


def test_versions_and_hardware():
    v = export_setup.versions_table()
    assert v.package.iloc[0] == "Python" and {"pyod", "numpy", "scipy"} <= set(v.package)
    hw = export_setup.hardware()
    assert hw["cpu"] and hw["logical_cores"] >= 1
