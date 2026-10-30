"""TDK exports: every exported LaTeX table compiles in the TDK preamble and fits the text block,
the two ULB figures are drawn at the 16 cm text width with no text below 8 pt and plot the table
values unchanged, and the LOF diagnosis reads no label.

LaTeX tests need XeLaTeX and tdk/main.tex; table and figure tests read the committed CSVs in
results/. Each skips with its reason when an input is missing (check the skip count).
"""
import json
import re

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from thesis import data, detectors, e2_figures, e3_ulb, e3_ulb_figures, e3_ulb_lof, export_setup
from thesis import frozen, texcheck
from thesis.paths import TABLES

SETUP, DATASETS, E2, E3 = (TABLES / d for d in ("setup", "datasets", "e2", "e3_ulb"))


def need(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("missing " + ", ".join(missing))


def need_latex():
    if texcheck.xelatex() is None:
        pytest.skip("xelatex not found")
    need(texcheck.TDK_MAIN)


# ------------------------------------------------------------------ table builders from the CSVs

def setup_tex() -> dict[str, str]:
    need(SETUP / "datasets.csv", SETUP / "hyperparameters.csv", SETUP / "versions.csv",
         SETUP / "runtimes.csv", SETUP / "hardware.csv", DATASETS / "duplicates.csv")
    hp = pd.read_csv(SETUP / "hyperparameters.csv", keep_default_na=False).astype(str)
    dup, none = export_setup.duplicates_table(pd.read_csv(DATASETS / "duplicates.csv"))
    meta = json.loads((DATASETS / "datasets.meta.json").read_text(encoding="utf-8"))
    return export_setup.tex_tables(
        pd.read_csv(SETUP / "datasets.csv"), hp, pd.read_csv(SETUP / "versions.csv", dtype=str),
        pd.read_csv(SETUP / "runtimes.csv"), pd.read_csv(SETUP / "hardware.csv").iloc[0].to_dict(),
        dup, none, meta, "abc1234")


def results_tex() -> dict[str, str]:
    need(E3 / "e3_ulb_budget.csv", E3 / "e3_ulb_hypotheses.csv", E3 / "e3_ulb_threshold.csv",
         E2 / "e2_summary.csv", E2 / "e2_friedman.csv")
    tex = e3_ulb.tex_tables(pd.read_csv(E3 / "e3_ulb_budget.csv"),
                            pd.read_csv(E3 / "e3_ulb_hypotheses.csv"),
                            pd.read_csv(E3 / "e3_ulb_threshold.csv"), frozen.load(), "abc1234")
    tex["e2_triples.tex"] = e2_figures.latex_triples(pd.read_csv(E2 / "e2_summary.csv"),
                                                     pd.read_csv(E2 / "e2_friedman.csv"),
                                                     {"commit": "abc1234"})
    return tex


def n_columns(spec: str) -> int:
    """Columns in a tabular spec such as 'llp{0.17\\linewidth}r'."""
    spec = re.sub(r"\{[^{}]*\}", "", re.sub(r"@\{[^{}]*\}", "", spec))
    return sum(spec.count(c) for c in "lcrpmb")


def cells_match_columns(tex: str) -> bool:
    spec = re.search(r"\\begin\{tabular\}\{(.*)\}\n", tex).group(1)
    body = tex[tex.index("\\toprule"):tex.index("\\bottomrule")]
    rows = [r for r in body.split("\\\\\n") if "&" in r and "\\multicolumn" not in r]
    return all(r.count("&") + 1 == n_columns(spec) for r in rows)


# ------------------------------------------------------------------ LaTeX tables

def test_every_setup_table_has_as_many_cells_as_columns():
    for name, tex in setup_tex().items():
        assert cells_match_columns(tex), name


def test_column_counter_catches_the_old_hyperparameter_spec():
    tex = setup_tex()["hyperparameters"]
    old = re.sub(r"\\begin\{tabular\}\{.*\}\n", lambda m: "\\begin{tabular}{lllp{0.45\\linewidth}}\n",
                 tex)
    assert cells_match_columns(tex) and not cells_match_columns(old)


def test_duplicate_table_comes_from_the_csv_unchanged():
    need(DATASETS / "duplicates.csv")
    dup = pd.read_csv(DATASETS / "duplicates.csv")
    table, none = export_setup.duplicates_table(dup)
    has = dup[dup.rows_removed > 0]
    assert list(table.dataset) == list(has.dataset)
    assert set(none) | set(table.dataset) == set(dup.dataset) and not set(none) & set(table.dataset)
    assert [int(v.replace(",", "")) for v in table.removed] == list(has.rows_removed)
    assert list(table["of them anomalies"]) == list(has.anomalies_removed)
    assert [float(v) for v in table["rate (%)"]] == [round(100 * a, 2) for a in has.anomaly_rate]


@pytest.mark.parametrize("which", ["setup", "results"])
def test_exported_tables_compile_in_the_tdk_preamble_and_fit(which):
    need_latex()
    tex = setup_tex() if which == "setup" else results_tex()
    for name, body in tex.items():
        r = texcheck.compile_snippet(body)
        assert r.ok, f"{name} does not compile:\n{r.log[-2000:]}"
        assert not r.overfull, f"{name} is wider than the text block: {r.overfull} pt"


def test_every_tex_file_in_results_compiles_and_fits():
    need_latex()
    files = sorted(TABLES.rglob("*.tex"))
    if not files:
        pytest.skip("no .tex in results/tables")
    for f in files:
        r = texcheck.compile_snippet(f.read_text(encoding="utf-8"))
        assert r.ok and not r.overfull, f"{f.name}: ok={r.ok}, overfull={r.overfull}"


def test_overfull_check_can_fail():
    need_latex()
    r = texcheck.compile_snippet("\\begin{tabular}{l}" + "x" * 400 + "\\end{tabular}")
    assert r.ok and r.overfull


# ------------------------------------------------------------------ ULB figures

def figures():
    need(E3 / "e3_ulb_deltas.csv", E3 / "e3_ulb_threshold.csv")
    cfg = frozen.load()
    deltas = pd.read_csv(E3 / "e3_ulb_deltas.csv")
    thr = pd.read_csv(E3 / "e3_ulb_threshold.csv")
    return cfg, deltas, thr


def test_ulb_figures_are_text_width_with_no_text_below_8_pt(tmp_path):
    cfg, deltas, thr = figures()
    for fig in (e3_ulb_figures.delta_fig(deltas, cfg), e3_ulb_figures.threshold_fig(thr, cfg)):
        assert fig.get_size_inches()[0] * 2.54 == pytest.approx(16.0)
        fig.canvas.draw()
        sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
        assert sizes and min(sizes) >= 8
    e3_ulb_figures.write(deltas, thr, cfg, tmp_path, "abc1234")
    for stem in ("delta_primary_budget", "threshold_check"):
        pdf = (tmp_path / f"{stem}.pdf").read_bytes()
        box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", pdf)
        assert float(box.group(1)) / 72 * 2.54 == pytest.approx(16.0, abs=0.01)


def plotted_points(fig) -> list[tuple[float, float]]:
    return [(float(l.get_xdata()[0]), float(l.get_ydata()[0])) for ax in fig.axes
            for l in ax.get_lines() if l.get_marker() == "o"]


def test_ulb_figures_plot_the_table_values_unchanged():
    cfg, deltas, thr = figures()
    pb = cfg["budgets"]["ulb"]["primary"]
    d = deltas[np.isclose(deltas.budget, pb) & deltas.comparator.isin(list(e3_ulb_figures.COMPARATORS))]
    xs = sorted(x for x, _ in plotted_points(e3_ulb_figures.delta_fig(deltas, cfg)))
    assert xs == sorted(d.d_fp_per_100)
    calib, agg = e3_ulb.fusion_configs(cfg)[0]
    t = thr[(thr.kind != "ensemble") | ((thr.calib == calib) & (thr["agg"] == agg))]
    xs = sorted(x for x, _ in plotted_points(e3_ulb_figures.threshold_fig(thr, cfg)))
    assert xs == sorted(100 * t.alert_rate)


# ------------------------------------------------------------------ LOF diagnosis (no labels)

def fake_ulb(path, class_values, n1=60, n2=50, seed=0):
    rng = np.random.default_rng(seed)
    n = n1 + n2
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.r_[np.linspace(0, 86_000, n1), np.linspace(86_400, 170_000, n2)])
    df["Amount"] = rng.exponential(50, size=n)
    df["Class"] = class_values(n)
    df.to_csv(path, index=False)


def run_on(path, monkeypatch, max_rows=40):
    monkeypatch.setattr(data, "ULB_CSV", path)
    X, t = data.load_ulb_unlabelled()
    day1 = t < data.ULB_DAY_SECONDS
    return e3_ulb_lof.diagnose(X[day1], X[~day1], list(data.ULB_FEATURES), 5, max_rows)


def test_unlabelled_loader_never_reads_the_class_column(tmp_path, monkeypatch):
    path = tmp_path / "creditcard.csv"
    fake_ulb(path, lambda n: np.zeros(n, dtype=int))
    seen = []
    real = pd.read_csv

    def spy(*a, **kw):
        seen.append(kw.get("usecols"))
        return real(*a, **kw)
    monkeypatch.setattr(data.pd, "read_csv", spy)
    run_on(path, monkeypatch)
    assert seen and all(u is not None and "Class" not in u for u in seen)


def test_diagnosis_is_identical_whatever_the_labels(tmp_path, monkeypatch):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    fake_ulb(a, lambda n: np.zeros(n, dtype=int))
    fake_ulb(b, lambda n: np.arange(n) % 2)                  # same features, other labels
    pd.testing.assert_frame_equal(run_on(a, monkeypatch), run_on(b, monkeypatch))


def test_knn_distances_are_the_ones_lof_uses(monkeypatch):
    rng = np.random.default_rng(1)
    X1, X2 = rng.normal(size=(80, 4)), rng.normal(loc=0.5, size=(30, 4))
    monkeypatch.setattr(detectors, "MAX_FIT_ROWS", 50)       # force the fit sample
    det, _ = detectors.fit_scores("LOF", X1)
    k = e3_ulb_lof.lof_k()
    d1, d2 = e3_ulb_lof.knn_distances(X1, X2, k)
    sample = e3_ulb_lof.fit_sample(len(X1))
    np.testing.assert_allclose(d1[sample], det.detector_._distances_fit_X_)
    np.testing.assert_allclose(d2, det.detector_.kneighbors(X2)[0])


def test_lof_k_must_be_the_pyod_default(monkeypatch):
    assert e3_ulb_lof.lof_k() == 20
    monkeypatch.setitem(detectors.PARAMS, "LOF", {**detectors.PARAMS["LOF"], "n_neighbors": 7})
    with pytest.raises(SystemExit, match="not the PyOD default"):
        e3_ulb_lof.lof_k()


def test_feature_shift_is_in_day1_standard_deviations():
    rng = np.random.default_rng(2)
    X1 = rng.normal(size=(500, 3)) * [1.0, 2.0, 5.0]
    X2 = X1[:200] + 2 * X1.std(axis=0)                       # every feature shifted by 2 day-1 SD
    fs = e3_ulb_lof.feature_shift(X1, X2, ["a", "b", "c"])
    np.testing.assert_allclose(fs.shift_sd, (X1[:200].mean(0) - X1.mean(0)) / X1.std(0) + 2)
    # day-2 rows never enter the scaler: changing them leaves the day-1 statistics unchanged
    fs2 = e3_ulb_lof.feature_shift(X1, X2 * 10, ["a", "b", "c"])
    np.testing.assert_array_equal(fs.day1_mean, fs2.day1_mean)
    np.testing.assert_array_equal(fs.day1_sd, fs2.day1_sd)
