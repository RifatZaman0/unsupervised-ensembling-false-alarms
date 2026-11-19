"""Chapter 5 exports (thesis.export_ch5): the tables and figures show the committed values
unchanged, the sentences that state a direction refuse data that contradict them, every table
compiles and fits the text block, figures are 16 cm wide with no title and no text below 8 pt,
and the exporter never writes into the IEEE-CIS run folder.

Reads the committed results; each test skips with its reason when an input is missing.
"""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from matplotlib.text import Text

from thesis import export_ch5 as x
from thesis import texcheck


def need(*paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("missing " + ", ".join(missing))


@pytest.fixture(scope="module")
def run():
    need(x.RUN / "run.meta.json", *(x.RUN / f"{f}.csv" for f in x.RUN_FILES))
    return x.load_run()


def copy(t):
    return {k: v.copy() for k, v in t.items()}


# ------------------------------------------------------------------ C23 values and sentences

def test_hypothesis_table_shows_the_committed_values(run):
    t, meta = run
    tab = x.hypothesis_table(t, meta).set_index("Hypothesis")
    h = t["rq4_hypotheses"].set_index("hypothesis")
    v1 = t["v1_hypotheses"].set_index("hypothesis")
    assert tab.loc["H4r", "Estimate [95% CI]"] == x.est_ci(*h.loc["H4r", ["estimate", "ci_low", "ci_high"]], 4)
    assert tab.loc["H4a", "Estimate [95% CI]"] == x.est_ci(*h.loc["H4a", ["estimate", "ci_low", "ci_high"]], 3)
    assert tab.loc["H1b", "Estimate [95% CI]"] == x.est_ci(*v1.loc["H1b", ["d_fp", "d_fp_ci_low", "d_fp_ci_high"]], 1, True)
    for hyp in ("H4r", "H4r-cost", "H4a", "H4b", "H4c"):
        assert tab.loc[hyp, "Verdict"] == h.loc[hyp, "verdict"]
    for hyp in ("H1a", "H1b"):
        assert tab.loc[hyp, "Verdict"] == v1.loc[hyp, "status"]
    assert tab.loc["H3a", "Verdict"] == t["h3a"]["verdict"].iloc[0]
    assert x.MINUS in tab.loc["H4r", "Estimate [95% CI]"] and "-" not in tab.loc["H4r", "Estimate [95% CI]"]
    assert set(tab.index) >= {"H4r", "H4r-cost", "H4a", "H4b", "H4c", "H1a", "H1b", "H2b (secondary)",
                              "H3a", "H3b", "H4d"}


def test_h1b_and_h4a_sentences(run):
    t, meta = run
    tab = x.hypothesis_table(t, meta).set_index("Hypothesis")
    assert tab.loc["H1b", "In plain units"].startswith("The ensemble is worse than default IForest")
    assert "CI entirely above 0" in tab.loc["H1b", "In plain units"]
    note = tab.loc["H4a", "In plain units"]
    assert "KNN and LOF, excluded by the guard on all 3 event days, had the smallest check deviation" in note
    assert "ECOD and HBOS, excluded on fewer days, were below target" in note
    assert "a result, not 0 by identity" in tab.loc["H4c", "In plain units"]


def test_h1b_sentence_refuses_a_ci_that_is_not_above_0(run):
    t, meta = run
    t2 = copy(t)
    d = t2["e3_deltas"]
    d.loc[(d["budget"] == x.PRIMARY) & (d["role"] == "H1b"), "d_fp_ci_low"] = -1.0
    with pytest.raises(ValueError, match="H1b"):
        x.hypothesis_table(t2, meta)


def test_h4a_note_refuses_when_the_broken_members_are_not_the_closest(run):
    t, meta = run
    t2 = copy(t)
    p = t2["rq4_pre_deployment_r"]
    p.loc[(p["budget"] == x.PRIMARY) & (p["member"] == "KNN"), "r"] = 0.05
    with pytest.raises(ValueError, match="H4a note"):
        x.hypothesis_table(t2, meta)


def test_h4c_sentence_refuses_changed_fp(run):
    t, meta = run
    t2 = copy(t)
    d = t2["rq4_daily_fp"]
    day = x.guard_facts(t)["changed"]["E-bench"][0]
    d.loc[(d["budget"] == x.PRIMARY) & (d["method"] == "guarded:E-bench") & (d["day"] == day), "fp_at_k"] += 1
    with pytest.raises(ValueError, match="H4c"):
        x.hypothesis_table(t2, meta)


# ------------------------------------------------------------------ C24, C26, C28

def test_h4r_tables(run):
    t, _ = run
    main = x.h4r_main_table(t)
    tri = t["rq4_h4r_triples"]
    assert main["D robust"].iloc[0] == f"{tri['D_robust'].mean():.3f}"
    assert list(main["Ensemble"].iloc[1:]) == ["E-bench", "E-bench-5%", "E-all"]
    full = x.h4r_triples_table(t)
    assert len(full) == 20
    assert list(full["dD"]) == [x.num(v, 3, True) for v in tri["dD"]]
    assert full["Ensemble"].str.contains(r"\(E-bench\)").sum() == 1


def test_guard_table_marks_exactly_the_excluded_member_days(run):
    t, _ = run
    g = x.guard_table(t)
    days = x.guard_facts(t)["event_days"]
    assert list(g.columns[1:]) == [f"Day {d}" for d in days]
    md = t["rq4_member_days"]
    md = md[md["budget"] == x.PRIMARY].set_index(["member", "day"])
    for m in x.MEMBERS:
        row = g[g[""] == f"r, {m}"].iloc[0]
        for d in days:
            assert row[f"Day {d}"].endswith("†") == (not md.loc[(m, d), "passes"])


def test_bound_table_refuses_a_broken_inequality(run):
    t, _ = run
    assert len(x.bound_table(t)) == len(t["rq4_bound_check"]) // 5
    t2 = copy(t)
    t2["rq4_bound_check"].loc[0, "bound"] = 0.0
    with pytest.raises(ValueError, match="bound check"):
        x.bound_table(t2)


def test_load_run_refuses_a_dry_run_or_mixed_commits(run, tmp_path):
    for f in [*x.RUN_FILES, "run.meta.json"]:
        name = f if f.endswith(".json") else f"{f}.csv"
        shutil.copy(x.RUN / name, tmp_path / name)
    meta = json.loads((tmp_path / "run.meta.json").read_text(encoding="utf-8"))
    (tmp_path / "run.meta.json").write_text(json.dumps({**meta, "dry_run": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="real, clean"):
        x.load_run(tmp_path)
    (tmp_path / "run.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    h = pd.read_csv(tmp_path / "h3a.csv").assign(commit="0" * 40)
    h.to_csv(tmp_path / "h3a.csv", index=False)
    with pytest.raises(ValueError, match="several commits"):
        x.load_run(tmp_path)


# ------------------------------------------------------------------ LaTeX

def test_every_ch5_table_compiles_and_fits(run):
    if texcheck.xelatex() is None:
        pytest.skip("xelatex not found")
    need(texcheck.TDK_MAIN, x.RF / "adbench_static.csv")
    t, meta = run
    tex = x.tex_outputs(t, meta, "abc1234")
    tex["adbench_static.tex"] = x.adbench_tex(pd.read_csv(x.RF / "adbench_static.csv"), "abc1234")
    assert len(tex) == 10
    for name, src in tex.items():
        assert src.startswith("% commit abc1234 (data ")
        r = texcheck.compile_snippet(src)
        missing = [ln for ln in r.log.splitlines() if "Missing character" in ln]
        assert r.ok and not r.overfull and not r.too_tall and not missing, \
            f"{name}: ok={r.ok}, overfull={r.overfull}, too_tall={r.too_tall}, missing={missing[:2]}"


# ------------------------------------------------------------------ figures

def check_figure(fig):
    w, _ = fig.get_size_inches()
    assert w * 2.54 == pytest.approx(x.WIDTH_CM)
    assert fig._suptitle is None
    for ax in fig.axes:
        assert ax.get_title() == "" and ax.get_title("left") == "" and ax.get_title("right") == ""
    sizes = [tx.get_fontsize() for tx in fig.findobj(Text) if tx.get_text().strip()]
    assert sizes and min(sizes) >= 8


def offsets(ax):
    return np.vstack([c.get_offsets() for c in ax.collections if len(c.get_offsets())])


def test_tradeoff_figure_plots_the_table_values(run):
    t, _ = run
    fig = x.tradeoff_figure(t)
    check_figure(fig)
    ax = fig.axes[0]
    tri = t["rq4_h4r_triples"]
    pts = {tuple(np.round(p, 9)) for p in offsets(ax)}
    want = {(round(a, 9), round(100 * (r / f - 1), 9))
            for a, r, f in zip(tri["dD"], tri["fp_total_robust"], tri["fp_total_frozen"])}
    assert want <= pts
    assert any(abs(ln.get_ydata()[0] - 5) < 1e-12 and ln.get_linestyle() == "--" for ln in ax.get_lines())


def test_h4a_figure_plots_the_pairs_and_marks_the_groups(run):
    t, _ = run
    fig = x.h4a_figure(t)
    check_figure(fig)
    ax = fig.axes[0]
    p = t["rq4_h4a_pairs"]
    pts = sorted(map(tuple, np.round(offsets(ax), 9)))
    assert pts == sorted(zip(np.round(p["x_abs_log_r_pre"], 9), np.round(p["y_abs_log_r"], 9)))
    grp = x.h4a_groups(t)
    assert grp["KNN"] == grp["LOF"] == x.H4A_GROUPS[0][0]
    assert grp["ECOD"] == grp["HBOS"] == x.H4A_GROUPS[1][0]
    assert grp["IForest"] == grp["PCA"] == x.H4A_GROUPS[2][0]
    assert len({s["marker"] for _, s in x.H4A_GROUPS}) == 3


def test_lof_figure_plots_l2_controls(run):
    need(x.LOF / "l2_controls.csv")
    l2 = pd.read_csv(x.LOF / "l2_controls.csv")
    fig = x.lof_figure(l2)
    check_figure(fig)
    ys = np.sort(offsets(fig.axes[0])[:, 1])
    want = np.sort(l2[(l2["q"] == 0.998) & l2["run"].isin([r for r, _ in x.L2_RUNS])]["r"].to_numpy())
    assert np.allclose(ys, want, rtol=0, atol=1e-12)
    assert len(set(x.L2_MARKERS.values())) == len(x.MEMBERS)


# ------------------------------------------------------------------ main never writes into run/

def digest(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_main_writes_outside_the_run_folder_and_leaves_it_unchanged(run, tmp_path, monkeypatch):
    need(x.RF / "adbench_static.csv", x.LOF / "l2_controls.csv")
    proj = tmp_path / "results"
    run_dir = proj / "tables" / "ieee_cis" / "run"
    shutil.copytree(x.RUN, run_dir)
    (proj / "tables" / "robust_fusion").mkdir(parents=True)
    shutil.copy(x.RF / "adbench_static.csv", proj / "tables" / "robust_fusion" / "adbench_static.csv")
    (proj / "tables" / "e3_ulb" / "lof").mkdir(parents=True)
    shutil.copy(x.LOF / "l2_controls.csv", proj / "tables" / "e3_ulb" / "lof" / "l2_controls.csv")
    before = digest(run_dir)
    monkeypatch.setattr(x, "RUN", run_dir)
    monkeypatch.setattr(x, "OUT", proj / "tables" / "ieee_cis" / "ch5")
    monkeypatch.setattr(x, "FIG", proj / "figures" / "ieee_cis")
    monkeypatch.setattr(x, "RF", proj / "tables" / "robust_fusion")
    monkeypatch.setattr(x, "LOF", proj / "tables" / "e3_ulb" / "lof")
    monkeypatch.setattr(x, "LOF_FIG", proj / "figures" / "e3_ulb" / "lof")
    monkeypatch.setattr(x, "FIGURES", proj / "figures")
    monkeypatch.setattr(x, "git_state", lambda: {"commit": "f" * 40, "dirty": False})
    x.main([])
    assert digest(run_dir) == before
    meta = json.loads((x.OUT / "ch5.meta.json").read_text(encoding="utf-8"))
    assert len(meta["written"]) == 16 and not any("/run/" in w for w in meta["written"])
    assert (x.FIG / "h4r_tradeoff.pdf").exists() and (x.LOF_FIG / "l2_controls_r.pdf").exists()
    monkeypatch.setattr(x, "OUT", run_dir / "ch5")
    with pytest.raises(SystemExit, match="never write into the run folder"):
        x.main([])
    monkeypatch.setattr(x, "git_state", lambda: {"commit": "f" * 40, "dirty": True})
    with pytest.raises(SystemExit, match="uncommitted"):
        x.main([])
