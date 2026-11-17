"""LaTeX tables for the LOF diagnosis and the stress test, rebuilt from the committed CSVs.

    python -m thesis.export_stress

Writes, next to each source CSV (nothing is recomputed; values are only rounded):
- results/tables/e3_ulb/lof/l2_controls.tex          (L2a-L2d, r per member)
- results/tables/ieee_cis/features_month1.tex        (the 26 freeze-v2 features)
- results/tables/ieee_cis/stress_month1.tex          (IEEE-CIS month-1 stress test)
- results/tables/e3_ulb/stress/stress_summary.tex    (ULB development: guard and rules)
"""
from __future__ import annotations

import sys

import pandas as pd

from thesis.export_datasets import git_state
from thesis.export_setup import booktabs
from thesis.paths import TABLES

LOF = TABLES / "e3_ulb" / "lof"
IEEE = TABLES / "ieee_cis"
STRESS = TABLES / "e3_ulb" / "stress"
RUN_LABELS = {"L2a": "L2a", "L2b": "L2b", "L2c": "L2c", "L2d": "L2d (E3)"}
PRIMARY_BUDGET = 0.002
STATIC = {"{IForest+PCA+KNN}": "E-bench", "{IForest+KNN+LOF}": "E-bench-5%",
          "{IForest+PCA+ECOD+HBOS+KNN+LOF}": "E-all"}


def one_commit(df: pd.DataFrame) -> str:
    commits = df["commit"].unique()
    if len(commits) != 1:
        raise SystemExit(f"several commits in one table: {commits}")
    return str(commits[0])[:7]


def l2_table(df: pd.DataFrame) -> pd.DataFrame:
    """Member x run: 'r at q 0.998 (r at q 0.995)', plus the k-distance ratio row."""
    out = {"Member": list(dict.fromkeys(df.member))}
    for run, label in RUN_LABELS.items():
        cells = []
        for m in out["Member"]:
            g = df[(df.run == run) & (df.member == m)].set_index("q").r
            cells.append(f"{g.loc[0.998]:.2f} ({g.loc[0.995]:.2f})")
        out[label] = cells
    t = pd.DataFrame(out)
    kd = df.groupby("run", sort=False).k_distance_ratio.first()
    t.loc[len(t)] = ["20th-NN distance ratio"] + [f"{kd.loc[r]:.2f}" for r in RUN_LABELS]
    return t


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    k = df[df.kept]
    return pd.DataFrame({"Feature": k.column, "Type": k.type, "Transform": k.transform,
                         "Missing in month 1 (%)": (100 * k.missing_share_month1).map(
                             lambda v: f"{v:.2f}")})


def ieee_stress_table(df: pd.DataFrame) -> pd.DataFrame:
    main = df[df.run == df.run.iloc[0]]
    out = {"Member": list(dict.fromkeys(main.member))}
    for q in (0.99, 0.98):
        g = main[main.q == q].set_index("member")
        out[f"r, q = {q:g}"] = [f"{g.loc[m, 'r']:.3f}" for m in out["Member"]]
        out[f"r / median, q = {q:g}"] = [f"{g.loc[m, 'r_over_median']:.2f}" for m in out["Member"]]
    return pd.DataFrame(out)


def compact_rules(rules: pd.DataFrame) -> str:
    """'all 12' or e.g. '1-sided rel. c = 2, 3; 2-sided rel. c = 3'."""
    if len(rules) == 12:
        return "all 12"
    parts = []
    for (side, basis), g in rules.groupby(["side", "basis"], sort=False):
        cs = ", ".join(str(c) for c in sorted(g.c))
        parts.append(f"{side[:3].replace('one', '1').replace('two', '2')}-sided "
                     f"{'abs.' if basis == 'absolute' else 'rel.'} c = {cs}")
    return "; ".join(parts)


def ulb_summary_table(dec: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    d = deltas[deltas.budget == PRIMARY_BUDGET].set_index(["variant", "static"])
    ch = dec[dec.changed].copy()
    ch["variant"] = "{" + ch.kept + "}"
    rows = []
    for (mech, ens, variant), g in ch.groupby(["mechanism", "ensemble", "variant"], sort=False):
        static = next(k for k, v in STATIC.items() if v == ens)
        r = d.loc[(variant, static)]
        rows.append({"Mechanism": mech, "Rules": compact_rules(g), "Ensemble": ens,
                     "Kept": variant.strip("{}").replace("+", ", "),
                     "dFP@K at 0.2% [95% CI]": f"{r.d_fp:+.1f} [{r.d_fp_ci_low:+.1f}, "
                                               f"{r.d_fp_ci_high:+.1f}]"})
    order = {"guard": 0, "pre-deployment": 1}
    return pd.DataFrame(rows).sort_values("Mechanism", key=lambda s: s.map(order), kind="stable")


def main(argv: list[str]) -> None:
    if argv:
        raise SystemExit("usage: python -m thesis.export_stress")
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before writing results.")
    me = state["commit"][:7]

    l2 = pd.read_csv(LOF / "l2_controls.csv")
    (LOF / "l2_controls.tex").write_text(booktabs(
        l2_table(l2),
        "Within-day and swapped controls on ULB: r = alert rate on the scored slice divided by "
        "its target, at the fit-slice threshold for q = 0.998 (q = 0.995 in brackets). L2a: "
        "random halves of day 1; L2b: day 1, 0--12 h to 12--24 h; L2c: day 2 to day 1; L2d: "
        "day 1 to day 2, as in E3. IForest: mean of five seeds. Label-free; development data.",
        "tab:lof-l2", f"{me} (data {one_commit(l2)})", "results/tables/e3_ulb/lof/l2_controls.csv",
        align="lrrrr", colsep="4pt", size="\\footnotesize"), encoding="utf-8")

    feats = pd.read_csv(IEEE / "features_month1.csv")
    (IEEE / "features_month1.tex").write_text(booktabs(
        feature_table(feats),
        "IEEE-CIS features fixed from month 1 without labels (freeze-v2): the 26 columns with a "
        "month-1 missing share below 20\\%. Medians and frequency maps are fitted on the fit "
        "window of each run.",
        "tab:ieee-features", f"{me} (data {one_commit(feats)})",
        "results/tables/ieee_cis/features_month1.csv", align="lllr", size="\\footnotesize"),
        encoding="utf-8")

    st = pd.read_csv(IEEE / "stress_month1.csv")
    (IEEE / "stress_month1.tex").write_text(booktabs(
        ieee_stress_table(st),
        "IEEE-CIS month 1, label-free stress test: fit on days 0--20, score days 21--29. r = "
        "alert rate / (1 - q) at the fit-window threshold, absolute and relative to the median "
        "member. IForest: mean of five seeds.",
        "tab:ieee-stress-month1", f"{me} (data {one_commit(st)})",
        "results/tables/ieee_cis/stress_month1.csv", align="lrrrr"), encoding="utf-8")

    dec = pd.read_csv(STRESS / "stress_decisions.csv")
    dl = pd.read_csv(STRESS / "stress_deltas.csv")
    (STRESS / "stress_summary.tex").write_text(booktabs(
        ulb_summary_table(dec, dl),
        "ULB development runs for RQ4: member sets that a rule changes, and the change in FP@K "
        "on day 2 at the 0.2\\% budget against the static ensemble (paired bootstrap, 1000 "
        "resamples). Guard: r on day 2 at the day-1 threshold; pre-deployment: r from the half-day "
        "split of day 1; q = 0.998. Development data; no rule was chosen with these numbers.",
        "tab:ulb-stress", f"{me} (data {one_commit(dec)})",
        "results/tables/e3_ulb/stress/stress_decisions.csv, stress_deltas.csv",
        align="l>{\\raggedright\\arraybackslash}p{4.3cm}l>{\\raggedright\\arraybackslash}p{3.3cm}r",
        colsep="3pt", size="\\footnotesize"), encoding="utf-8")
    print("wrote 4 tables")


if __name__ == "__main__":
    main(sys.argv[1:])
