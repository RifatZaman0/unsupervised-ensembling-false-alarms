"""E-bench: the triple chosen on ADBench by a rule declared before any fraud-data result.

    python -m thesis.ebench      (after python -m thesis.e2)

Rule: among the 20 triples, the lowest average FP@K rank over the 15
ADBench sets under the primary fusion config (min-max + mean) at the primary 2% budget. FP@K
uses test labels, so this is a label-based choice on ADBench only; fraud data never enters it.
The same rule at 5% names the secondary ensemble E-bench-5%. Writes results/tables/e2/ebench.md
from e2_summary.csv and e2_friedman.csv.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from thesis import e2, e2_figures, fusion
from thesis.export_datasets import git_state
from thesis.paths import TABLES

OUT = TABLES / "e2"
PRIMARY_BUDGET, SECONDARY_BUDGET = 0.02, 0.05
SIZE = 3
TOP_N = 5
E_LIT = e2_figures.E_LIT


def triples(summ: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Triples at one budget under the primary config, by average rank (ties: by name)."""
    calib, agg = fusion.PRIMARY
    t = summ[(summ["size"] == SIZE) & (summ.calib == calib) & (summ["agg"] == agg)
             & (summ.budget.round(9) == round(budget, 9))].copy()
    if t.empty:
        raise ValueError(f"no triples at budget {budget}")
    t["avg_rank_fp_at_k"] = t.avg_rank_fp_at_k.round(e2_figures.TIE_DECIMALS)
    t = t.sort_values(["avg_rank_fp_at_k", "combo"]).reset_index(drop=True)
    t["position"] = t.avg_rank_fp_at_k.rank(method="min").astype(int)
    return t


def select(summ: pd.DataFrame, budget: float = PRIMARY_BUDGET) -> tuple[str, ...]:
    """Members of the triple with the lowest average FP@K rank; refuses an exact tie for first,
    which the rule does not resolve."""
    t = triples(summ, budget)
    if (t.position == 1).sum() > 1:
        raise ValueError(f"tie for the lowest average rank at {budget}: "
                         f"{list(t.combo[t.position == 1])}")
    return tuple(t.combo.iloc[0].split(e2.SEP))


def markdown(summ: pd.DataFrame, fried: pd.DataFrame, meta: dict) -> str:
    calib, agg = (fusion.LABELS[x] for x in fusion.PRIMARY)
    bench, bench5 = e2.combo_name(select(summ)), e2.combo_name(select(summ, SECONDARY_BUDGET))
    n = int(summ.n_datasets.max())
    lines = ["# E-bench: the triple chosen on ADBench", "",
             f"Commit `{meta['commit']}` · source `e2_summary.csv`, `e2_friedman.csv` "
             f"(E2 commit `{meta['e2_commit'][:7]}`) · created {meta['created_utc']}", "",
             "## Rule (declared before any fraud-data result)", "",
             f"E-bench is the triple with the **lowest average FP@K rank** among the "
             f"{len(triples(summ, PRIMARY_BUDGET))} triples of the 6-detector pool, over the {n} "
             f"ADBench sets (seed means), under the primary fusion config **{calib} calibration "
             f"+ {agg} fusion**, at the primary ADBench budget **{PRIMARY_BUDGET:.0%}**. FP@K "
             "needs test labels, so this choice uses labels on ADBench only; no fraud data "
             "enters it. The rule was fixed before any fraud-data result.", ""]
    for b in (PRIMARY_BUDGET, SECONDARY_BUDGET):
        t = triples(summ, b)
        f = fried[(fried.budget.round(9) == round(b, 9)) & (fried["size"] == SIZE)].iloc[0]
        lines += [f"## Top {TOP_N} triples at {b:.0%}", "",
                  "| Pos. | Triple | Avg. FP@K rank (of 20) | Mean FP@K | "
                  f"Beats best member (of {n}) | Beats median member (of {n}) |",
                  "|---|---|---|---|---|---|"]
        for r in t.head(TOP_N).itertuples():
            lines.append(f"| {r.position} | {r.combo} | {r.avg_rank_fp_at_k:.2f} | "
                         f"{r.mean_fp_at_k:.1f} | {r.n_beats_best} | {r.n_beats_median} |")
        lines += ["", f"Friedman test over the {f.n_combos} triples ({f.n_datasets} sets): "
                      f"χ² = {f.friedman_chi2:.2f}, p = {f.friedman_p:.3f}; Nemenyi CD = "
                      f"{f.nemenyi_cd:.2f} at α = {f.alpha}; {f.n_within_cd} of {f.n_combos} "
                      "triples lie within CD of the lowest average rank.", ""]
    p2 = fried[(fried.budget.round(9) == PRIMARY_BUDGET) & (fried["size"] == SIZE)].iloc[0]
    p5 = fried[(fried.budget.round(9) == SECONDARY_BUDGET) & (fried["size"] == SIZE)].iloc[0]
    lines += ["## Reading", "",
              f"- **The triples cannot be told apart on these {n} sets** (Friedman p = "
              f"{p2.friedman_p:.3f} at 2%, {p5.friedman_p:.3f} at 5%; every triple lies within "
              "the critical difference of the first). E-bench has the lowest average rank under "
              "the declared rule; it is not shown to be better than the other triples.",
              f"- **E-bench = {bench}." + (f" This is the same triple as E-lit ({E_LIT}), the "
              "published default.** The coincidence must be stated in the paper: the benchmark "
              "rule confirms the literature default. E-bench (= E-lit) is the one primary "
              "ensemble; E-all is a secondary, descriptive control."
              if bench == E_LIT else f"** E-lit is {E_LIT}."),
              f"- **E-bench-5% = {bench5}** (the same rule at the {SECONDARY_BUDGET:.0%} "
              "budget) is declared a **secondary** ensemble and frozen with the others before "
              "any fraud-data result. It is reported next to the primary ensembles, never "
              "chosen after seeing fraud results.",
              "- E2 finding, reported as found: the ensembles rarely beat their best member but "
              "usually beat their median member, and the members' training overlap is only "
              "weakly associated with the gain.", ""]
    return "\n".join(lines)


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("Working tree has uncommitted changes; commit before exporting results.")
    summ = pd.read_csv(OUT / "e2_summary.csv")
    fried = pd.read_csv(OUT / "e2_friedman.csv")
    e2_meta = json.loads((OUT / "e2_runs.meta.json").read_text(encoding="utf-8"))
    meta = {**state, "e2_commit": e2_meta["commit"],
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (OUT / "ebench.md").write_text(markdown(summ, fried, meta), encoding="utf-8")
    print("E-bench:", e2.combo_name(select(summ)), "| E-bench-5%:",
          e2.combo_name(select(summ, SECONDARY_BUDGET)))


if __name__ == "__main__":
    main()
