"""Load and check frozen_config.json (README § Layout).

`load()` returns the parsed config after `validate()`; experiments on fraud data read their
ensembles, budgets and thresholds from here, so they cannot drift from what was frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

from thesis import detectors, fusion
from thesis.paths import FROZEN_CONFIG


def validate(cfg: dict) -> None:
    pool = cfg["detectors"]["pool"]
    if tuple(pool) != detectors.DETECTORS:
        raise ValueError(f"pool {pool} is not the detector pool {detectors.DETECTORS}")
    for name, e in cfg["ensembles"].items():
        m = e["members"]
        if not set(m) <= set(pool):
            raise ValueError(f"{name}: members {sorted(set(m) - set(pool))} not in the pool")
        if len(set(m)) != len(m) or len(m) < 2:
            raise ValueError(f"{name}: needs at least 2 distinct members, got {m}")
        if e["role"] not in ("primary", "secondary"):
            raise ValueError(f"{name}: unknown role {e['role']!r}")
    configs = [cfg["fusion"]["primary"], *cfg["fusion"]["sensitivity"]]
    pairs = [(c["calibration"], c["aggregation"]) for c in configs]
    if pairs[0] != fusion.PRIMARY or sorted(pairs) != sorted(fusion.FUSION_CONFIGS):
        raise ValueError(f"fusion configs {pairs} do not match {fusion.FUSION_CONFIGS}")
    for data, b in cfg["budgets"].items():
        vals = b["values"]
        if not all(0 < v < 1 for v in vals) or vals != sorted(set(vals)):
            raise ValueError(f"{data}: budgets {vals} must be distinct, ascending, in (0, 1)")
        if b["primary"] not in vals:
            raise ValueError(f"{data}: primary budget {b['primary']} not among {vals}")
    for data, t in cfg["thresholds"].items():
        q = t["training_score_quantiles"]
        if not all(0.5 < v < 1 for v in q) or len(set(q)) != len(q):
            raise ValueError(f"{data}: quantiles {q} must be distinct and in (0.5, 1)")


def load(path: Path = FROZEN_CONFIG) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(cfg)
    return cfg
