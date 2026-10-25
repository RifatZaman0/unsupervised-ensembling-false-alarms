"""Score cache: one .npz per (dataset, split seed, detector) under cache/scores/.

Every experiment after E1 re-combines cached scores instead of refitting detectors.
The cache key contains a hash of the detector parameters, the preprocessing and the PyOD
version, so changing any of them writes to a new folder instead of reusing stale scores.
Each file also stores the split's row positions; a cached file whose rows do not match the
current split is rejected rather than silently reused.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyod

from thesis import detectors
from thesis.data import Split
from thesis.paths import CACHE

PREPROCESSING = "zscore-train"  # thesis.data.standardize, statistics from the training split


def cache_key() -> str:
    spec = {"params": detectors.PARAMS, "max_fit_rows": detectors.MAX_FIT_ROWS,
            "preprocessing": PREPROCESSING, "pyod": pyod.__version__,
            "ecod": "train-only-v1"}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]


def cache_path(dataset: str, seed: int | None, detector: str, root: Path = CACHE) -> Path:
    s = "na" if seed is None else str(seed)
    return root / cache_key() / dataset / f"seed{s}" / f"{detector}.npz"


def get_scores(split: Split, detector: str, root: Path = CACHE) -> dict:
    """Return {'train', 'test', 'runtime_s'} for a preprocessed split, computing if needed.

    runtime_s is fit + score time in seconds (from the run that filled the cache).

    The split's seed doubles as the detector seed, so IForest gets 5 seeds over the 5 splits.
    """
    path = cache_path(split.dataset, split.seed, detector, root)
    if path.exists():
        with np.load(path) as z:
            if not (np.array_equal(z["idx_train"], split.idx_train)
                    and np.array_equal(z["idx_test"], split.idx_test)):
                raise RuntimeError(f"{path}: cached rows do not match the split; delete the file")
            return {"train": z["train"], "test": z["test"], "runtime_s": float(z["runtime_s"])}
    t0 = time.perf_counter()
    s_train, s_test = detectors.score(detector, split.X_train, split.X_test, split.seed)
    seconds = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, train=s_train, test=s_test, idx_train=split.idx_train,
             idx_test=split.idx_test, runtime_s=seconds)
    return {"train": s_train, "test": s_test, "runtime_s": seconds}
