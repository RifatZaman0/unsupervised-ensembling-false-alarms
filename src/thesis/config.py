"""Protocol switches for the preprocessing and metric choices of the evaluation protocol.

The defaults are the protocol used for every reported result. `python -m thesis.e1` takes a command-line flag for
each one (see `add_arguments`), so a different choice is one rerun. If a choice is changed for
good, change its default here; with the defaults, the score cache key stays what it was before
these switches existed (21ba8c24b9), so no rescoring is needed.

    zscore_inputs           z-score every detector's input with training-split statistics
                            (only changes KNN/LOF; the other four are per-feature affine-invariant)
    k_rule                  K = ceil / round / floor of budget * n_test, at least 1
    pca_drop_zero_variance  PCA drops components with explained-variance ratio <= 1e-10;
                            off = plain PyOD PCA, which gives non-finite scores on some sets
    adbench_dedup           drop exact duplicate (features, label) rows from ADBench sets;
                            off = keep them
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields

K_RULES = ("ceil", "round", "floor")

# Fixed (not switches). KNN / LOF are fitted on a sample of at most this many training rows
# (30,000-row sample on fraud data). The sample is drawn with the
# split seed, or with KNN_LOF_SAMPLE_SEED when the split has none (ULB, IEEE-CIS). ADBench
# training splits are all smaller (largest: 29,720 rows), so the sample never applies there.
KNN_LOF_MAX_FIT_ROWS = 30_000
KNN_LOF_SAMPLE_SEED = 0


@dataclass(frozen=True)
class Protocol:
    zscore_inputs: bool = True
    k_rule: str = "ceil"
    pca_drop_zero_variance: bool = True
    adbench_dedup: bool = False

    def __post_init__(self):
        if self.k_rule not in K_RULES:
            raise ValueError(f"k_rule must be one of {K_RULES}, got {self.k_rule!r}")

    def tag(self) -> str:
        """'' for the defaults, else a folder name listing the changed switches."""
        default, parts = Protocol(), []
        if self.zscore_inputs != default.zscore_inputs:
            parts.append("zscore" if self.zscore_inputs else "no-zscore")
        if self.k_rule != default.k_rule:
            parts.append(f"k-{self.k_rule}")
        if self.pca_drop_zero_variance != default.pca_drop_zero_variance:
            parts.append("pca-drop" if self.pca_drop_zero_variance else "no-pca-drop")
        if self.adbench_dedup != default.adbench_dedup:
            parts.append("dedup" if self.adbench_dedup else "no-dedup")
        return "_".join(parts)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Read at call time by data, detectors, metrics and scorecache; set by thesis.e1 from its flags.
PROTOCOL = Protocol()


def use(protocol: Protocol) -> None:
    global PROTOCOL
    PROTOCOL = protocol


def add_arguments(parser: argparse.ArgumentParser) -> None:
    d = Protocol()
    bool_flag = argparse.BooleanOptionalAction
    parser.add_argument("--zscore", dest="zscore_inputs", action=bool_flag,
                        default=d.zscore_inputs, help="z-score detector inputs (default: on)")
    parser.add_argument("--k-rule", choices=K_RULES, default=d.k_rule,
                        help="K from budget * n_test (default: ceil)")
    parser.add_argument("--pca-drop", dest="pca_drop_zero_variance", action=bool_flag,
                        default=d.pca_drop_zero_variance,
                        help="drop zero-variance PCA components (default: on)")
    parser.add_argument("--dedup", dest="adbench_dedup", action=bool_flag,
                        default=d.adbench_dedup,
                        help="drop duplicate ADBench rows (default: off)")


def from_args(args: argparse.Namespace) -> Protocol:
    return Protocol(args.zscore_inputs, args.k_rule, args.pca_drop_zero_variance,
                    args.adbench_dedup)
