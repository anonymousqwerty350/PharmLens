#!/usr/bin/env python
import collections
import os
from dataclasses import asdict, dataclass, field
from typing import List

import numpy as np


@dataclass(frozen=True)
class Criteria:
    min_minority: int = 50

    min_minority_scaffolds: int = 20

    max_scaffold_overlap: float = 0.45

    min_fold_minority: int = 5

    n_splits: int = 5


DEFAULT = Criteria()

REPORT_COLUMNS = ("target", "trainable", "reason", "n", "n_pos", "n_neg", "minority",
                  "minority_label", "minority_scaffolds", "scaffold_overlap",
                  "fold_minority", "csv_mtime", "csv_size")


@dataclass
class Verdict:
    target: str
    trainable: bool
    reasons: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def reason(self):
        return "; ".join(self.reasons) if self.reasons else "ok"

    def row(self):
        r = {"target": self.target, "trainable": self.trainable, "reason": self.reason}
        r.update({k: self.stats.get(k) for k in REPORT_COLUMNS if k not in r})
        return r


def dataset_stats(csv_path, smiles_col=None, n_splits=5):
    from .datasets import load_labeled_dataset
    from .splits import _generic_scaffold, scaffold_folds

    ds = load_labeled_dataset(csv_path, smiles_col=smiles_col)
    y = np.asarray(ds.y)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = n - n_pos
    minority_label = 1 if n_pos <= n_neg else 0
    minority = min(n_pos, n_neg)

    stats = {"n": n, "n_pos": n_pos, "n_neg": n_neg, "minority": minority,
             "minority_label": minority_label, "minority_scaffolds": 0,
             "scaffold_overlap": 0.0, "fold_minority": 0}
    if n == 0 or minority == 0:
        return stats

    labels_by_scaffold = collections.defaultdict(set)
    for smi, label in zip(ds.flat_smiles, y):
        labels_by_scaffold[_generic_scaffold(smi)].add(int(label))
    minority_scaffolds = {s for s, ls in labels_by_scaffold.items() if minority_label in ls}
    mixed = {s for s, ls in labels_by_scaffold.items() if len(ls) > 1}
    stats["minority_scaffolds"] = len(minority_scaffolds)
    stats["scaffold_overlap"] = (round(len(mixed) / len(minority_scaffolds), 3)
                                 if minority_scaffolds else 0.0)

    if n >= n_splits:
        assign = scaffold_folds(ds.flat_smiles, n_splits=n_splits)
        stats["fold_minority"] = min(int((y[assign == f] == minority_label).sum())
                                     for f in range(n_splits))
    return stats


def check(target, csv_path, criteria=DEFAULT, smiles_col=None):
    if not os.path.exists(csv_path):
        return Verdict(target, False, ["no dataset CSV (build not run)"], {})
    stat = os.stat(csv_path)
    meta = {"csv_mtime": int(stat.st_mtime), "csv_size": stat.st_size}
    try:
        stats = dataset_stats(csv_path, smiles_col=smiles_col, n_splits=criteria.n_splits)
    except Exception as e:                                        # noqa: BLE001
        return Verdict(target, False, [f"cannot read dataset: {type(e).__name__}: {e}"],
                       meta)
    stats.update(meta)

    reasons = []
    if stats["n"] == 0:
        reasons.append("0 molecules left after labelling")
    elif stats["minority"] == 0:
        side = "negative" if stats["n_pos"] else "positive"
        reasons.append(f"only one class present ({side} 0) — not a classification problem")
    else:
        if stats["minority"] < criteria.min_minority:
            reasons.append(f"minority class {stats['minority']} < {criteria.min_minority}")
        if stats["minority_scaffolds"] < criteria.min_minority_scaffolds:
            reasons.append(f"minority-class scaffolds {stats['minority_scaffolds']} "
                           f"< {criteria.min_minority_scaffolds} (a single class)")
        if stats["scaffold_overlap"] > criteria.max_scaffold_overlap:
            reasons.append(f"scaffold overlap {stats['scaffold_overlap']:.2f} "
                           f"> {criteria.max_scaffold_overlap} (no scaffold split possible)")
        if stats["fold_minority"] < criteria.min_fold_minority:
            reasons.append(f"smallest CV-fold minority class {stats['fold_minority']} "
                           f"< {criteria.min_fold_minority} (metrics undefined)")
    return Verdict(target, not reasons, reasons, stats)


def screen(targets, data_dir, criteria=DEFAULT, smiles_col=None, cache_path=None,
           refresh=False):
    cached = _load_cache(cache_path) if (cache_path and not refresh) else {}
    verdicts = []
    for name, fname in targets.items():
        csv_path = os.path.join(data_dir, fname)
        hit = cached.get(name)
        if hit and os.path.exists(csv_path):
            stat = os.stat(csv_path)
            if (hit.get("csv_mtime") == int(stat.st_mtime)
                    and hit.get("csv_size") == stat.st_size):
                verdicts.append(Verdict(name, bool(hit["trainable"]),
                                        [] if hit.get("reason") == "ok"
                                        else str(hit.get("reason", "")).split("; "),
                                        {k: hit.get(k) for k in REPORT_COLUMNS
                                         if k not in ("target", "trainable", "reason")}))
                continue
        verdicts.append(check(name, csv_path, criteria=criteria, smiles_col=smiles_col))
    return verdicts


def write_report(verdicts, path, criteria=DEFAULT):
    import pandas as pd

    df = pd.DataFrame([v.row() for v in verdicts])
    for col in REPORT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(REPORT_COLUMNS)]
    df.to_csv(path, index=False)
    return df


def _load_cache(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(path)
        return {r["target"]: r.to_dict() for _, r in df.iterrows()}
    except Exception:                                             # noqa: BLE001
        return {}


def criteria_line(criteria=DEFAULT):
    return ", ".join(f"{k}={v}" for k, v in asdict(criteria).items())
