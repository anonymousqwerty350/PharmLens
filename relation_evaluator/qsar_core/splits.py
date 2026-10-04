#!/usr/bin/env python
from collections import defaultdict
import numpy as np


def _generic_scaffold(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        gen = MurckoScaffold.MakeScaffoldGeneric(scaf)
        s = Chem.MolToSmiles(gen)
        return s if s else Chem.MolToSmiles(scaf)
    except Exception:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))


def scaffold_folds(smiles_list, n_splits=5, seed=0):
    smiles_list = list(smiles_list)
    groups = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        groups[_generic_scaffold(smi)].append(i)
    ordered = sorted(groups.values(),
                     key=lambda idx: (-len(idx), smiles_list[idx[0]]))
    fold_sizes = [0] * n_splits
    fold_idx = [[] for _ in range(n_splits)]
    for idx in ordered:
        j = int(np.argmin(fold_sizes))
        fold_idx[j].extend(idx)
        fold_sizes[j] += len(idx)
    assign = np.full(len(smiles_list), -1, dtype=int)
    for f in range(n_splits):
        assign[fold_idx[f]] = f
    return assign


def random_folds(y, n_splits=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    y = np.asarray(y)
    assign = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        assign[te] = f
    return assign


def make_folds(smiles=None, y=None, method="scaffold", n_splits=5, seed=0):
    if method == "scaffold":
        if smiles is None:
            raise ValueError("scaffold split needs smiles=")
        return scaffold_folds(smiles, n_splits=n_splits, seed=seed)
    if method == "random":
        if y is None:
            raise ValueError("random split needs y=")
        return random_folds(y, n_splits=n_splits, seed=seed)
    raise ValueError(f"unknown split method {method!r} (use 'scaffold' or 'random')")
