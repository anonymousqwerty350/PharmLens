#!/usr/bin/env python
from collections import namedtuple

import pandas as pd

from .features import standardize

SMILES_CANDIDATES = ["smiles", "canonical_smiles"]

LabeledDataset = namedtuple("LabeledDataset", "flat_smiles y n n_pos")


def _find_smiles_col(columns, smiles_col=None):
    lower = {c.lower(): c for c in columns}
    if smiles_col is not None:
        if smiles_col in columns:
            return smiles_col
        if smiles_col.lower() in lower:
            return lower[smiles_col.lower()]
        raise KeyError(f"smiles_col={smiles_col!r} not in columns {list(columns)}")
    for cand in SMILES_CANDIDATES:
        if cand in lower:
            return lower[cand]
    raise KeyError(
        f"no SMILES column found in {list(columns)}; tried {SMILES_CANDIDATES} "
        f"(case-insensitive). Pass smiles_col= explicitly.")


def load_labeled_dataset(csv_path, smiles_col=None, label_col="label"):
    df = pd.read_csv(csv_path)
    scol = _find_smiles_col(df.columns, smiles_col)
    if label_col not in df.columns:
        raise KeyError(f"label_col={label_col!r} not in columns {list(df.columns)}")

    flat = [standardize(s)[1] for s in df[scol]]
    df = df.assign(flat_smiles=flat)
    df = df[df["flat_smiles"].notna()].copy()
    df = df.sort_values(label_col, ascending=False)
    df = df.drop_duplicates(subset="flat_smiles", keep="first").reset_index(drop=True)

    y = df[label_col].astype(int).to_numpy()
    flat_smiles = df["flat_smiles"].tolist()
    return LabeledDataset(flat_smiles, y, len(flat_smiles), int((y == 1).sum()))
