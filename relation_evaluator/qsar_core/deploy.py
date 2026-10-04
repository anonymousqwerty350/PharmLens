#!/usr/bin/env python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from .architecture import SEED, ALGOS, fit_full
from .features import (DESCS, standardize, build_descriptor_frames,
                       inference_descriptor_frames)

RDLogger.DisableLog("rdApp.*")


def train_bundle(flat_smiles, y, target, task, svm_proba=True, seed=SEED,
                 inner_cv=None, descs=DESCS, algos=ALGOS):
    if inner_cv is None:
        from sklearn.model_selection import StratifiedKFold
        inner_cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    y = np.asarray(y).astype(int)
    mols = [Chem.MolFromSmiles(s) for s in flat_smiles]
    frames = build_descriptor_frames(mols)

    bundle = {"target": target, "task": task, "descs": list(descs),
              "algos": list(algos), "svm_proba": svm_proba, "seed": seed,
              "fitted": {}, "platt": {}, "feature_columns": {},
              "n_train": int(len(y)), "n_pos": int((y == 1).sum())}
    for desc in descs:
        X = frames[desc].values
        bundle["feature_columns"][desc] = [str(c) for c in frames[desc].columns]
        for algo in algos:
            fr = fit_full(algo, X, y, seed=seed, inner_cv=inner_cv,
                          svm_proba=svm_proba)
            bundle["fitted"][(desc, algo)] = fr.estimator
            if fr.platt is not None:
                bundle["platt"][(desc, algo)] = fr.platt
    return bundle


def save_bundle(bundle, path):
    import os, joblib
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    joblib.dump(bundle, tmp, compress=3)
    os.replace(tmp, path)
    return path


def load_bundle(path):
    import joblib
    return joblib.load(path)


def _model_proba(bundle, desc, algo, X):
    est = bundle["fitted"][(desc, algo)]
    platt = bundle.get("platt", {}).get((desc, algo))
    if platt is not None:
        dec = est.decision_function(X).reshape(-1, 1)
        return platt.predict_proba(dec)[:, 1]
    return est.predict_proba(X)[:, 1]


def predict_bundle(bundle, smiles, per_model=False):
    if isinstance(smiles, str):
        smiles = [smiles]
    descs, algos = bundle["descs"], bundle["algos"]
    pairs = [(d, a) for d in descs for a in algos]

    std, ok_mols, ok_rows = [], [], []
    for i, s in enumerate(smiles):
        flat = standardize(s)[1]
        std.append(flat)
        if flat is not None:
            m = Chem.MolFromSmiles(flat)
            if m is not None:
                ok_mols.append(m)
                ok_rows.append(i)

    proba = np.full((len(smiles), len(pairs)), np.nan)
    if ok_mols:
        feats = inference_descriptor_frames(ok_mols)
        for j, (desc, algo) in enumerate(pairs):
            X = (feats[desc].reindex(columns=bundle["feature_columns"][desc])
                 .fillna(0.0).values)
            proba[ok_rows, j] = _model_proba(bundle, desc, algo, X)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cons = np.nanmean(proba, axis=1)
    pred = pd.array([pd.NA if np.isnan(c) else int(c >= 0.5) for c in cons],
                    dtype="Int64")
    out = pd.DataFrame({
        "input_smiles": smiles,
        "standardized": std,
        "valid": [s is not None for s in std],
        "proba": cons,
        "pred": pred,
    })
    if per_model:
        for j, (desc, algo) in enumerate(pairs):
            out[f"{desc}/{algo}"] = proba[:, j]
    return out
