#!/usr/bin/env python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

from .architecture import SEED, ALGOS, fit_fold, fold_metrics
from .features import DESCS, build_descriptor_frames
from .splits import make_folds

RDLogger.DisableLog("rdApp.*")


def scaffold_cv_metrics(flat_smiles, y, target=None, n_splits=5,
                        method="scaffold", seed=SEED, descs=DESCS, algos=ALGOS):
    y = np.asarray(y).astype(int)
    assign = make_folds(smiles=flat_smiles, y=y, method=method,
                        n_splits=n_splits, seed=seed)
    mols = [Chem.MolFromSmiles(s) for s in flat_smiles]
    frames = build_descriptor_frames(mols)

    rows = []
    for f in range(n_splits):
        te = np.where(assign == f)[0]
        tr = np.where(assign != f)[0]
        if len(te) == 0 or len(tr) == 0:
            continue
        yte = y[te]
        fold_probas = []
        for desc in descs:
            X = frames[desc].values
            for algo in algos:
                fr = fit_fold(algo, X[tr], y[tr], X[te], seed=seed)
                m = fold_metrics(yte, fr.proba, fr.pred)
                rows.append(dict(target=target, descriptor=desc,
                                 algorithm=algo, fold=f, **m))
                fold_probas.append(fr.proba)
        cons = np.mean(fold_probas, axis=0)
        cpred = (cons >= 0.5).astype(int)
        m = fold_metrics(yte, cons, cpred)
        rows.append(dict(target=target, descriptor="Consensus",
                         algorithm="Consensus", fold=f, **m))

    fold_df = pd.DataFrame(rows)
    g = fold_df.groupby(["target", "descriptor", "algorithm"], dropna=False)[
        ["AUPRC", "AUROC", "CCR", "MCC"]]
    summ = g.agg(["mean", "std"])
    summ.columns = [f"{metric}_{stat}" for metric, stat in summ.columns]
    summ = summ.reset_index()
    return fold_df, summ
