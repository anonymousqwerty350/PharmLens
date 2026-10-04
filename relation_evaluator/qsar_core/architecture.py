#!/usr/bin/env python
from collections import namedtuple
import numpy as np

SEED = 0
ALGOS = ["DNN", "RF", "SVM", "XGB"]

GRIDS = {
    "DNN": {"clf__hidden_layer_sizes": [(256, 128, 64), (128, 64, 32)],
            "clf__alpha": [1e-4, 1e-3]},
    "RF":  {"clf__n_estimators": [100, 300],
            "clf__max_depth": [10, None]},
    "SVM": {"clf__kernel": ["rbf"],
            "clf__C": [1, 10],
            "clf__gamma": ["scale", 1e-2]},
    "XGB": {"clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 6],
            "clf__learning_rate": [0.1]},
}


def make_model(algo, svm_proba=False, seed=SEED):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from xgboost import XGBClassifier
    if algo == "DNN":
        return MLPClassifier(max_iter=500, random_state=seed), GRIDS["DNN"]
    if algo == "RF":
        return (RandomForestClassifier(class_weight="balanced",
                                       random_state=seed, n_jobs=1),
                GRIDS["RF"])
    if algo == "SVM":
        return (SVC(probability=svm_proba, class_weight="balanced",
                    random_state=seed),
                GRIDS["SVM"])
    if algo == "XGB":
        return (XGBClassifier(eval_metric="logloss", random_state=seed,
                              n_jobs=1, verbosity=0),
                GRIDS["XGB"])
    raise ValueError(f"unknown algorithm: {algo}")


def build_estimator(algo, svm_proba=False, seed=SEED):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    mdl, grid = make_model(algo, svm_proba=svm_proba, seed=seed)
    return Pipeline([("sc", StandardScaler()), ("clf", mdl)]), grid


FitResult = namedtuple("FitResult", "proba pred best_params estimator platt")


def _fit_grid(algo, X, y, seed, inner_cv, scoring, svm_proba=False):
    from sklearn.model_selection import GridSearchCV
    pipe, grid = build_estimator(algo, svm_proba=svm_proba, seed=seed)
    gs = GridSearchCV(pipe, grid, cv=inner_cv, scoring=scoring, n_jobs=1)
    gs.fit(X, y)
    return gs


def _platt(gs, X, y):
    from sklearn.linear_model import LogisticRegression
    d = gs.decision_function(X).reshape(-1, 1)
    return LogisticRegression().fit(d, y)


def fit_fold(algo, Xtr, ytr, Xte, seed=SEED, inner_cv=3, scoring="roc_auc"):
    gs = _fit_grid(algo, Xtr, ytr, seed, inner_cv, scoring)
    if algo == "SVM":
        platt = _platt(gs, Xtr, ytr)
        proba = platt.predict_proba(gs.decision_function(Xte).reshape(-1, 1))[:, 1]
    else:
        platt = None
        proba = gs.predict_proba(Xte)[:, 1]
    pred = gs.predict(Xte)
    return FitResult(proba, pred, gs.best_params_, gs.best_estimator_, platt)


def fit_full(algo, X, y, seed=SEED, inner_cv=3, scoring="roc_auc", svm_proba=False):
    gs = _fit_grid(algo, X, y, seed, inner_cv, scoring, svm_proba=svm_proba)
    platt = _platt(gs, X, y) if (algo == "SVM" and not svm_proba) else None
    return FitResult(None, None, gs.best_params_, gs.best_estimator_, platt)


def consensus(probas):
    return np.mean(probas, axis=0)


def fold_metrics(y, proba, pred):
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                  confusion_matrix, matthews_corrcoef)
    y = np.asarray(y)
    both_classes = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return dict(AUPRC=average_precision_score(y, proba) if y.sum() > 0 else np.nan,
                AUROC=roc_auc_score(y, proba) if both_classes else np.nan,
                CCR=(sens + spec) / 2,
                MCC=matthews_corrcoef(y, pred))

