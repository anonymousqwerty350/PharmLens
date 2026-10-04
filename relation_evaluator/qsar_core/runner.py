#!/usr/bin/env python
import argparse
import os
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _log(msg, log_path=None):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    if log_path:
        with open(log_path, "a") as fh:
            fh.write(line + "\n")


def _limit_blas():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _deploy_one(name, csv_path, task, out_path, smiles_col, svm_proba):
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from qsar_core.datasets import load_labeled_dataset
    from qsar_core.deploy import train_bundle, save_bundle
    if os.path.exists(out_path):
        return name, "skip", None
    ds = load_labeled_dataset(csv_path, smiles_col=smiles_col)
    bundle = train_bundle(ds.flat_smiles, ds.y, target=name, task=task,
                          svm_proba=svm_proba)
    save_bundle(bundle, out_path)
    return name, "done", (ds.n, ds.n_pos)


def _eval_one(name, csv_path, task, smiles_col, method, n_splits):
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from qsar_core.datasets import load_labeled_dataset
    from qsar_core.evaluate import scaffold_cv_metrics
    ds = load_labeled_dataset(csv_path, smiles_col=smiles_col)
    fold_df, summ = scaffold_cv_metrics(ds.flat_smiles, ds.y, target=name,
                                        n_splits=n_splits, method=method)
    return name, ds.n, ds.n_pos, fold_df, summ


def run_deploy(targets, data_dir, model_dir, task, smiles_col=None, only=None,
               svm_proba=True, workers=1, log_path=None):
    os.makedirs(model_dir, exist_ok=True)
    names = list(only) if only else list(targets)
    tasks = [(n, os.path.join(data_dir, targets[n]), task,
              os.path.join(model_dir, f"{n}.pkl"), smiles_col, svm_proba)
             for n in names]
    _log(f"[deploy] {len(tasks)} target(s), workers={workers}", log_path)
    t0 = time.time()

    def _report(res):
        name, status, info = res
        if status == "skip":
            _log(f"[deploy] {name}: skip (pkl exists)", log_path)
        else:
            n, n_pos = info
            _log(f"[deploy] {name}: N={n} pos={n_pos} -> models/{name}.pkl "
                 f"[{(time.time()-t0)/60:.1f}m]", log_path)

    if workers and workers > 1:
        _limit_blas()
        from joblib import Parallel, delayed
        for res in Parallel(n_jobs=workers, backend="loky",
                            return_as="generator")(
                delayed(_deploy_one)(*a) for a in tasks):
            _report(res)
    else:
        for a in tasks:
            _report(_deploy_one(*a))
    _log(f"[deploy] done in {(time.time()-t0)/60:.1f}m", log_path)


def run_eval(targets, data_dir, out_dir, task, smiles_col=None, only=None,
             method="scaffold", n_splits=5, workers=1, log_path=None):
    os.makedirs(out_dir, exist_ok=True)
    names = list(only) if only else list(targets)
    tasks = [(n, os.path.join(data_dir, targets[n]), task, smiles_col,
              method, n_splits) for n in names]
    _log(f"[eval] {len(tasks)} target(s), {n_splits}-fold {method} CV, "
         f"workers={workers}", log_path)
    t0 = time.time()
    fold_frames, summ_frames = [], []

    def _collect(res):
        name, n, n_pos, fold_df, summ = res
        cons = summ[summ["descriptor"] == "Consensus"]
        tag = ""
        if len(cons):
            r = cons.iloc[0]
            tag = (f" Consensus AUPRC={r.AUPRC_mean:.3f} AUROC={r.AUROC_mean:.3f} "
                   f"CCR={r.CCR_mean:.3f} MCC={r.MCC_mean:.3f}")
        _log(f"[eval] {name}: N={n} pos={n_pos}{tag} "
             f"[{(time.time()-t0)/60:.1f}m]", log_path)
        fold_frames.append(fold_df)
        summ_frames.append(summ)

    if workers and workers > 1:
        _limit_blas()
        from joblib import Parallel, delayed
        for res in Parallel(n_jobs=workers, backend="loky",
                            return_as="generator")(
                delayed(_eval_one)(*a) for a in tasks):
            _collect(res)
    else:
        for a in tasks:
            _collect(_eval_one(*a))

    fold_all = pd.concat(fold_frames, ignore_index=True)
    summ_all = pd.concat(summ_frames, ignore_index=True)
    fold_csv = os.path.join(out_dir, "cv_fold_metrics.csv")
    summ_csv = os.path.join(out_dir, "cv_summary.csv")
    _append_csv(fold_all, fold_csv, keys=["target", "descriptor", "algorithm", "fold"])
    _append_csv(summ_all, summ_csv, keys=["target", "descriptor", "algorithm"])
    _log(f"[eval] wrote {os.path.basename(fold_csv)}, "
         f"{os.path.basename(summ_csv)}", log_path)
    _log(f"[eval] done in {(time.time()-t0)/60:.1f}m", log_path)


def _append_csv(df, path, keys):
    if os.path.exists(path):
        try:
            old = pd.read_csv(path)
            merged = pd.concat([old, df], ignore_index=True)
            merged = merged.drop_duplicates(subset=keys, keep="last")
            df = merged
        except Exception:
            pass
    df.to_csv(path, index=False)


def _apply_gate(targets, data_dir, base_dir, log_path, smiles_col=None, criteria=None,
                enforce=True, refresh=False):
    from qsar_core import eligibility

    crit = criteria or eligibility.DEFAULT
    report_path = os.path.join(base_dir, "eligibility.csv")
    verdicts = eligibility.screen(targets, data_dir, criteria=crit, smiles_col=smiles_col,
                                  cache_path=report_path, refresh=refresh)
    eligibility.write_report(verdicts, report_path, criteria=crit)

    ok = [v for v in verdicts if v.trainable]
    bad = [v for v in verdicts if not v.trainable]
    _log(f"[gate] {len(ok)}/{len(verdicts)} targets passed the training criteria "
         f"({eligibility.criteria_line(crit)})", log_path)
    for v in bad:
        n = v.stats.get("n")
        size = f"N={n} minority={v.stats.get('minority')}" if n is not None else "no data"
        verb = "excluded" if enforce else "below criteria (forced through with --no-gate)"
        _log(f"[gate] {v.target}: {verb} — {size}; {v.reason}", log_path)
    _log(f"[gate] rationale written to {os.path.basename(report_path)}", log_path)
    return [v.target for v in ok]


def train_main(task, targets, base_dir, data_subdir="dataset",
               model_subdir="models", smiles_col=None, svm_proba=True, argv=None,
               criteria=None):
    ap = argparse.ArgumentParser(
        description=f"Train per-{task} QSAR deployment models + scaffold-CV eval.")
    ap.add_argument("--stage", choices=["deploy", "eval", "all"], default="all")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="restrict to a subset (e.g. --targets EGFR)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel target workers (loky)")
    ap.add_argument("--split", choices=["scaffold", "random"], default="scaffold",
                    help="CV split for --stage eval")
    ap.add_argument("--no-gate", action="store_true",
                    help="fit every target on the panel, however thin its dataset. The "
                         "verdicts are still reported; only the skipping is disabled.")
    ap.add_argument("--gate-only", action="store_true",
                    help="report eligibility and exit without training")
    ap.add_argument("--regate", action="store_true",
                    help="recompute eligibility instead of reusing eligibility.csv rows "
                         "whose dataset CSV is unchanged")
    args = ap.parse_args(argv)

    data_dir = os.path.join(base_dir, data_subdir)
    model_dir = os.path.join(base_dir, model_subdir)
    log_path = os.path.join(base_dir, "train.log")
    only = args.targets
    if only:
        unknown = [t for t in only if t not in targets]
        if unknown:
            sys.exit(f"unknown target(s): {unknown}; valid: {list(targets)}")

    keep = _apply_gate(targets, data_dir, base_dir, log_path, smiles_col=smiles_col,
                       criteria=criteria, enforce=not args.no_gate, refresh=args.regate)
    if args.gate_only:
        return
    if not args.no_gate:
        allowed = set(keep)
    else:
        absent = [t for t, f in targets.items()
                  if not os.path.exists(os.path.join(data_dir, f))]
        if absent:
            _log(f"[gate] --no-gate, but the dataset CSV itself is missing, so skipped: {absent}", log_path)
        allowed = set(targets) - set(absent)
    targets = {t: f for t, f in targets.items() if t in allowed}
    if only:
        dropped = [t for t in only if t not in allowed]
        if dropped:
            _log(f"[gate] named by --targets but not trainable -> skipped: {dropped}"
                 f"{'' if args.no_gate else ' (use --no-gate to force)'}", log_path)
        only = [t for t in only if t in allowed]
    if not (only if only is not None else list(targets)):
        sys.exit("no targets to train — see the reason column of eligibility.csv.")

    if args.stage in ("deploy", "all"):
        run_deploy(targets, data_dir, model_dir, task, smiles_col=smiles_col,
                   only=only, svm_proba=svm_proba, workers=args.workers,
                   log_path=log_path)
    if args.stage in ("eval", "all"):
        run_eval(targets, data_dir, base_dir, task, smiles_col=smiles_col,
                 only=only, method=args.split, workers=args.workers,
                 log_path=log_path)


def predict(target, smiles, model_dir, per_model=False):
    from qsar_core.deploy import load_bundle, predict_bundle
    bundle = load_bundle(os.path.join(model_dir, f"{target}.pkl"))
    return predict_bundle(bundle, smiles, per_model=per_model)

