import json
import os
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             matthews_corrcoef, roc_auc_score)

from agent import config


def score(preds: pd.DataFrame, knowledge: Dict[str, Any]) -> Dict[str, Any]:
    total = len(preds)
    ok = preds[preds.pred.isin([0, 1])]
    y, p = ok.label.values, ok.pred.values

    out: Dict[str, Any] = {"n": int(len(ok)), "n_total": int(total),
                           "unparsed": int(total - len(ok))}
    if len(ok) == 0 or len(set(y)) < 2:
        out.update({"auroc": None, "accuracy": None, "macro_f1": None,
                    "binary_f1": None, "mcc": None, "report": None})
        return out

    out["auroc"] = round(float(roc_auc_score(y, p)), 4)
    out["accuracy"] = round(float(accuracy_score(y, p)), 4)
    out["macro_f1"] = round(float(f1_score(y, p, average="macro")), 4)
    out["binary_f1"] = round(float(f1_score(y, p, average="binary")), 4)
    out["mcc"] = round(float(matthews_corrcoef(y, p)), 4)
    out["report"] = classification_report(
        y, p, target_names=knowledge.get("target_names", ["0", "1"]),
        output_dict=True, zero_division=0)
    return out


def summarize(task: str) -> pd.DataFrame:
    root = config.run_dir(task)
    rows: List[Dict[str, Any]] = []
    for run in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        path = os.path.join(root, run, "metrics.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            m = json.load(f)
        rows.append({"run": run, "n": m.get("n"), "unparsed": m.get("unparsed"),
                     "AUROC": m.get("auroc"), "Accuracy": m.get("accuracy"),
                     "Macro-F1": m.get("macro_f1"), "MCC": m.get("mcc")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    for t in (sys.argv[1:] or list(config.TASKS)):
        df = summarize(t)
        print(f"\n=== {t} ===")
        print(df.to_string(index=False) if len(df) else "  (no runs yet)")
