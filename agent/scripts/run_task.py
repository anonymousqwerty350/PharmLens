import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from agent import config
from agent.agents import physchem_agent
from agent.eval.evaluate import score
from agent.graph import build_graph, load_knowledge, run_one, trace

ABLATIONS = ["no_physchem", "no_gate", "no_refcomp", "no_knowledge",
             "no_recognition_gate",
             "no_lce", "no_pce", "no_ie"]
FEATURES = []


class _Tee:
    def __init__(self, path: str):
        self._log = open(path, "a", buffering=1)
        self._stdout = sys.stdout

    def write(self, s):
        self._log.write(s)
        return self._stdout.write(s)

    def flush(self):
        self._log.flush()
        self._stdout.flush()


def knowledge_stamp(task: str, knowledge: dict) -> dict:
    path = config.TASKS[task]["knowledge"]
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "knowledge_file": os.path.basename(path),
        "knowledge_path": path,
        "knowledge_sha256": sha[:12],
        "knowledge_mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(os.path.getmtime(path))),
        "protein_panel": list((knowledge.get("relevant_proteins") or {}).keys()),
    }


def load_test_set(task: str, limit: int = None, seed: int = 42,
                  split: str = "test") -> pd.DataFrame:
    t = config.TASKS[task]
    df = pd.read_csv(os.path.join(t["dataset_dir"], f"{split}.csv"))
    df = df.rename(columns={t["smiles_col"]: "smiles", t["label_col"]: "label"})
    df = df[["smiles", "label"]].dropna()
    df["label"] = df["label"].astype(int)

    from agent.tools.evidence import cache_path
    df = df[df["smiles"].apply(lambda s: os.path.exists(cache_path(s, task, split)))]
    df = df.reset_index(drop=True)

    if limit:
        df = df.sample(min(limit, len(df)), random_state=seed).reset_index(drop=True)
    return df


def write_reasoning(records, knowledge, out_dir: str):
    names = knowledge.get("target_names", ["class 0", "class 1"])

    with open(os.path.join(out_dir, "reasoning.jsonl"), "w") as jf, \
         open(os.path.join(out_dir, "reasoning.txt"), "w") as tf:
        for i, rec in enumerate(records, 1):
            if rec is None:
                continue
            label, pred = rec.get("label"), rec.get("prediction")
            correct = None if pred not in (0, 1) else int(pred == label)

            relations = [
                {"protein": p["protein"], "relation": p.get("relation"),
                 "source": p.get("relation_source"),
                 "ml_proba": p.get("ml_proba"), "band": p.get("band"),
                 "docking_score": p.get("docking_score"),
                 "docking_used": p.get("docking_used")}
                for p in (rec.get("proteins") or [])
            ]

            jf.write(json.dumps({
                "idx": i,
                "smiles": rec.get("smiles"),
                "label": label, "prediction": pred, "correct": correct,
                "physchem_verdict": (rec.get("physchem_verdict") or {}).get("verdict"),
                "physchem_rationale": (rec.get("physchem_verdict") or {}).get("rationale"),
                "protein_relations": relations,
                "escalated": rec.get("escalated"),
                "reasoning": rec.get("prediction_text"),
            }, ensure_ascii=False) + "\n")

            mark = "?" if correct is None else ("O" if correct else "X")
            tf.write(f"{'=' * 78}\n")
            tf.write(f"[{i}] {mark}  label={label} ({names[label] if label in (0, 1) else '?'})"
                     f"  pred={pred}\n")
            tf.write(f"SMILES: {rec.get('smiles')}\n")
            tf.write(f"{'-' * 78}\n")
            pv = rec.get("physchem_verdict") or {}
            tf.write(f"Physicochemical: {pv.get('verdict')} "
                     f"(confidence {pv.get('confidence')})\n")
            for r in relations:
                p_txt = "n/a" if r["ml_proba"] is None else f"{r['ml_proba']:.3f}"
                d_txt = "n/a" if r["docking_score"] is None else f"{r['docking_score']:.2f}"
                tf.write(f"  {r['protein']:<8} relation={str(r['relation']):<18} "
                         f"via={str(r['source']):<21} P={p_txt:<6} dock={d_txt:<7} "
                         f"used={r['docking_used']}\n")
            tf.write(f"{'-' * 78}\n")
            tf.write(f"{rec.get('prediction_text') or '(no output)'}\n\n")


def load_done(out_dir: str):
    path = os.path.join(out_dir, "trace.jsonl")
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("smiles"):
                done[rec["smiles"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(config.TASKS))
    ap.add_argument("--split", choices=["test", "train_val"], default="test",
                    help="which split to run. The dataset CSV and the docking cache follow it")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ablation", nargs="*", default=[], choices=ABLATIONS)
    ap.add_argument("--features", nargs="*", default=[], choices=FEATURES,
                    help="opt-in additions to the baseline pipeline (off by default)")
    ap.add_argument("--model", default=config.DEFAULT_MODEL,
                    help="backbone LLM. A name listed in config.MODEL_BACKENDS goes to that endpoint; "
                         "anything else is treated as an OpenAI model (default config.DEFAULT_MODEL)")
    args = ap.parse_args()

    config.DEFAULT_MODEL = args.model

    ablation = {a: True for a in args.ablation}
    ablation.update({f: True for f in args.features})
    run_name = args.run_name or ("full" if not ablation else "-".join(sorted(ablation)))
    out_dir = os.path.join(config.run_dir(args.task), run_name)
    os.makedirs(out_dir, exist_ok=True)

    sys.stdout = _Tee(os.path.join(out_dir, f"{run_name}.log"))

    meta_path = os.path.join(out_dir, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev = json.load(f)
        for key, now, absent in (("split", args.split, "test"),
                                 ("model", args.model, args.model)):
            was = prev.get(key, absent)
            if was != now:
                sys.exit(f"[abort] {out_dir} already holds a '{was}' {key} run; "
                         f"--{key} {now} would mix two kinds of output into one directory. "
                         f"Use a new --run-name.")
    with open(meta_path, "w") as f:
        json.dump({"task": args.task, "split": args.split,
                   "run": run_name, "model": args.model},
                  f, indent=2)

    knowledge = load_knowledge(args.task)
    stamp = knowledge_stamp(args.task, knowledge)
    df = load_test_set(args.task, args.limit, split=args.split)
    print(f"[{args.task}/{run_name}] {len(df)} molecules from {args.split} "
          f"(label=1: {int(df.label.sum())}, label=0: {int((df.label == 0).sum())})")
    print(f"  knowledge: {stamp['knowledge_file']}  "
          f"sha256={stamp['knowledge_sha256']}  modified {stamp['knowledge_mtime']}")
    print(f"  proteins: {', '.join(stamp['protein_panel']) or '(none)'}")
    _backend = config.MODEL_BACKENDS.get(args.model)
    if not _backend:
        _where = " (OpenAI)"
    elif _backend.get("base_url") or _backend.get("base_url_env"):
        _where = f" -> {config.backend_url(_backend)}"
    else:
        _where = f" -> {_backend['transport']}:{_backend['api_model']}"
    if _backend and _backend.get("max_tokens"):
        _where += f" (max_tokens={_backend['max_tokens']})"
    print(f"  model: {args.model}{_where}")
    if ablation:
        print(f"  flags: {', '.join(sorted(ablation))}")

    if not ablation.get("no_physchem"):
        sel = physchem_agent.select_tools(args.task, knowledge)
        print(f"  physchem tools: {len(sel['descriptors'])} descriptors, "
              f"{len(sel['functional_groups'])} functional groups")

    graph = build_graph()
    t0 = time.time()
    records = [None] * len(df)

    def work(i: int, smiles: str, label: int):
        state = run_one(graph, args.task, knowledge, smiles, label=label, ablation=ablation,
                        split=args.split)
        return i, trace(state)

    already = load_done(out_dir)
    todo = []
    for i, r in df.iterrows():
        hit = already.get(r.smiles)
        if hit is None:
            todo.append(i)
        else:
            records[i] = hit
    if already:
        print(f"  resuming: {len(df) - len(todo)} molecules already traced, "
              f"{len(todo)} left to run")

    trace_path = os.path.join(out_dir, "trace.jsonl")
    with open(trace_path, "a") as jf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, i, df.smiles[i], int(df.label[i])): i for i in todo}
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                i, rec = fut.result()
            except Exception as exc:                                # noqa: BLE001
                rec = {"smiles": df.smiles[i], "label": int(df.label[i]),
                       "prediction": -1, "errors": [f"{exc}\n{traceback.format_exc()}"]}
            records[i] = rec
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jf.flush()
            done += 1
            ok = "✓" if rec.get("prediction") == rec.get("label") else "✗"
            print(f"  [{done:3d}/{len(todo)}] label={rec['label']} "
                  f"pred={rec.get('prediction')} {ok}  {rec['smiles'][:44]}")

    with open(trace_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_reasoning(records, knowledge, out_dir)

    preds = pd.DataFrame({"smiles": df.smiles, "label": df.label,
                          "pred": [r.get("prediction", -1) for r in records]})
    preds.to_csv(os.path.join(out_dir, "preds.csv"), index=False)

    metrics = score(preds, knowledge)
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    metrics["run"] = run_name
    metrics["split"] = args.split
    metrics["model"] = args.model
    metrics.update(stamp)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    def _fmt(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)

    print(f"\n{'=' * 56}")
    print(f"[{args.task}/{run_name}]  n={metrics['n']}  unparsed={metrics['unparsed']}")
    print(f"  AUROC     {_fmt(metrics['auroc'])}")
    print(f"  Accuracy  {_fmt(metrics['accuracy'])}")
    print(f"  Macro-F1  {_fmt(metrics['macro_f1'])}")
    print(f"  MCC       {_fmt(metrics['mcc'])}")
    print(f"  elapsed   {metrics['elapsed_sec']}s   -> {out_dir}")
    _cap = (_backend or {}).get("max_tokens")
    print(config.usage_report({k: _cap for k in config.LLM_USAGE}), end="")
    print("=" * 56)


if __name__ == "__main__":
    sys.exit(main())
