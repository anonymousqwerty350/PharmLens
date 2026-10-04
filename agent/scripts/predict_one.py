import contextlib
import glob
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import config                                    # noqa: E402
from agent.agents import physchem_agent                     # noqa: E402
from agent.graph import build_graph, run_one                # noqa: E402
from agent.registry.proteins import resolve_proteins        # noqa: E402
from agent.tools import evidence as ev                      # noqa: E402

ADHOC_SPLIT = "adhoc"

RECEPTOR_ROOT = config.DOCKING_RESULTS
WORK_ROOT = os.path.join(config.CACHE_DIR, "adhoc_docking")
UNIDOCK_MAX_ATOMS = 100


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self._stream = stream
        self.buf = io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self._stream.write(s)

    def flush(self):
        self._stream.flush()


def resolve_knowledge(task: str, knowledge: Union[None, str, Dict[str, Any]] = None):
    if isinstance(knowledge, dict):
        return knowledge, None

    if knowledge is None:
        path = config.TASKS[task]["knowledge"]
    elif os.path.isabs(knowledge) or os.path.exists(knowledge):
        path = knowledge
    else:
        path = config.knowledge_path(config.TASKS[task]["knowledge_group"], knowledge)

    if not os.path.exists(path):
        directory = os.path.join(config.KNOWLEDGE_DIR, config.TASKS[task]["knowledge_group"])
        available = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
        raise FileNotFoundError(
            f"knowledge file not found: {path}\n"
            f"  available for '{task}': {available}")
    with open(path) as f:
        return json.load(f), path


def _sha(path: Optional[str], knowledge: Dict[str, Any]) -> str:
    if path:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:12]
    blob = json.dumps(knowledge, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def list_tasks() -> None:
    for name, t in config.TASKS.items():
        try:
            with open(t["knowledge"]) as f:
                k = json.load(f)
            panel = ", ".join((k.get("relevant_proteins") or {}).keys()) or "(none)"
        except Exception as exc:                                  # noqa: BLE001
            panel = f"<knowledge unreadable: {exc}>"
        print(f"{name:<28} {os.path.basename(t['knowledge']):<42} {panel}")


def _panel_pdbs(knowledge: Dict[str, Any]) -> Dict[str, str]:
    out = {}
    for p in resolve_proteins(knowledge):
        if p["pdb_id"] and p["pdb_id"] != "unknown":
            out.setdefault(p["pdb_id"], p["protein"])
    return out


def _usable(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("docking_score") is not None


def harvest_cached(smiles: str, wanted: List[str]) -> Dict[str, Any]:
    found: Dict[str, Any] = {}
    seen_dirs = set()
    for task in config.TASKS:
        for split in ("test", "train_val", ADHOC_SPLIT):
            d = config.docking_cache_dir(task, split)
            if d in seen_dirs:
                continue
            seen_dirs.add(d)
            path = os.path.join(d, f"{hashlib.md5(smiles.encode()).hexdigest()[:12]}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue
            for pdb in wanted:
                if pdb not in found and _usable(data.get(pdb)):
                    found[pdb] = data[pdb]
    return found


def _find_receptor(pdb_id: str) -> Optional[Dict[str, Any]]:
    for root in (RECEPTOR_ROOT, WORK_ROOT):
        hits = glob.glob(os.path.join(root, "**", f"protein_{pdb_id}_clean.pdbqt"),
                         recursive=True)
        for pdbqt in hits:
            prep = os.path.dirname(pdbqt)
            bs = os.path.join(prep, f"protein_{pdb_id}_binding_site.json")
            raw = os.path.join(prep, f"protein_{pdb_id}_raw.pdb")
            if os.path.exists(bs) and os.path.exists(raw):
                with open(bs) as f:
                    binding_site = json.load(f)
                return {"pdbqt": pdbqt, "pdb": raw, "binding_site": binding_site}
    return None


def _prepare_receptor(pdb_id: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    from docking_pipeline.execution import UniDockRunner
    from docking_pipeline.preparation import ProteinPreparator

    prep_dir = os.path.join(WORK_ROOT, "receptors")
    os.makedirs(prep_dir, exist_ok=True)
    raw = os.path.join(prep_dir, f"protein_{pdb_id}_raw.pdb")
    clean = os.path.join(prep_dir, f"protein_{pdb_id}_clean.pdb")
    pdbqt = os.path.join(prep_dir, f"protein_{pdb_id}_clean.pdbqt")
    fixed = os.path.join(prep_dir, f"protein_{pdb_id}_clean_fixed.pdb")
    bs_file = os.path.join(prep_dir, f"protein_{pdb_id}_binding_site.json")

    prot = ProteinPreparator()
    try:
        if not os.path.exists(pdbqt):
            if verbose:
                print(f"    [{pdb_id}] receptor not prepared yet — downloading + preparing")
            prot.download_pdb(pdb_id, raw)
            prot.extract_clean_protein(raw, clean)
            prot.pdb_to_pdbqt(clean, pdbqt, fixed)
        if not os.path.exists(bs_file):
            site = UniDockRunner().estimate_binding_site(raw)
            if site is None:
                return None
            with open(bs_file, "w") as f:
                json.dump(site, f, indent=2)
    except Exception as exc:                                      # noqa: BLE001
        print(f"    [{pdb_id}] receptor preparation failed: {exc}")
        return None
    return _find_receptor(pdb_id)


def dock_molecule(smiles: str, pdb_ids: List[str], names: Dict[str, str],
                  gpu: str = "3", num_poses: int = 3, exhaustiveness: int = 8,
                  verbose: bool = True) -> Dict[str, Any]:
    from docking_pipeline.analysis import PLIPAnalyzer
    from docking_pipeline.execution import FormatConverter, UniDockRunner
    from docking_pipeline.preparation import MoleculePreparator
    from rdkit import Chem

    mol_id = hashlib.md5(smiles.encode()).hexdigest()[:12]
    work = os.path.join(WORK_ROOT, mol_id)
    os.makedirs(work, exist_ok=True)

    mol_prep, runner = MoleculePreparator(), UniDockRunner()
    converter, analyzer = FormatConverter(), PLIPAnalyzer(plip_path="plip")

    lig_sdf = os.path.join(work, "ligand.sdf")
    if not os.path.exists(lig_sdf) and not mol_prep.smiles_to_conformer(smiles, lig_sdf):
        print(f"  [dock] 3D embedding failed for this SMILES — no docking evidence")
        return {}
    m = Chem.MolFromMolFile(lig_sdf, removeHs=False)
    if m is None:
        print(f"  [dock] could not read the generated conformer — no docking evidence")
        return {}
    if m.GetNumAtoms() > UNIDOCK_MAX_ATOMS:
        print(f"  [dock] {m.GetNumAtoms()} atoms > {UNIDOCK_MAX_ATOMS}: UniDock's GPU batch "
              f"cannot take it, so this molecule runs without docking evidence")
        return {}

    prev_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    entries: Dict[str, Any] = {}
    try:
        for pdb_id in pdb_ids:
            receptor = _find_receptor(pdb_id) or _prepare_receptor(pdb_id, verbose)
            if receptor is None:
                print(f"    [{pdb_id}] no prepared receptor and preparation failed — skipped")
                continue

            out_dir = os.path.join(work, pdb_id)
            os.makedirs(out_dir, exist_ok=True)
            out_sdf = os.path.join(out_dir, f"{Path(lig_sdf).stem}_out.sdf")
            if not os.path.exists(out_sdf):
                if verbose:
                    print(f"    [{pdb_id}] docking...", flush=True)
                runner.run_batch(
                    receptor_pdbqt=receptor["pdbqt"], ligand_sdfs=[lig_sdf],
                    output_dir=out_dir, center=receptor["binding_site"]["center"],
                    size=receptor["binding_site"]["size"],
                    search_mode="balance", exhaustiveness=exhaustiveness,
                )
            if not os.path.exists(out_sdf):
                print(f"    [{pdb_id}] UniDock produced no pose — skipped")
                continue

            scores = runner.parse_scores(out_sdf)
            if not scores:
                print(f"    [{pdb_id}] no score parsed from the pose file — skipped")
                continue

            all_inter, all_scores = [], []
            for i in range(1, min(num_poses, len(scores)) + 1):
                lig_pdb = os.path.join(out_dir, f"pose{i}_ligand.pdb")
                cplx = os.path.join(out_dir, f"pose{i}_complex.pdb")
                if not converter.sdf_to_pdb(out_sdf, lig_pdb, model_num=i):
                    continue
                if not converter.create_complex(protein_pdb=receptor["pdb"],
                                                ligand_pdb=lig_pdb,
                                                output_complex_pdb=cplx):
                    continue
                inter = analyzer.analyze(complex_pdb=cplx,
                                         output_dir=os.path.join(out_dir, f"pose{i}_plip"))
                if inter:
                    all_inter.append({
                        key: [{"type": it.type, "residue": it.residue,
                               "distance": it.distance, "description": it.description}
                              for it in items]
                        for key, items in inter.items()
                    })
                    all_scores.append(scores[i - 1])

            entries[pdb_id] = {
                "protein_name": names.get(pdb_id, pdb_id),
                "docking_score": min(all_scores) if all_scores else scores[0],
                "all_scores": all_scores or scores[:1],
                "interactions": all_inter,
            }
            if verbose:
                print(f"    [{pdb_id}] {entries[pdb_id]['docking_score']:.2f} kcal/mol, "
                      f"{len(all_inter)} pose(s) with contacts")
    finally:
        if prev_gpu is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_gpu

    return entries


def ensure_evidence(smiles: str, task: str, knowledge: Dict[str, Any],
                    dock: Union[bool, str] = "auto", gpu: str = "3",
                    verbose: bool = True) -> str:
    wanted = _panel_pdbs(knowledge)
    cache_file = ev.cache_path(smiles, task, ADHOC_SPLIT)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    entry: Dict[str, Any] = {"smiles": smiles}
    if os.path.exists(cache_file) and dock != "force":
        with open(cache_file) as f:
            entry.update(json.load(f))

    if dock != "force":
        missing = [p for p in wanted if not _usable(entry.get(p))]
        if missing:
            reused = harvest_cached(smiles, missing)
            if reused and verbose:
                print(f"  [evidence] reusing {len(reused)} cached pose(s): "
                      f"{', '.join(reused)}")
            entry.update(reused)

    missing = [p for p in wanted if not _usable(entry.get(p))]
    if missing and dock:
        if verbose:
            print(f"  [evidence] docking {len(missing)} protein(s): {', '.join(missing)}")
        entry.update(dock_molecule(smiles, missing, wanted, gpu=gpu, verbose=verbose))
    elif missing and verbose:
        print(f"  [evidence] no docking evidence for: {', '.join(missing)} "
              f"(dock=False — these proteins will be judged without a pose)")

    with open(cache_file, "w") as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    have = [p for p in wanted if _usable(entry.get(p))]
    if verbose:
        print(f"  [evidence] {len(have)}/{len(wanted)} panel proteins have a pose "
              f"-> {cache_file}")
    return ADHOC_SPLIT


def predict(smiles: str, task: str,
            knowledge: Union[None, str, Dict[str, Any]] = None, *,
            label: Optional[int] = None,
            ablation: Optional[List[str]] = None,
            dock: Union[bool, str] = "auto",
            gpu: str = "3",
            out: Optional[str] = None,
            append: bool = False,
            verbose: bool = True,
            show: bool = True) -> Dict[str, Any]:
    if task not in config.TASKS:
        raise KeyError(f"unknown task {task!r}. Registered: {sorted(config.TASKS)}")

    k, kpath = resolve_knowledge(task, knowledge)

    if verbose:
        print(f"[{task}]  {smiles}")
        print(f"  knowledge: {os.path.basename(kpath) if kpath else '<inline dict>'}  "
              f"sha256={_sha(kpath, k)}")
        print(f"  model: {config.DEFAULT_MODEL}")
        print(f"  panel: {', '.join(_panel_pdbs(k).values()) or '(none)'}")

    pinned = kpath == config.TASKS[task]["knowledge"]
    saved_dir = config.TOOL_SELECTION_DIR
    if not pinned:
        config.TOOL_SELECTION_DIR = os.path.join(config.CACHE_DIR, "tool_selection_adhoc",
                                                 _sha(kpath, k))
        os.makedirs(config.TOOL_SELECTION_DIR, exist_ok=True)

    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            split = ensure_evidence(smiles, task, k, dock=dock, gpu=gpu, verbose=verbose)
            if verbose:
                sel = physchem_agent.select_tools(task, k)
                print(f"  physchem tools: {len(sel['descriptors'])} descriptors, "
                      f"{len(sel['functional_groups'])} functional groups")
                print("  running agents...", flush=True)

        state = run_one(build_graph(), task, k, smiles, label=label,
                        ablation={a: True for a in (ablation or [])},
                        split=split)
    finally:
        config.TOOL_SELECTION_DIR = saved_dir

    if show:
        show_result(state, k)
    if out:
        save_result(state, k, out, meta={
            "task": task, "split": split,
            "model": config.DEFAULT_MODEL,
            "knowledge_file": os.path.basename(kpath) if kpath else "<inline dict>",
            "knowledge_path": kpath, "knowledge_sha256": _sha(kpath, k),
            "ablation": sorted(ablation or []),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, verbose=verbose, append=append, docking_log=tee.buf.getvalue())
    return state


def predict_many(smiles_list: List[str], task: str, **kw) -> List[Dict[str, Any]]:
    out = []
    for i, smi in enumerate(smiles_list, 1):
        print(f"\n### [{i}/{len(smiles_list)}] {smi}")
        out.append(predict(smi, task, **kw))
    return out


def format_result(state: Dict[str, Any], knowledge: Dict[str, Any]) -> str:
    from agent.agents import relation_evidence

    names = knowledge.get("target_names", ["class 0", "class 1"])
    pred, label = state.get("prediction"), state.get("label")
    out = []

    out.append("=" * 78)
    out.append(f"SMILES: {state.get('smiles')}")
    verdict = names[pred] if pred in (0, 1) and len(names) > 1 else "unparsed"
    head = f"PREDICTION: {pred}  ({verdict})"
    if label is not None:
        head += (f"   label={label} ({names[label] if label in (0, 1) else '?'})  "
                 f"{'O' if pred == label else 'X'}")
    out.append(head)
    out.append("=" * 78)

    pv = state.get("physchem_verdict") or {}
    out.append(f"\n[1] Physicochemical: {pv.get('verdict')} "
               f"(confidence {pv.get('confidence')})")
    for e in pv.get("key_evidence", []) or []:
        out.append(f"      - {e}")
    if pv.get("rationale"):
        out.append(f"    rationale: {pv['rationale']}")

    out.append("\n[2] Proteins")
    for p in state.get("proteins", []):
        pr = "n/a" if p.get("ml_proba") is None else f"{p['ml_proba']:.3f}"
        ds = "n/a" if p.get("docking_score") is None else f"{p['docking_score']:.2f}"
        out.append(f"    {p['protein']:<10} ({p['pdb_id']})  "
                   f"relation={str(p.get('relation')):<20} "
                   f"via={str(p.get('relation_source')):<21} P={pr:<6} dock={ds:<7} "
                   f"used={p.get('docking_used')}")
        if p.get("relation_evidence"):
            out.append(f"        basis: {p['relation_evidence']}")

    out.append("\n[3] Relation agent (sufficiency / escalation)")
    for s in state.get("sufficiency", []) or []:
        flag = "sufficient" if s.get("sufficient") else f"ESCALATED ({s.get('trigger')})"
        out.append(f"    {s['protein']:<10} {flag}  — {s.get('reason')}")

    out.append("\n[4] Full evidence as the prediction agent saw it")
    out.append(relation_evidence.format_evidence(state.get("proteins", [])))

    out.append("\n[5] Prediction agent")
    out.append(state.get("prediction_text") or "(no output)")

    if state.get("errors"):
        out.append("\n[!] errors")
        for e in state["errors"]:
            out.append(f"    {e}")

    return "\n".join(out)


def show_result(state: Dict[str, Any], knowledge: Dict[str, Any]) -> None:
    print("\n" + format_result(state, knowledge))


def _reasoning_record(rec: Dict[str, Any], knowledge: Dict[str, Any]) -> str:
    from agent.scripts.run_task import write_reasoning

    with tempfile.TemporaryDirectory() as tmp:
        write_reasoning([rec], knowledge, tmp)
        with open(os.path.join(tmp, "reasoning.jsonl")) as f:
            line = f.read()

    r = json.loads(line)
    if r.get("label") is None:
        r["correct"] = None
    return json.dumps(r, ensure_ascii=False) + "\n"


def save_result(state: Dict[str, Any], knowledge: Dict[str, Any], out: str,
                meta: Optional[Dict[str, Any]] = None, verbose: bool = True,
                append: bool = False, docking_log: Optional[str] = None) -> str:
    from agent.graph import trace

    rec = trace(state)
    rec["meta"] = meta or {}
    line = json.dumps(rec, ensure_ascii=False) + "\n"

    if out.endswith(".jsonl"):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "a") as f:
            f.write(line)
        target = out
    elif out.endswith(".json"):
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        target = out
    else:
        os.makedirs(out, exist_ok=True)
        mode = "a" if append else "w"

        with open(os.path.join(out, "trace.jsonl"), mode) as f:
            f.write(line)
        with open(os.path.join(out, "reasoning.jsonl"), mode) as f:
            f.write(_reasoning_record(rec, knowledge))
        with open(os.path.join(out, "reasoning.txt"), mode) as f:
            f.write(format_result(state, knowledge) + "\n\n")
        with open(os.path.join(out, "run_meta.json"), "w") as f:
            json.dump(meta or {}, f, indent=2, ensure_ascii=False)
        if docking_log:
            with open(os.path.join(out, "docking.log"), mode) as f:
                f.write(docking_log)
        target = out

    if verbose:
        print(f"  [saved] {target}")
    return target


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="agent.scripts.predict_one — run the framework on ONE molecule you hand it.")
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--task", required=True, choices=sorted(config.TASKS))
    ap.add_argument("--knowledge", default=None,
                    help="filename under knowledge/<group>/ or a full path")
    ap.add_argument("--label", type=int, default=None)
    ap.add_argument("--ablation", nargs="*", default=[])
    ap.add_argument("--no-dock", action="store_true",
                    help="reuse cached poses only; never launch UniDock")
    ap.add_argument("--force-dock", action="store_true")
    ap.add_argument("--gpu", default="3")
    ap.add_argument("--out", default=None,
                    help="save the result: a .jsonl (appended), a .json (overwritten), "
                         "or a directory (the five files run_task leaves)")
    ap.add_argument("--append", action="store_true",
                    help="directory --out: keep every call instead of replacing the last")
    args = ap.parse_args()

    dock = "force" if args.force_dock else (False if args.no_dock else "auto")
    predict(args.smiles, args.task, args.knowledge, label=args.label,
            ablation=args.ablation, dock=dock, gpu=args.gpu, out=args.out,
            append=args.append)


if __name__ == "__main__":
    main()
