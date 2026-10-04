import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docking_pipeline.analysis import PLIPAnalyzer          # noqa: E402
from docking_pipeline.execution import FormatConverter, UniDockRunner  # noqa: E402
from docking_pipeline.preparation import MoleculePreparator  # noqa: E402
from agent import config                                # noqa: E402
from agent.registry.proteins import resolve_proteins    # noqa: E402
from agent.tools import reference_ligands as rl         # noqa: E402

DOCK_ROOT = config.DOCKING_RESULTS
WORK_ROOT = os.path.join(config.CACHE_DIR, "reference_work")

UNIDOCK_MAX_ATOMS = 100


def find_receptor(pdb_id: str) -> Optional[Dict[str, Any]]:
    hits = glob.glob(os.path.join(DOCK_ROOT, "**", f"protein_{pdb_id}_clean.pdbqt"),
                     recursive=True)
    if not hits:
        return None
    prep = os.path.dirname(hits[0])
    bs_file = os.path.join(prep, f"protein_{pdb_id}_binding_site.json")
    raw_pdb = os.path.join(prep, f"protein_{pdb_id}_raw.pdb")
    if not (os.path.exists(bs_file) and os.path.exists(raw_pdb)):
        return None
    with open(bs_file) as f:
        binding_site = json.load(f)
    return {"pdbqt": hits[0], "pdb": raw_pdb, "binding_site": binding_site}


def _ligand_key(smiles: str) -> str:
    from rdkit import Chem as RDChem

    mol = RDChem.MolFromSmiles(smiles)
    canonical = RDChem.MolToSmiles(mol) if mol is not None else smiles
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def dock_set(pdb_id: str, receptor: Dict[str, Any], smiles_list: List[str],
             tag: str) -> List[Dict[str, Any]]:
    work = os.path.join(WORK_ROOT, pdb_id, tag)
    lig_dir = os.path.join(work, "ligands")
    out_dir = os.path.join(work, "docked")
    pose_dir = os.path.join(work, "poses")
    for d in (lig_dir, out_dir, pose_dir):
        os.makedirs(d, exist_ok=True)

    mol_prep = MoleculePreparator()
    runner = UniDockRunner()
    converter = FormatConverter()
    analyzer = PLIPAnalyzer(plip_path="plip")

    from rdkit import Chem as RDChem

    sdf_of: Dict[str, str] = {}
    for smi in smiles_list:
        sdf = os.path.join(lig_dir, f"lig_{_ligand_key(smi)}.sdf")
        if not os.path.exists(sdf) and not mol_prep.smiles_to_conformer(smi, sdf):
            print(f"    [{tag}] SDF failed: {smi[:50]}")
            continue
        m = RDChem.MolFromMolFile(sdf, removeHs=False)
        if m is None or m.GetNumAtoms() > UNIDOCK_MAX_ATOMS:
            print(f"    [{tag}] atom count exceeded / parse failed, excluded: {smi[:50]}")
            continue
        sdf_of[smi] = sdf

    if not sdf_of:
        return []

    todo = [s for s in sdf_of.values()
            if not os.path.exists(os.path.join(out_dir, f"{Path(s).stem}_out.sdf"))]
    if todo:
        print(f"    [{tag}] unidock {len(todo)} ligands...")
        runner.run_batch(
            receptor_pdbqt=receptor["pdbqt"], ligand_sdfs=todo, output_dir=out_dir,
            center=receptor["binding_site"]["center"],
            size=receptor["binding_site"]["size"],
            search_mode="balance", exhaustiveness=8,
        )

    results = []
    no_pose, no_score = [], []
    for smi, sdf in sdf_of.items():
        out_sdf = os.path.join(out_dir, f"{Path(sdf).stem}_out.sdf")
        if not os.path.exists(out_sdf):
            no_pose.append(smi)
            continue
        scores = runner.parse_scores(out_sdf)
        if not scores:
            no_score.append(smi)
            continue

        stem = Path(sdf).stem
        lig_pdb = os.path.join(pose_dir, f"{stem}_pose1.pdb")
        cplx_pdb = os.path.join(pose_dir, f"{stem}_complex1.pdb")
        contacts: Dict[str, List[str]] = {}
        if converter.sdf_to_pdb(out_sdf, lig_pdb, model_num=1) and converter.create_complex(
                protein_pdb=receptor["pdb"], ligand_pdb=lig_pdb, output_complex_pdb=cplx_pdb):
            inter = analyzer.analyze(complex_pdb=cplx_pdb,
                                     output_dir=os.path.join(pose_dir, f"{stem}_plip"))
            for key, items in (inter or {}).items():
                for it in items:
                    contacts.setdefault(key, []).append(f"{it.description}{it.residue}")

        results.append({"smiles": smi, "score": float(min(scores)), "contacts": contacts})

    lost = len(smiles_list) - len(results)
    if lost:
        print(f"    [{tag}] ⚠ only {len(results)} of {len(smiles_list)} docked "
              f"(preparation failed {len(smiles_list) - len(sdf_of)}, "
              f"no pose {len(no_pose)}, no score {len(no_score)})")
        for smi in no_pose[:5]:
            print(f"        └ no pose: {smi[:60]}")
    return results


def summarise(entries: List[Dict[str, Any]], attempted: int = 0) -> Dict[str, Any]:
    if not entries:
        return {"n": 0, "n_attempted": attempted}
    scores = np.array([e["score"] for e in entries if e["score"] < 0])
    n = len(entries)

    freq: Dict[str, Counter] = {}
    for e in entries:
        for key, residues in e["contacts"].items():
            freq.setdefault(key, Counter()).update(set(residues))

    top = {}
    for key, counter in freq.items():
        ranked = [(res, round(cnt / n, 2)) for res, cnt in counter.most_common(8)]
        top[key] = [f"{res}({int(p*100)}%)" for res, p in ranked if p >= 0.2]

    out = {"n": n, "n_attempted": attempted, "top_contacts": {k: v for k, v in top.items() if v}}
    if len(scores):
        out.update({
            "score_p25": round(float(np.percentile(scores, 25)), 2),
            "score_p50": round(float(np.percentile(scores, 50)), 2),
            "score_p75": round(float(np.percentile(scores, 75)), 2),
            "score_mean": round(float(scores.mean()), 2),
        })
    return out


def build(task: str, force: bool = False):
    knowledge = json.load(open(config.TASKS[task]["knowledge"]))
    os.makedirs(config.REFERENCE_DIR, exist_ok=True)

    for p in resolve_proteins(knowledge):
        name, pdb_id = p["protein"], p["pdb_id"]

        if p["category"] == "unknown":
            print(f"[{task}] {name} ({pdb_id}) — not in registry (excluded from panel) → skip")
            continue
        if p["category"] == "cyp_substrate":
            print(f"[{task}] {name} ({pdb_id}) — decided by CypReact → no profile needed")
            continue

        out_path = os.path.join(config.REFERENCE_DIR, f"{pdb_id}.json")
        if os.path.exists(out_path) and not force:
            print(f"[{task}] {name} ({pdb_id}) — using cache")
            continue

        print(f"\n[{task}] {name} ({pdb_id}) building reference profile")
        receptor = find_receptor(pdb_id)
        if receptor is None:
            print(f"  no receptor → skip")
            continue

        ligands = rl.collect(name)
        if not ligands["actives"]:
            print(f"  no reference actives → skip (reference comparison disabled for this protein)")
            continue

        inactives = ligands["inactives"]

        print(f"  actives={len(ligands['actives'])} inactives={len(inactives)} "
              f"source={ligands['source']}")

        active_res = dock_set(pdb_id, receptor, ligands["actives"], "actives")
        inactive_res = dock_set(pdb_id, receptor, inactives, "inactives")

        profile = {
            "protein": name,
            "pdb_id": pdb_id,
            "relation": ligands["relation"],
            "active_source": ligands["source"],
            "inactive_source": ligands["source"] if inactives else None,
            "actives_reference": summarise(active_res, len(ligands["actives"])),
            "inactives_reference": summarise(inactive_res, len(inactives)),
        }
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        act = profile["actives_reference"]
        print(f"  saved: {out_path}  actives n={act['n']}/{act['n_attempted']} "
              f"p50={act.get('score_p50')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["cardiotoxicity", "bbbp"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="delete the docking work directory after the build")
    args = ap.parse_args()

    for t in args.tasks:
        build(t, force=args.force)

    if args.cleanup and os.path.isdir(WORK_ROOT):
        shutil.rmtree(WORK_ROOT)
        print(f"\nremoving work directory: {WORK_ROOT}")
