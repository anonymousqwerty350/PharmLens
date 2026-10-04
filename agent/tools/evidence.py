import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from agent import config

INTERACTION_KEYS = [
    "hydrogen_bonds", "hydrophobic", "salt_bridges", "pi_stacking",
    "pi_cation", "halogen_bonds", "water_bridges", "metal_complexes",
]
_LABELS = {
    "hydrogen_bonds": "H-bond", "hydrophobic": "hydrophobic",
    "salt_bridges": "salt bridge", "pi_stacking": "pi-stacking",
    "pi_cation": "pi-cation", "halogen_bonds": "halogen bond",
    "water_bridges": "water bridge", "metal_complexes": "metal complex",
}

STATS_DIR = os.path.join(config.CACHE_DIR, "score_stats")


def cache_path(smiles: str, task: str, split: str = "test") -> str:
    mol_id = hashlib.md5(smiles.encode()).hexdigest()[:12]
    return os.path.join(config.docking_cache_dir(task, split), f"{mol_id}.json")


def load_cache(smiles: str, task: str, split: str = "test") -> Dict[str, Any]:
    path = cache_path(smiles, task, split)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "smiles" and isinstance(v, dict)}


def build_score_stats(task: str) -> Dict[str, Any]:
    cache_dir = config.docking_cache_dir(task, "train_val")
    scores: Dict[str, List[float]] = {}
    names: Dict[str, str] = {}

    for fn in os.listdir(cache_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(cache_dir, fn)) as f:
            data = json.load(f)
        for pdb, entry in data.items():
            if pdb == "smiles" or not isinstance(entry, dict):
                continue
            s = entry.get("docking_score")
            if s is None or s > 0:
                continue
            scores.setdefault(pdb, []).append(float(s))
            names.setdefault(pdb, entry.get("protein_name", pdb))

    proteins = {}
    for pdb, vals in scores.items():
        a = np.array(vals)
        proteins[pdb] = {
            "protein_name": names[pdb], "n": len(vals),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "p50": round(float(np.percentile(a, 50)), 3),
            "p75": round(float(np.percentile(a, 75)), 3),
            "mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
        }

    stats = {"task": task, "split": "train_val", "proteins": proteins}
    os.makedirs(STATS_DIR, exist_ok=True)
    with open(os.path.join(STATS_DIR, f"{task}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def load_score_stats(task: str) -> Dict[str, Any]:
    path = os.path.join(STATS_DIR, f"{task}.json")
    if not os.path.exists(path):
        return build_score_stats(task)["proteins"]
    with open(path) as f:
        return json.load(f)["proteins"]


def quartile_label(score: Optional[float], stats: Dict[str, Any]) -> Optional[str]:
    if score is None:
        return None
    if score > 0:
        return ("NO BINDING (score > 0) — no favorable pose exists in this pocket, "
                "so the molecule does not bind this protein at all")
    p25, p50, p75 = stats.get("p25"), stats.get("p50"), stats.get("p75")
    if p25 is None:
        return None
    if score < p25:
        return "Q1 — strongest 25% of train set"
    if score < p50:
        return "Q2 — 25th-50th percentile"
    if score < p75:
        return "Q3 — 50th-75th percentile"
    return "Q4 — weakest 25% of train set"


def format_score_stats(stats: Optional[Dict[str, Any]]) -> Optional[str]:
    if not stats or stats.get("p25") is None:
        return None
    return (f"n={stats['n']}, p25/p50/p75 = {stats['p25']:.2f} / {stats['p50']:.2f} / "
            f"{stats['p75']:.2f} kcal/mol, range {stats['min']:.2f} (strongest) to "
            f"{stats['max']:.2f} (weakest)")


def extract_plip(entry: Dict[str, Any]) -> Dict[str, List[str]]:
    poses = entry.get("interactions") or []
    if not poses:
        return {}
    pose = poses[0]
    out: Dict[str, List[str]] = {}
    for key in INTERACTION_KEYS:
        items = pose.get(key) or []
        if not items:
            continue
        contacts = []
        for it in items:
            res = f"{it.get('description', '')}{it.get('residue', '?')}"
            dist = it.get("distance") or 0.0
            contacts.append(f"{res}({dist:.2f}A)" if dist > 0 else res)
        out[key] = contacts
    return out


def format_plip(plip: Optional[Dict[str, List[str]]]) -> str:
    if not plip:
        return "    (no interactions detected)"
    lines = []
    for key, contacts in plip.items():
        lines.append(f"    {_LABELS.get(key, key)} ({len(contacts)}): {', '.join(contacts)}")
    return "\n".join(lines)
