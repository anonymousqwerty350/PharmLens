import csv
import os
from typing import Any, Dict, List, Optional

from agent import config
from agent.registry.proteins import lookup
from agent.tools.reference_discovery import selection
from agent.tools.reference_discovery.literature import _canonical
from agent.tools.reference_discovery.selection import representatives

_FILENAMES = {
    "transporter": "{p}_substrate.csv",
    "inhibitor":   "{p}.csv",
    "antagonist":  "antagonist_{p}.csv",
    "agonist":     "agonist_{p}.csv",
}

_SMILES_COLS = ("smiles", "canonical_smiles")
_NAME_COLS = ("preferred_name", "pid", "chembl_ids", "name")
_LABEL_COLS = ("label", "y", "active")

_MAX_ATOMS = selection.MAX_ATOMS
_too_large = selection.too_large


def _column(fieldnames: List[str], candidates: tuple) -> Optional[str]:
    lowered = {(f or "").strip().lower(): f for f in (fieldnames or [])}
    for want in candidates:
        if want in lowered:
            return lowered[want]
    return None


def _dataset_path(canonical: str, category: str) -> Optional[str]:
    directory = config.DATASET_DIRS.get(category)
    if not directory:
        return None
    path = os.path.join(directory, _FILENAMES[category].format(p=canonical))
    return path if os.path.exists(path) else None


def is_allowed(protein: str) -> bool:
    return lookup(protein) is not None


def _read_labelled(protein: str, wanted_label: str) -> tuple:
    entry = lookup(protein)
    if entry is None or not is_allowed(protein):
        return [], {}, None
    path = _dataset_path(entry["canonical"], entry["category"])
    if not path:
        return [], {}, None

    pool: List[str] = []
    names: Dict[str, str] = {}
    seen: set = set()
    oversize = 0
    with open(path) as f:
        reader = csv.DictReader(f)
        smiles_col = _column(reader.fieldnames, _SMILES_COLS)
        label_col = _column(reader.fieldnames, _LABEL_COLS)
        name_col = _column(reader.fieldnames, _NAME_COLS)
        if not smiles_col or not label_col:
            return [], {}, path
        for row in reader:
            if str(row.get(label_col, "")).strip() not in (wanted_label, f"{wanted_label}.0"):
                continue
            smi = _canonical(row.get(smiles_col, ""))
            if not smi or smi in seen:
                continue
            seen.add(smi)
            if _too_large(smi):
                oversize += 1
                continue
            pool.append(smi)
            label = (row.get(name_col) or "").strip() if name_col else ""
            if label:
                names[smi] = label
    if oversize:
        print(f"    [{protein}] excluded {oversize} of label={wanted_label} "
              f"(atom count incl. hydrogens > {_MAX_ATOMS}, beyond UniDock)")
    return pool, names, path


def dataset_actives(protein: str, limit: int = 15) -> Dict[str, Any]:
    pool, names, path = _read_labelled(protein, "1")
    if not pool:
        return {"actives": [], "resolved": [], "dataset": path, "n_pool": 0}

    picked = representatives(pool, limit)
    actives = [pool[i] for i in picked]
    resolved = [{"name": names.get(s, ""), "smiles": s, "class": "",
                 "source": "ml_dataset", "dataset": os.path.basename(path)}
                for s in actives]
    return {"actives": actives, "resolved": resolved, "dataset": path, "n_pool": len(pool)}


def dataset_inactives(protein: str, limit: int = 15,
                      actives: Optional[List[str]] = None) -> Dict[str, Any]:
    pool, names, path = _read_labelled(protein, "0")
    if not pool:
        return {"inactives": [], "resolved": [], "dataset": path, "n_pool": 0}

    matcher = selection.property_matcher(actives or [])
    picked = representatives(pool, limit, priority=lambda i: matcher(pool[i]))
    inactives = [pool[i] for i in picked]
    resolved = [{"name": names.get(s, ""), "smiles": s,
                 "source": "ml_dataset", "dataset": os.path.basename(path)}
                for s in inactives]
    return {"inactives": inactives, "resolved": resolved,
            "dataset": path, "n_pool": len(pool)}
