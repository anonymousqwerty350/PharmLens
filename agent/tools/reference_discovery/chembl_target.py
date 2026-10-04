import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from agent import config

_RCSB = "https://data.rcsb.org/rest/v1/core/polymer_entity"
_CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

CACHE_DIR = os.path.join(config.CACHE_DIR, "chembl_targets")


def _get(url: str, timeout: int = 60, retries: int = 3) -> Optional[Dict[str, Any]]:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:                                       # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def uniprot_for_pdb(pdb_id: str) -> List[str]:
    entry = _get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
    entity_ids = ((entry or {}).get("rcsb_entry_container_identifiers") or {}) \
        .get("polymer_entity_ids") or ["1"]

    out = []
    for eid in entity_ids[:6]:
        doc = _get(f"{_RCSB}/{pdb_id}/{eid}")
        ids = ((doc or {}).get("rcsb_polymer_entity_container_identifiers") or {}) \
            .get("reference_sequence_identifiers") or []
        for i in ids:
            acc = i.get("database_accession")
            if i.get("database_name") == "UniProt" and acc and acc not in out:
                out.append(acc)
    return out


_FUSION_PARTNERS = {
    "P00720",
    "P0ABE7",
    "P42212",
    "P0AEX9",
    "P0AA25",
    "P24297",
    "P00698",
}


def chembl_target_for(accession: str) -> Optional[str]:
    q = urllib.parse.urlencode({"target_components__accession": accession,
                                "format": "json", "limit": 20})
    doc = _get(f"{_CHEMBL}/target?{q}")
    targets = (doc or {}).get("targets") or []
    for t in targets:
        if t.get("target_type") == "SINGLE PROTEIN":
            return t.get("target_chembl_id")
    return None


def _activity_count(target: str) -> int:
    q = urllib.parse.urlencode({"target_chembl_id": target, "pchembl_value__isnull": "false",
                                "format": "json", "limit": 1})
    doc = _get(f"{_CHEMBL}/activity?{q}")
    return int(((doc or {}).get("page_meta") or {}).get("total_count") or 0)


def resolve_target(protein: str, pdb_id: str) -> Optional[str]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{protein}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("target_chembl_id")

    best, best_acc, best_n = None, None, 0
    candidates = []
    for accession in uniprot_for_pdb(pdb_id):
        if accession in _FUSION_PARTNERS:
            candidates.append({"accession": accession, "target": None,
                               "n_activities": 0, "skipped": "crystallisation fusion partner"})
            continue
        target = chembl_target_for(accession)
        if target is None:
            candidates.append({"accession": accession, "target": None, "n_activities": 0})
            continue
        n = _activity_count(target)
        candidates.append({"accession": accession, "target": target, "n_activities": n})
        if n > best_n:
            best, best_acc, best_n = target, accession, n

    with open(path, "w") as f:
        json.dump({"protein": protein, "pdb_id": pdb_id, "accession": best_acc,
                   "target_chembl_id": best, "n_activities": best_n,
                   "candidates": candidates}, f, indent=2)
    return best
