import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from agent import config
from agent.registry.proteins import _norm, lookup

CACHE_DIR = os.path.join(config.CACHE_DIR, "protein_synonyms")

_UNIPROT = "https://rest.uniprot.org/uniprotkb"

_MAX_WORDS = 4
_MAX_SYNONYMS = 6


def _get(url: str, timeout: int = 60) -> Optional[Dict[str, Any]]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "agent-synonyms/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:                                          # noqa: BLE001
        return None


def _names_of(accession: str) -> List[str]:
    doc = _get(f"{_UNIPROT}/{accession}.json")
    if doc is None:
        return []

    genes, short, full = [], [], []
    for gene in doc.get("genes") or []:
        genes.append(((gene.get("geneName") or {}).get("value")))
        genes.extend(s.get("value") for s in (gene.get("synonyms") or []))
    desc = doc.get("proteinDescription") or {}
    for block in [desc.get("recommendedName") or {}, *(desc.get("alternativeNames") or [])]:
        short.extend(s.get("value") for s in (block.get("shortNames") or []))
        full.append(((block.get("fullName") or {}).get("value")))
    return [n for n in genes + short + full if n]


def _matches(name: str, candidates: List[str]) -> bool:
    n = _norm(name)
    if len(n) < 3:
        return False
    return any(n in _norm(c) or _norm(c) in n for c in candidates if len(c) > 2)


def _pick_chain(protein: str, accessions: List[str]) -> Optional[str]:
    entry = lookup(protein) or {}
    ours = [protein, *entry.get("aliases", [])]

    per_chain = {acc: _names_of(acc) for acc in accessions[:5]}
    for acc, names in per_chain.items():
        if any(_matches(o, names) for o in ours):
            return acc
    if len(accessions) == 1:
        return accessions[0]
    return None


def _searchable(names: List[str], protein: str) -> List[str]:
    seen, out = set(), []
    for n in [protein, *names]:
        n = (n or "").strip()
        if not n or len(n.split()) > _MAX_WORDS:
            continue
        key = n.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
        if len(out) >= _MAX_SYNONYMS:
            break
    return out


def synonyms_for(protein: str, pdb_id: Optional[str] = None) -> List[str]:
    entry = lookup(protein)
    canonical = entry["canonical"] if entry else protein
    fallback = _searchable(list((entry or {}).get("aliases", [])), canonical)

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{canonical}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("search_names") or fallback
    if not pdb_id:
        return fallback

    from agent.tools.reference_discovery.chembl_target import uniprot_for_pdb

    accessions = uniprot_for_pdb(pdb_id)
    chosen = _pick_chain(canonical, accessions) if accessions else None
    uniprot_names = _names_of(chosen) if chosen else []
    search = _searchable(uniprot_names + list((entry or {}).get("aliases", [])), canonical)

    record = {"protein": canonical, "pdb_id": pdb_id, "accessions": accessions,
              "chosen_accession": chosen, "uniprot_names": uniprot_names,
              "search_names": search}
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return search or fallback


def mentions(protein: str, text: str, pdb_id: Optional[str] = None) -> bool:
    t = _norm(text)
    return any(_norm(n) in t for n in synonyms_for(protein, pdb_id) if len(n) > 2)
