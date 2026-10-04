import os
import re
from typing import Any, Dict, List, Optional

from agent import config

ML_LABELS = {
    "transporter": "substrate",
    "inhibitor":   "potent_inhibitor",
    "antagonist":  "antagonist",
    "cyp_substrate": "substrate",
    "agonist":       "agonist",
    "unknown":       "binder",
}

RELATION_WORDS = {
    "transporter":   "substrate",
    "inhibitor":     "inhibitor",
    "antagonist":    "antagonist",
    "cyp_substrate": "substrate",
    "agonist":       "agonist",
}


def panel_proteins() -> List[str]:
    import json

    out = []
    for cfg in config.TASKS.values():
        with open(cfg["knowledge"]) as f:
            for p in resolve_proteins(json.load(f)):
                hit = lookup(p["protein"])
                if hit and hit["category"] not in ("unknown", "cyp_substrate") \
                        and hit["canonical"] not in out:
                    out.append(hit["canonical"])
    return sorted(out)


def relation_for(protein: str) -> str:
    entry = lookup(protein)
    return RELATION_WORDS.get(entry["category"] if entry else None, "substrate")

_ENTRIES: Dict[str, tuple] = {
    "Pgp":     ("transporter", ["P-gp", "P-glycoprotein", "ABCB1", "MDR1"]),
    "BCRP":    ("transporter", ["ABCG2"]),
    "MRP1":    ("transporter", ["ABCC1"]),
    "MRP2":    ("inhibitor", ["ABCC2"]),
    "hERG":    ("inhibitor", ["KCNH2", "Kv11.1"]),
    "Nav1.5":  ("inhibitor", ["SCN5A"]),
    "VEGFR2":  ("inhibitor", ["KDR", "FLK1"]),
    "COX2":    ("inhibitor", ["COX-2", "PTGS2"]),
    "ABL1":    ("inhibitor", ["BCR-ABL"]),
    "EGFR":    ("inhibitor", ["ERBB1", "HER1"]),
    "DHFR":    ("inhibitor", []),
    "ACE":     ("inhibitor", []),
    "CYP17A1": ("inhibitor", ["CYP17"]),
    "CYP19A1": ("inhibitor", ["CYP19", "aromatase"]),
    "MEK1":    ("inhibitor", ["MAP2K1"]),
    "BRAF":    ("inhibitor", ["B-Raf"]),
    "mTOR":    ("inhibitor", ["MTOR", "FRAP1"]),
    "PSMB5":   ("inhibitor", ["proteasome", "20S proteasome"]),
    "DHODH":   ("inhibitor", ["dihydroorotate dehydrogenase"]),
    "ESR1":    ("antagonist", ["estrogen receptor", "ER-alpha", "ERa"]),
    "DRD2":    ("antagonist", ["dopamine D2"]),
    "FXR":     ("antagonist", ["NR1H4"]),
    "OPRM1":   ("agonist", ["mu-opioid receptor"]),
    "PXR":     ("agonist", ["NR1I2"]),
}

_NO_MODEL: Dict[str, tuple] = {
    "TYMS":    ("inhibitor",   ["thymidylate synthase"]),
    "VKORC1":  ("inhibitor",   []),
    "LAT1":    ("transporter", ["SLC7A5"]),
    "PEPT1":   ("transporter", ["SLC15A1", "PepT1", "peptide transporter 1"]),
    "MATE1":   ("transporter", ["SLC47A1"]),
    "OATP1B1": ("transporter", ["SLCO1B1"]),
    "BSEP":    ("inhibitor",   ["ABCB11"]),
    "ENT1":    ("transporter", ["SLC29A1"]),
    "OAT1":    ("transporter", ["SLC22A6"]),
    "CYP1A2":  ("cyp_substrate", []),
    "CYP2C9":  ("cyp_substrate", []),
    "CYP2C19": ("cyp_substrate", []),
    "CYP2D6":  ("cyp_substrate", []),
    "CYP2E1":  ("cyp_substrate", []),
    "CYP3A4":  ("cyp_substrate", []),
    "SRD5A2":  ("inhibitor",   []),
    "KCNQ4":   ("inhibitor",   ["Kv7.4"]),
    "TRPA1":   ("agonist",     []),
    "NKCC1":   ("inhibitor",   ["SLC12A2"]),
    "Prestin": ("inhibitor",   ["SLC26A5"]),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", str(name).lower())


def _split_names(raw: str) -> List[str]:
    parts = re.split(r"[()/,]", str(raw))
    return [p.strip() for p in parts if p.strip()]


def _build_alias_index() -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for table, has_model in ((_ENTRIES, True), (_NO_MODEL, False)):
        for canonical, (category, aliases) in table.items():
            bundle = None
            if has_model:
                p = os.path.join(config.MODEL_DIRS[category], f"{canonical}.pkl")
                bundle = p if os.path.exists(p) else None
            entry = {
                "canonical": canonical,
                "category": category,
                "ml_bundle": bundle,
                "ml_label": ML_LABELS[category],
                "aliases": list(aliases),
            }
            for alias in [canonical, *aliases]:
                index[_norm(alias)] = entry
    return index


_ALIAS_INDEX = _build_alias_index()


def lookup(name: str) -> Optional[Dict[str, Any]]:
    for token in _split_names(name):
        hit = _ALIAS_INDEX.get(_norm(token))
        if hit:
            return hit
    return None


def resolve_proteins(knowledge: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = knowledge.get("relevant_proteins") or knowledge.get("proteins") or {}
    resolved: List[Dict[str, Any]] = []

    for key, info in raw.items():
        info = info or {}
        pdb_id = info.get("pdb_id") or (key if re.fullmatch(r"[0-9][A-Za-z0-9]{3}", key) else None)
        display = key if pdb_id != key else (info.get("name") or key)

        entry = lookup(display) or lookup(info.get("name", "")) or lookup(info.get("gene", ""))
        if entry is None:
            resolved.append({
                "protein": display, "pdb_id": pdb_id, "category": "unknown",
                "ml_bundle": None, "ml_label": None,
                "stage": info.get("stage") or info.get("role", ""),
                "logic": info.get("logic", ""),
            })
            continue

        resolved.append({
            "protein": entry["canonical"] if pdb_id == key else display,
            "pdb_id": pdb_id,
            "category": entry["category"],
            "ml_bundle": entry["ml_bundle"],
            "ml_label": entry["ml_label"],
            "stage": info.get("stage") or info.get("role", ""),
            "logic": info.get("logic", ""),
        })
    return resolved
