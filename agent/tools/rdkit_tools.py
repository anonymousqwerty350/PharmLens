import json
import os
from typing import Any, Dict, List

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Fragments, Lipinski

from agent import config

RDLogger.DisableLog("rdApp.*")

FG_FUNCS = {name: fn for name, fn in vars(Fragments).items()
            if name.startswith("fr_") and callable(fn)}
DESC_FUNCS = dict(Descriptors.descList)

FG_CATALOG_PATH = os.path.join(config.CATALOG_DIR, "functional_groups.json")
DESC_CATALOG_PATH = os.path.join(config.CATALOG_DIR, "descriptors.json")


def _docstring(fn) -> str:
    doc = (fn.__doc__ or "").strip().replace("\n", " ")
    return " ".join(doc.split())[:200] or "(no description)"


def build_catalogs() -> Dict[str, int]:
    os.makedirs(config.CATALOG_DIR, exist_ok=True)
    fg = [{"name": n, "description": _docstring(f)} for n, f in sorted(FG_FUNCS.items())]
    desc = [{"name": n, "description": _docstring(f)} for n, f in sorted(DESC_FUNCS.items())]
    with open(FG_CATALOG_PATH, "w") as f:
        json.dump(fg, f, indent=2)
    with open(DESC_CATALOG_PATH, "w") as f:
        json.dump(desc, f, indent=2)
    return {"functional_groups": len(fg), "descriptors": len(desc)}


def load_catalogs() -> Dict[str, List[Dict[str, str]]]:
    if not os.path.exists(FG_CATALOG_PATH):
        build_catalogs()
    with open(FG_CATALOG_PATH) as f:
        fg = json.load(f)
    with open(DESC_CATALOG_PATH) as f:
        desc = json.load(f)
    return {"functional_groups": fg, "descriptors": desc}


def lipinski(mol) -> Dict[str, Any]:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    return {"MolWt": round(mw, 1), "MolLogP": round(logp, 2),
            "HBD": hbd, "HBA": hba, "violations": violations}


def compute(smiles: str, fg_names: List[str], desc_names: List[str]) -> Dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}

    out: Dict[str, Any] = {"valid": True, "lipinski": lipinski(mol),
                           "descriptors": {}, "functional_groups": {}, "unknown_tools": []}

    for name in desc_names:
        fn = DESC_FUNCS.get(name)
        if fn is None:
            out["unknown_tools"].append(name)
            continue
        try:
            v = fn(mol)
            out["descriptors"][name] = round(float(v), 3)
        except Exception:
            out["descriptors"][name] = None

    for name in fg_names:
        fn = FG_FUNCS.get(name)
        if fn is None:
            out["unknown_tools"].append(name)
            continue
        try:
            out["functional_groups"][name] = int(fn(mol))
        except Exception:
            out["functional_groups"][name] = None

    return out


def format_for_prompt(values: Dict[str, Any]) -> str:
    if not values.get("valid"):
        return "  (invalid SMILES — no physicochemical evidence)"

    lip = values["lipinski"]
    lines = [
        "Lipinski Rule of 5: "
        f"MolWt={lip['MolWt']}, MolLogP={lip['MolLogP']}, "
        f"HBD={lip['HBD']}, HBA={lip['HBA']}, violations={lip['violations']}"
    ]

    desc = {k: v for k, v in values["descriptors"].items() if v is not None}
    if desc:
        lines.append("RDKit descriptors: "
                     + ", ".join(f"{k}={v}" for k, v in desc.items()))

    fg = values["functional_groups"]
    present = {k: v for k, v in fg.items() if v}
    absent = [k for k, v in fg.items() if v == 0]
    if present:
        lines.append("Functional groups present (count): "
                     + ", ".join(f"{k}={v}" for k, v in present.items()))
    if absent:
        lines.append("Functional groups absent (count=0): " + ", ".join(absent))

    return "\n".join(f"  {ln}" for ln in lines)
