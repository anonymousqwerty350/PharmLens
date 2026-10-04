from typing import Dict, List, Optional
from rdkit import Chem, RDLogger
from agent.tools import rdkit_tools

RDLogger.DisableLog("rdApp.*")


def catalog() -> List[Dict[str, str]]:
    return rdkit_tools.load_catalogs()["functional_groups"]


def _canon(name) -> Optional[str]:
    n = str(name).strip()
    if n in rdkit_tools.FG_FUNCS:
        return n
    if f"fr_{n}" in rdkit_tools.FG_FUNCS:
        return f"fr_{n}"
    return None


def normalize(groups) -> List[str]:
    if not isinstance(groups, list):
        return []
    out = []
    for g in groups:
        c = _canon(g)
        if c and c not in out:
            out.append(c)
    return out


def has_all_groups(smiles: str, groups: List[str]) -> Optional[bool]:
    valid = normalize(groups)
    if not valid:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return all(rdkit_tools.FG_FUNCS[g](mol) > 0 for g in valid)


def groups_recall(groups: List[str], smiles_list: List[str]) -> Optional[float]:
    valid = normalize(groups)
    checkable = [s for s in smiles_list if Chem.MolFromSmiles(str(s)) is not None]
    if not valid or not checkable:
        return None
    hits = sum(1 for s in checkable if has_all_groups(s, valid))
    return round(hits / len(checkable), 2)
