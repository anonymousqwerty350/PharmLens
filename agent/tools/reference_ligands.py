import csv
import json
import os
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from agent import config

RDLogger.DisableLog("rdApp.*")

K = 15

DISCOVERY_CACHE_DIR = os.path.join(config.CACHE_DIR, "reference_ligands")


def _skeleton(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:                                          # noqa: BLE001
        return None
    return key.split("-")[0] or None


def _fp(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    return None if mol is None else AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)


_PROTEIN_TASKS: Optional[Dict[str, List[str]]] = None
_TASK_SKELETONS: Dict[str, set] = {}


def tasks_using(protein: str) -> List[str]:
    global _PROTEIN_TASKS
    if _PROTEIN_TASKS is None:
        from agent.registry.proteins import lookup, resolve_proteins

        idx: Dict[str, List[str]] = {}
        for task, t in config.TASKS.items():
            with open(t["knowledge"]) as f:
                for p in resolve_proteins(json.load(f)):
                    hit = lookup(p["protein"])
                    name = hit["canonical"] if hit else p["protein"]
                    idx.setdefault(name, []).append(task)
        _PROTEIN_TASKS = idx
    return sorted(set(_PROTEIN_TASKS.get(protein, [])))


def _test_skeletons(task: str) -> set:
    if task not in _TASK_SKELETONS:
        t = config.TASKS[task]
        path = os.path.join(t["dataset_dir"], "test.csv")
        out = set()
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    s = _skeleton(row.get(t["smiles_col"], ""))
                    if s:
                        out.add(s)
        _TASK_SKELETONS[task] = out
    return _TASK_SKELETONS[task]


def _drop_test_molecules(smiles_list: List[str], protein: str) -> List[str]:
    test = set()
    for task in tasks_using(protein):
        test |= _test_skeletons(task)
    return [s for s in smiles_list if _skeleton(s) not in test]


_CURATED_SUBSTRATES = {
    "PEPT1": [
        ("glycylsarcosine",         "NCC(=O)N(C)CC(O)=O"),
        ("glycylglycine",           "NCC(=O)NCC(O)=O"),
        ("glycyl-L-phenylalanine",  "NCC(=O)N[C@@H](Cc1ccccc1)C(O)=O"),
        ("L-alanyl-L-alanine",      "C[C@H](N)C(=O)N[C@@H](C)C(O)=O"),
        ("L-valyl-L-tyrosine",      "CC(C)[C@H](N)C(=O)N[C@@H](Cc1ccc(O)cc1)C(O)=O"),
        ("carnosine",               "NCCC(=O)N[C@@H](Cc1c[nH]cn1)C(O)=O"),
        ("cephalexin",   "CC1=C(C(O)=O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccccc3)[C@H]2SC1"),
        ("cefaclor",     "ClC1=C(C(O)=O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccccc3)[C@H]2SC1"),
        ("amoxicillin",  "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(O)=O"),
        ("captopril",    "C[C@H](CS)C(=O)N1CCC[C@H]1C(O)=O"),
        ("enalapril",    "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H](C)C(=O)N1CCC[C@H]1C(O)=O"),
        ("lisinopril",   "NCCCC[C@H](N[C@@H](CCc1ccccc1)C(O)=O)C(=O)N1CCC[C@H]1C(O)=O"),
        ("benazepril",   "CCOC(=O)[C@H](CCc1ccccc1)N[C@@H]1CCc2ccccc2N(CC(O)=O)C1=O"),
        ("bestatin",              "N[C@@H](Cc1ccccc1)[C@H](O)C(=O)N[C@@H](CC(C)C)C(O)=O"),
        ("5-aminolevulinic acid", "NCC(=O)CCC(O)=O"),
    ],
    "MCT1": [
        ("L-lactate",          "C[C@H](O)C(O)=O"),
        ("pyruvate",           "CC(=O)C(O)=O"),
        ("acetate",            "CC(O)=O"),
        ("propionate",         "CCC(O)=O"),
        ("butyrate",           "CCCC(O)=O"),
        ("beta-hydroxybutyrate", "C[C@@H](O)CC(O)=O"),
        ("acetoacetate",       "CC(=O)CC(O)=O"),
        ("formate",            "OC=O"),
        ("glycolate",          "OCC(O)=O"),
        ("2-oxobutyrate",      "CCC(=O)C(O)=O"),
        ("valproate",          "CCCC(CCC)C(O)=O"),
        ("salicylate",         "OC(=O)c1ccccc1O"),
        ("nicotinate",         "OC(=O)c1cccnc1"),
        ("gamma-hydroxybutyrate", "OCCCC(O)=O"),
        ("mevalonate",         "CC(O)(CCO)CC(O)=O"),
    ],
    "LAT1": [
        ("L-DOPA",         "N[C@@H](Cc1ccc(O)c(O)c1)C(O)=O"),
        ("L-phenylalanine", "N[C@@H](Cc1ccccc1)C(O)=O"),
        ("L-tyrosine",     "N[C@@H](Cc1ccc(O)cc1)C(O)=O"),
        ("L-tryptophan",   "N[C@@H](Cc1c[nH]c2ccccc12)C(O)=O"),
        ("L-leucine",      "CC(C)C[C@H](N)C(O)=O"),
        ("L-isoleucine",   "CC[C@H](C)[C@H](N)C(O)=O"),
        ("L-valine",       "CC(C)[C@H](N)C(O)=O"),
        ("L-methionine",   "CSCC[C@H](N)C(O)=O"),
        ("L-histidine",    "N[C@@H](Cc1c[nH]cn1)C(O)=O"),
        ("gabapentin",     "NCC1(CC(O)=O)CCCCC1"),
        ("pregabalin",     "CC(C)C[C@H](CN)CC(O)=O"),
        ("melphalan",      "N[C@@H](Cc1ccc(N(CCCl)CCCl)cc1)C(O)=O"),
        ("baclofen",       "NC[C@@H](CC(O)=O)c1ccc(Cl)cc1"),
        ("methyldopa",     "C[C@](N)(Cc1ccc(O)c(O)c1)C(O)=O"),
        ("levothyroxine",  "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(O)=O"),
    ],
}


def _discovery_record(protein: str) -> Optional[Dict[str, object]]:
    path = os.path.join(DISCOVERY_CACHE_DIR, f"{protein}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def collect(protein: str) -> Dict[str, object]:
    from agent.registry.proteins import lookup, relation_for

    entry = lookup(protein)
    if entry is None:
        return {"protein": protein, "relation": None, "source": None,
                "inactive_source": None, "test_scope": [], "actives": [], "inactives": []}

    canonical = entry["canonical"]
    relation = relation_for(canonical)
    scope = tasks_using(canonical)
    empty = {"protein": canonical, "relation": relation, "source": None,
             "inactive_source": None, "test_scope": scope, "actives": [], "inactives": []}

    record = _discovery_record(canonical)
    if record is None:
        return empty
    if record.get("relation") != relation:
        return empty

    actives = _drop_test_molecules(list(record.get("actives") or [])[:K], canonical)
    if not actives:
        return empty
    inactives = _drop_test_molecules(list(record.get("inactives") or [])[:K], canonical)

    return {"protein": canonical, "relation": relation,
            "source": record.get("source") or "llm_discovery",
            "active_sources": record.get("active_sources") or {},
            "inactive_source": record.get("inactive_source") if inactives else None,
            "test_scope": scope, "actives": actives, "inactives": inactives}
