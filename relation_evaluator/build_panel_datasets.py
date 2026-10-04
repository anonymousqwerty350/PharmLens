import argparse
import collections
import json
import math
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

from panel import COLUMNS, TARGETS, csv_stem, names_for_category
from qsar_core import eligibility

RDLogger.DisableLog("rdApp.*")

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


ACTIVATION = {"EC50", "AC50", "Potency"}
INHIBITION = {"IC50"}
BINDING = {"Ki", "Kd"}

POTENT = 6.0
WEAK = 5.0

CELL_ASSAY = re.compile(
    r"cell viability|cell growth|cytotox|antiprolifer|proliferation of|growth inhibition|"
    r"antimycobacter|antibacter|antimalarial|antifungal|antitumor|antiviral|"
    r"survival of|colony|\bmic\b|parasit")

KI_INHIBITION = re.compile(r"inhibit|competitive|kinetic|enzyme activ|catalytic|substrate")
KI_DISPLACEMENT = re.compile(r"displacement|radioligand|\[3h\]|\[125i\]|\[35s\]|"
                             r"competition binding|binding affinity|\bspa\b")


def _get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read())
        except Exception:                                       # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(3 * (attempt + 1))
    return None


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chembl_cache")


def fetch_activities(target, refresh=False):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{target}.json")
    if not refresh and os.path.exists(path):
        try:
            with open(path) as f:
                rows = json.load(f)
            print(f"  using cache, {len(rows)} records ({path})", flush=True)
            return rows
        except Exception:                                       # noqa: BLE001
            pass
    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({"target_chembl_id": target, "format": "json",
                                    "limit": 1000, "offset": offset})
        doc = _get(f"{CHEMBL}/activity?{q}")
        batch = (doc or {}).get("activities") or []
        rows.extend(batch)
        if len(batch) < 1000 or offset > 120000:
            break
        offset += 1000
        time.sleep(0.15)
    with open(path, "w") as f:
        json.dump(rows, f)
    return rows


def direction_from_assay(desc, default_inhibition):
    if not isinstance(desc, str):
        return "antagonist" if default_inhibition else None
    d = desc.lower()
    ago = re.search(r"\bagonist|agonistic|activation of|transactivation|induction of|stimulat", d)
    if "antagonist" in d or "inverse agonist" in d:
        return "antagonist"
    ant_pattern = (r"antagoni|inhibition of|inhibitory|inhibit|blockade|blocking|repress"
                   if default_inhibition else
                   r"antagoni|inhibition of|blockade|blocking|repress")
    ant = re.search(ant_pattern, d)
    if ago and not ant:
        return "agonist"
    if ant and not ago:
        return "antagonist"
    if default_inhibition and not ago:
        return "antagonist"
    return None


def ki_is_inhibition(desc, target_class):
    if target_class != "enzyme":
        return False
    d = str(desc).lower()
    return bool(KI_INHIBITION.search(d)) and not KI_DISPLACEMENT.search(d)


def record_potency(a):
    rel = a.get("standard_relation")
    if rel == "=" and a.get("pchembl_value"):
        try:
            p = float(a["pchembl_value"])
        except (TypeError, ValueError):
            return None, None, False
        if p >= POTENT:
            return "strong", p, False
        return ("weak", p, False) if p < WEAK else (None, None, False)
    if rel == ">" and a.get("standard_value") and a.get("standard_units") == "nM":
        try:
            upper = -math.log10(float(a["standard_value"]) * 1e-9)
        except (TypeError, ValueError):
            return None, None, False
        return ("weak", upper, True) if upper <= WEAK else (None, None, False)
    return None, None, False


def parent_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def build(name, target, direction, target_class, refresh=False):
    default_inhibition = direction == "inhibitor"
    print(f"\n[{name}] {target}  direction={direction}  class={target_class}", flush=True)
    rows = fetch_activities(target, refresh=refresh)
    print(f"  ChEMBL records {len(rows)}", flush=True)

    stats = collections.Counter()
    structures = {}
    for a in rows:
        smi, mol_id = a.get("canonical_smiles"), a.get("molecule_chembl_id")
        stype = a.get("standard_type")
        if not smi or not mol_id or stype not in ACTIVATION | INHIBITION | BINDING:
            continue
        desc = a.get("assay_description")
        if CELL_ASSAY.search(str(desc).lower()):
            stats["excluded: cell/organism assay"] += 1
            continue
        s, pval, is_bound = record_potency(a)
        if s is None:
            continue
        parent = parent_structure(smi)
        if parent is None:
            continue

        d = direction_from_assay(desc, default_inhibition)
        if stype in ACTIVATION and d == "agonist":
            axis = "activation"
        elif stype in INHIBITION and d == "antagonist":
            axis = "inhibition"
        elif stype in BINDING and ki_is_inhibition(desc, target_class) and stype == "Ki":
            axis = "inhibition"
            stats["accepted enzyme Ki as inhibition"] += 1
        elif stype in BINDING:
            axis = "binding"
        else:
            stats["excluded: direction unknown"] += 1
            continue

        rec = structures.setdefault(parent, {"ids": set(), "flags": collections.defaultdict(bool),
                                             "n": 0, "best": {}})
        rec["ids"].add(mol_id)
        rec["n"] += 1
        rec["flags"][f"{axis}_{s}"] = True
        prev = rec["best"].get(axis)
        if prev is None or pval > prev[0]:
            rec["best"][axis] = (pval, f"{stype} (>)" if is_bound else stype)
        if is_bound:
            stats["used '>' bound as negative evidence"] += 1

    axis = "activation" if direction == "agonist" else "inhibition"
    out, n_conflict = [], 0
    for parent, rec in structures.items():
        f = rec["flags"]
        if f[f"{axis}_strong"] and (f[f"{axis}_weak"] or f["binding_weak"]):
            n_conflict += 1
            continue
        if f[f"{axis}_strong"] or f[f"{axis}_weak"]:
            label = 1 if f[f"{axis}_strong"] else 0
            best = rec["best"].get(axis)
        elif f["binding_weak"]:
            label = 0
            best = rec["best"].get("binding")
        else:
            continue
        out.append({"smiles": parent, "label": label,
                    "type": best[1] if best else None,
                    "pChEMBL": round(best[0], 2) if best else None,
                    "n_records": rec["n"],
                    "chembl_ids": ";".join(sorted(rec["ids"]))})

    for k, v in stats.items():
        print(f"    {k}: {v}", flush=True)
    print(f"    structures excluded as contradictory: {n_conflict}", flush=True)
    return pd.DataFrame(out, columns=list(COLUMNS)), len(structures)


def scaffold_count(df, label):
    scaffolds = set()
    for smi in df[df.label == label]["smiles"]:
        try:
            scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(smi))
        except Exception:                                       # noqa: BLE001
            continue
    return len(scaffolds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", choices=["inhibitor", "antagonist", "agonist"],
                    help="build one relation's datasets only; the three are separate "
                         "datasets with separate rules and must be built separately")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--out_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--refresh", action="store_true",
                    help="re-download from ChEMBL instead of using the cached responses")
    args = ap.parse_args()

    if args.targets:
        names = args.targets
    elif args.category:
        names = names_for_category(args.category)
    else:
        ap.error("one of --category or --targets must be given "
                 "(the three relations are separate datasets)")
    print(f"[build] {args.category or 'custom'}: {len(names)} target(s) — {', '.join(names)}",
          flush=True)

    summary = []
    for name in names:
        target, direction, tclass = TARGETS[name]
        df, n_structures = build(name, target, direction, tclass, refresh=args.refresh)

        out_dir = os.path.join(args.out_root, direction, "dataset")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{csv_stem(name)}.csv")
        df.to_csv(path, index=False)

        pos = int((df.label == 1).sum()) if len(df) else 0
        neg = int((df.label == 0).sum()) if len(df) else 0
        s1 = scaffold_count(df, 1) if pos else 0
        s0 = scaffold_count(df, 0) if neg else 0
        verdict = eligibility.check(name, path)
        summary.append({"target": name, "direction": direction, "class": tclass,
                        "label1": pos, "label0": neg, "scaffold1": s1, "scaffold0": s0,
                        "structures_seen": n_structures,
                        "trainable": verdict.trainable, "gate_reason": verdict.reason})
        print(f"  saved {path}", flush=True)
        print(f"    label1={pos} ({s1} scaffolds)  label0={neg} ({s0} scaffolds)", flush=True)
        if not verdict.trainable:
            print(f"    excluded from training — {verdict.reason}", flush=True)

    print()
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    tag = args.category or "custom"
    sdf.to_csv(os.path.join(args.out_root, f"panel_dataset_summary_{tag}.csv"), index=False)


if __name__ == "__main__":
    main()
