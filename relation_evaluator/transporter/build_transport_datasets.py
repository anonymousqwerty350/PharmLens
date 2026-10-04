import argparse
import json
import os
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

_CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

TARGETS = {
    "LAT1":    ("CHEMBL4459",    "Q01650", "SLC7A5, large neutral amino acid transporter"),
    "MATE1":   ("CHEMBL1743126", "Q96FL8", "SLC47A1, multidrug and toxin extrusion 1"),
    "OAT1":    ("CHEMBL1641347", "Q4U2R8", "SLC22A6, organic anion transporter 1"),
    "OATP1B1": ("CHEMBL1697668", "Q9Y6L6", "SLCO1B1, hepatic organic anion transporter"),
    "PEPT1":   ("CHEMBL4605",    "P46059", "SLC15A1, proton-coupled peptide transporter 1"),
}

CURATED = {"Pgp", "BCRP", "MRP1", "MRP2"}

TRANSPORT_TYPES = ("Km", "Vmax", "Kt", "Jmax", "Vmax/Km")

_INHIBITION_CONTEXT = re.compile(
    r"inhibit|block|antagon|suppress|efflux ratio|reversal|IC50|\bKi\b", re.I)


def _get(url, retries=3, timeout=90):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:                                       # noqa: BLE001
            if attempt == retries - 1:
                return None
            time.sleep(2 * (attempt + 1))
    return None


def fetch_transport_rows(target):
    per_molecule, rejected = {}, 0
    for stype in TRANSPORT_TYPES:
        offset = 0
        while True:
            q = urllib.parse.urlencode({"target_chembl_id": target, "standard_type": stype,
                                        "format": "json", "limit": 1000, "offset": offset})
            doc = _get(f"{_CHEMBL}/activity?{q}")
            rows = (doc or {}).get("activities") or []
            for a in rows:
                mid = a.get("molecule_chembl_id")
                if not mid:
                    continue
                description = a.get("assay_description") or ""
                if _INHIBITION_CONTEXT.search(description):
                    rejected += 1
                    continue
                rec = per_molecule.setdefault(mid, {"types": set(), "n": 0, "assay": ""})
                rec["types"].add(stype)
                rec["n"] += 1
                if not rec["assay"]:
                    rec["assay"] = description[:160]
            if len(rows) < 1000:
                break
            offset += 1000
    return per_molecule, rejected


def fetch_molecules(chembl_ids):
    out = {}
    for i in range(0, len(chembl_ids), 40):
        chunk = chembl_ids[i:i + 40]
        doc = _get(f"{_CHEMBL}/molecule/set/{';'.join(chunk)}?format=json")
        for mol in ((doc or {}).get("molecules") or []):
            mid = mol.get("molecule_chembl_id")
            if not mid:
                continue
            out[mid] = (mol.get("pref_name") or "",
                        (mol.get("molecule_structures") or {}).get("canonical_smiles") or "")
    return out


def make_standardizers():
    from rdkit.Chem.MolStandardize import rdMolStandardize
    return rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def standardize(smiles, lfc, unch):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None, None
    try:
        mol = lfc.choose(mol)
        if not any(a.GetSymbol() == "C" for a in mol.GetAtoms()):
            return None, None
        mol = unch.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:                                           # noqa: BLE001
        return None, None
    canonical = Chem.MolToSmiles(mol)
    flat = Chem.MolToSmiles(mol, isomericSmiles=False)
    return canonical, flat


def build(name, target):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    print(f"[{name}] target={target}", flush=True)
    per_molecule, rejected = fetch_transport_rows(target)
    print(f"    transport measurements: {len(per_molecule)} unique molecules "
          f"({rejected} records excluded as inhibition-context assays)", flush=True)
    if not per_molecule:
        return pd.DataFrame(columns=["PID", "Source", "Preferred_name", "SMILES", "label"])

    info = fetch_molecules(sorted(per_molecule))
    lfc, unch = make_standardizers()

    rows, seen_flat = [], {}
    dropped = 0
    for mid in sorted(per_molecule):
        pref, raw = info.get(mid, ("", ""))
        canonical, flat = standardize(raw, lfc, unch)
        if canonical is None:
            dropped += 1
            continue
        prev = seen_flat.get(flat)
        if prev is not None and per_molecule[prev]["n"] >= per_molecule[mid]["n"]:
            continue
        seen_flat[flat] = mid
        rows.append({
            "PID": mid,
            "Source": "ChEMBL_Km_Vmax",
            "Preferred_name": pref,
            "SMILES": canonical,
            "label": 1,
            "measured": ",".join(sorted(per_molecule[mid]["types"])),
            "n_records": per_molecule[mid]["n"],
            "assay": per_molecule[mid]["assay"],
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)
    print(f"    passed standardization {len(df)} (structure failed / inorganic {dropped})", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="*", default=sorted(TARGETS),
                    help=f"default: {', '.join(sorted(TARGETS))}")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite an existing CSV. Never use this on the author-curated sets")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for name in args.targets:
        if name in CURATED:
            print(f"[{name}] skipped — author-curated set (contains label 0), not overwritten")
            continue
        if name not in TARGETS:
            print(f"[{name}] skipped — not in TARGETS")
            continue

        path = os.path.join(args.out_dir, f"{name}_substrate.csv")
        if os.path.exists(path) and not args.overwrite:
            print(f"[{name}] skipped — already present: {path}  (force with --overwrite)")
            continue

        df = build(name, TARGETS[name][0])
        df.to_csv(path, index=False)
        print(f"    -> {path}  ({len(df)} rows, label 1 only)", flush=True)
        summary.append({"target": name, "chembl": TARGETS[name][0],
                        "label1": len(df), "label0": 0, "path": os.path.basename(path)})

    if summary:
        sdf = pd.DataFrame(summary)
        print()
        print(sdf.to_string(index=False))
        report = os.path.join(args.out_dir, "transport_dataset_summary.csv")
        sdf.to_csv(report, index=False)
        print(f"\nSummary: {report}")
        print("label 0 is not produced — a molecule that is not transported has no Km, so it has no negative counterpart.")


if __name__ == "__main__":
    main()
