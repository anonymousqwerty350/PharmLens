import io
import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

PMCID = "PMC12587445"
EUROPEPMC_SUPPL = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles"
SI_XLSX_NAME = "mp5c01065_si_002.xlsx"
TABLE_S2_SHEET = "Table S2"
HEADER_ROW = 2

TARGETS = ["Pgp", "BCRP", "MRP1", "MRP2"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def download_si_xlsx(out_path):
    print(f"[1] downloading SI from Europe PMC: {EUROPEPMC_SUPPL}")
    req = urllib.request.Request(EUROPEPMC_SUPPL, headers={"Accept": "application/zip"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = zf.namelist()
    match = [n for n in names if n.endswith(SI_XLSX_NAME)]
    if not match:
        raise FileNotFoundError(f"{SI_XLSX_NAME} not in SI zip. contents={names}")
    with zf.open(match[0]) as f, open(out_path, "wb") as g:
        g.write(f.read())
    print(f"    -> {out_path} ({os.path.getsize(out_path)} bytes)")
    return out_path


def load_table_s2(xlsx_path):
    print(f"[2] parsing Table S2: {xlsx_path} sheet='{TABLE_S2_SHEET}'")
    df = pd.read_excel(xlsx_path, sheet_name=TABLE_S2_SHEET, header=HEADER_ROW, engine="openpyxl")
    df = df.rename(columns=lambda c: str(c).strip())
    keep = ["PID", "Source", "Preferred_name", "SMILES"] + \
           [f"{t}_substrate" for t in TARGETS] + [f"{t}_inhibitor" for t in TARGETS]
    df = df[[c for c in keep if c in df.columns]].copy()
    print(f"    -> {len(df)} rows, cols={list(df.columns)}")
    return df


def split_substrate_sets(s2):
    print("[3] splitting substrate sets per transporter")
    sets = {}
    for t in TARGETS:
        col = f"{t}_substrate"
        sub = s2[s2[col].notna()].copy()
        sub = sub[sub[col].isin([0, 1, 0.0, 1.0])]
        sub["label"] = sub[col].astype(int)
        sets[t] = sub[["PID", "Source", "Preferred_name", "SMILES", "label"]].reset_index(drop=True)
        n1 = int((sets[t]["label"] == 1).sum()); n0 = int((sets[t]["label"] == 0).sum())
        print(f"    {t:5s}: {len(sets[t]):4d} rows  (substrate={n1}, non={n0})")
    return sets


def _make_standardizers():
    from rdkit.Chem.MolStandardize import rdMolStandardize
    return rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def standardize_set(df, lfc, unch):
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    recs = []
    n_unparseable = 0
    for _, row in df.iterrows():
        m = Chem.MolFromSmiles(str(row["SMILES"]))
        if m is None:
            n_unparseable += 1; continue
        m = lfc.choose(m)
        if m is None or m.GetNumAtoms() == 0:
            n_unparseable += 1; continue
        if not any(a.GetSymbol() == "C" for a in m.GetAtoms()):
            n_unparseable += 1; continue
        try:
            m = unch.uncharge(m); Chem.SanitizeMol(m)
        except Exception:
            n_unparseable += 1; continue
        canon = Chem.MolToSmiles(m)
        flat = Chem.MolToSmiles(m, isomericSmiles=False)
        r = row.to_dict(); r["canonical_smiles"] = canon; r["flat_smiles"] = flat
        recs.append(r)

    clean = pd.DataFrame(recs)
    n_before = len(clean)
    clean = clean.sort_values("label", ascending=False)
    clean = clean.drop_duplicates(subset="flat_smiles", keep="first").reset_index(drop=True)
    n_dup = n_before - len(clean)

    report = {"orig": len(df), "unparseable_or_inorganic": n_unparseable,
              "dup_removed": n_dup, "final": len(clean),
              "final_sub": int((clean["label"] == 1).sum()),
              "final_non": int((clean["label"] == 0).sum())}
    return clean, report


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xlsx_path = os.path.join(OUT_DIR, SI_XLSX_NAME)

    if not os.path.exists(xlsx_path):
        download_si_xlsx(xlsx_path)
    else:
        print(f"[1] SI already present: {xlsx_path}")

    s2 = load_table_s2(xlsx_path)
    s2.to_csv(os.path.join(OUT_DIR, "paper_SI_TableS2_curated_8794.csv"), index=False)

    sets = split_substrate_sets(s2)

    print("[4] standardization + 2D deduplication")
    lfc, unch = _make_standardizers()
    report_rows = []
    for t in TARGETS:
        raw = sets[t]
        raw.to_csv(os.path.join(OUT_DIR, f"{t}_substrate.csv"), index=False)
        clean, rep = standardize_set(raw, lfc, unch)
        clean.to_csv(os.path.join(OUT_DIR, f"{t}_substrate_clean.csv"), index=False)
        rep["transporter"] = t
        report_rows.append(rep)
        print(f"    {t:5s}: orig={rep['orig']} -> final={rep['final']} "
              f"(removed unparseable/inorganic={rep['unparseable_or_inorganic']}, dup={rep['dup_removed']})")

    report = pd.DataFrame(report_rows)[
        ["transporter", "orig", "unparseable_or_inorganic", "dup_removed",
         "final", "final_sub", "final_non"]]
    report.to_csv(os.path.join(OUT_DIR, "standardization_report.csv"), index=False)


if __name__ == "__main__":
    main()
