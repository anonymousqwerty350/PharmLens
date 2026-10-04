import os

TARGETS = {
    "BSEP":    ("CHEMBL6020", "inhibitor", "transporter"),
    "MRP2":    ("CHEMBL5748", "inhibitor", "transporter"),
    "NKCC1":   ("CHEMBL1615383", "inhibitor", "transporter"),
    "ACE":     ("CHEMBL1808", "inhibitor", "enzyme"),
    "VKORC1":  ("CHEMBL1930", "inhibitor", "enzyme"),
    "COX2":    ("CHEMBL230",  "inhibitor", "enzyme"),
    "CYP17A1": ("CHEMBL3522", "inhibitor", "enzyme"),
    "CYP19A1": ("CHEMBL1978", "inhibitor", "enzyme"),
    "DHFR":    ("CHEMBL202",  "inhibitor", "enzyme"),
    "TYMS":    ("CHEMBL1952", "inhibitor", "enzyme"),
    "SRD5A2":  ("CHEMBL1856", "inhibitor", "enzyme"),
    "DHODH":   ("CHEMBL1966", "inhibitor", "enzyme"),
    "PSMB5":   ("CHEMBL4662", "inhibitor", "enzyme"),
    "ABL1":    ("CHEMBL1862", "inhibitor", "kinase"),
    "EGFR":    ("CHEMBL203",  "inhibitor", "kinase"),
    "VEGFR2":  ("CHEMBL279",  "inhibitor", "kinase"),
    "BRAF":    ("CHEMBL5145", "inhibitor", "kinase"),
    "MEK1":    ("CHEMBL3587", "inhibitor", "kinase"),
    "mTOR":    ("CHEMBL2842", "inhibitor", "kinase"),
    "hERG":    ("CHEMBL240",  "inhibitor", "channel"),
    "Nav1.5":  ("CHEMBL1980", "inhibitor", "channel"),
    "KCNQ4":   ("CHEMBL3576", "inhibitor", "channel"),

    "FXR":     ("CHEMBL2047", "antagonist", "receptor"),
    "ESR1":    ("CHEMBL206",  "antagonist", "receptor"),
    "DRD2":    ("CHEMBL217",  "antagonist", "receptor"),

    "TRPA1":   ("CHEMBL6007", "agonist",    "channel"),
    "PXR":     ("CHEMBL3401", "agonist",    "receptor"),
    "OPRM1":   ("CHEMBL233",  "agonist",    "receptor"),
}

COLUMNS = ("smiles", "label", "type", "pChEMBL", "n_records", "chembl_ids")


def csv_stem(name):
    direction = TARGETS[name][1]
    return name if direction == "inhibitor" else f"{direction}_{name}"


def names_for_category(category):
    return [n for n, v in TARGETS.items() if v[1] == category]


def targets_for(category):
    return {n: f"{csv_stem(n)}.csv" for n in names_for_category(category)}


def dataset_path(name, root=None):
    root = root or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, TARGETS[name][1], "dataset", f"{csv_stem(name)}.csv")
