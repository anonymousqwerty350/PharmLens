#!/usr/bin/env python
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.ML.Descriptors import MoleculeDescriptors

RDLogger.DisableLog("rdApp.*")

DESCS = ["ECFP6", "MACCS", "RDKit"]

RDKIT_NAMES = [d[0] for d in Descriptors.descList]
_RDKIT_CALC = MoleculeDescriptors.MolecularDescriptorCalculator(RDKIT_NAMES)

_lfc = rdMolStandardize.LargestFragmentChooser()
_unch = rdMolStandardize.Uncharger()


def _is_organic(mol):
    return any(a.GetSymbol() == "C" for a in mol.GetAtoms())


def standardize(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    mol = _lfc.choose(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    if not _is_organic(mol):
        return None, None
    try:
        mol = _unch.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None
    return Chem.MolToSmiles(mol), Chem.MolToSmiles(mol, isomericSmiles=False)


def mol_ecfp6(mol):
    return [float(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024)]


def mol_maccs(mol):
    return [float(x) for x in MACCSkeys.GenMACCSKeys(mol)]


def build_descriptor_frames(mols, drop_inf_cols=True):
    ecfp6 = pd.DataFrame([mol_ecfp6(m) for m in mols])
    maccs = pd.DataFrame([mol_maccs(m) for m in mols])
    rd = pd.DataFrame([list(_RDKIT_CALC.CalcDescriptors(m)) for m in mols],
                      columns=RDKIT_NAMES)
    rd = rd.replace([np.inf, -np.inf], np.nan)
    rd = rd.fillna(rd.mean())
    if drop_inf_cols:
        rd = rd.loc[:, ~rd.isin([np.inf, -np.inf]).any(axis=0)]
    return {"ECFP6": ecfp6.reset_index(drop=True),
            "MACCS": maccs.reset_index(drop=True),
            "RDKit": rd.reset_index(drop=True)}


def inference_descriptor_frames(mols):
    frames = build_descriptor_frames(mols, drop_inf_cols=False)
    frames["ECFP6"].columns = [str(c) for c in frames["ECFP6"].columns]
    frames["MACCS"].columns = [str(c) for c in frames["MACCS"].columns]
    return frames
