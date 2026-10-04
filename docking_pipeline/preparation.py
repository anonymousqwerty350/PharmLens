import os
import logging
import subprocess
import requests
from typing import Optional, Tuple
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

logger = logging.getLogger(__name__)


class MoleculePreparator:
    def _finalize_and_write(self, mol: Chem.Mol, smiles: str, output_sdf: str) -> bool:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except ValueError as e:
            logger.warning(f"MMFF optimization failed ({e}), using unoptimized conformer: {smiles[:80]}")
        try:
            Chem.Kekulize(mol, clearAromaticFlags=True)
        except Exception:
            pass
        try:
            writer = Chem.SDWriter(output_sdf)
            writer.write(mol)
            writer.close()
        except Exception as e:
            logger.warning(f"SDF write failed ({e}): {smiles[:80]}")
            return False
        return True

    def smiles_to_conformer(self, smiles: str, output_sdf: str) -> bool:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            mol = Chem.AddHs(mol)

            if AllChem.EmbedMolecule(mol, randomSeed=42) != -1:
                return self._finalize_and_write(mol, smiles, output_sdf)

            params = AllChem.ETKDGv3()
            params.useRandomCoords       = True
            params.randomSeed            = 0xDEAD
            params.useMacrocycleTorsions = True
            params.useSmallRingTorsions  = True
            params.maxIterations         = 1000
            mol_fresh = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if AllChem.EmbedMolecule(mol_fresh, params) != -1 and mol_fresh.GetNumConformers() > 0:
                logger.warning(f"RDKit default failed, succeeded with ETKDGv3 macrocycle: {smiles[:80]}")
                return self._finalize_and_write(mol_fresh, smiles, output_sdf)
        else:
            logger.warning(f"RDKit SMILES parsing failed, trying OpenBabel fallback: {smiles[:80]}")

        try:
            proc = subprocess.run(
                ["obabel", f"-:{smiles}", "-O", output_sdf, "--gen3D", "best", "-h"],
                capture_output=True, text=True, timeout=120,
            )
            if os.path.exists(output_sdf) and os.path.getsize(output_sdf) > 0:
                logger.warning(f"RDKit failed, succeeded via OpenBabel fallback: {smiles[:80]}")
                return True
        except Exception as e:
            logger.warning(f"OpenBabel run failed ({e}): {smiles[:80]}")

        logger.warning(f"3D embedding failed (all methods exhausted): {smiles[:80]}")
        return False

class ProteinPreparator:
    def download_pdb(self, pdb_id: str, output_path: str) -> bool:
        logger.info(f"[ProteinPreparation] downloading PDB: {pdb_id}")

        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(response.text)
            return True

        logger.warning(f"  - no PDB format (HTTP {response.status_code}), trying CIF...")
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        cif_response = requests.get(cif_url, timeout=30)

        if cif_response.status_code != 200:
            logger.error(f"  - CIF download also failed: HTTP {cif_response.status_code}")
            return False

        try:
            import gemmi
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp:
                tmp.write(cif_response.content)
                tmp_cif = tmp.name
            st = gemmi.read_structure(tmp_cif)
            st.write_pdb(output_path)
            os.remove(tmp_cif)
            logger.info(f"  - CIF → PDB conversion complete: {output_path}")
            return True
        except Exception as e:
            logger.error(f"  - CIF → PDB conversion failed: {e}")
            return False


    def extract_clean_protein(self, input_pdb: str, output_pdb: str) -> bool:
        with open(input_pdb, 'r') as f_in:
            with open(output_pdb, 'w') as f_out:
                for line in f_in:
                    if line.startswith('ATOM'):
                        f_out.write(line)
                    elif line.startswith('TER'):
                        f_out.write(line)

                f_out.write('END\n')

        return True


    def fix_protein_valence(self, input_pdb, output_pdb):
        logger.info(f"[Fix] force-repairing valence errors: {input_pdb}")

        mol = Chem.MolFromPDBFile(input_pdb, sanitize=False, removeHs=False)
        if mol is None:
            logger.error("Cannot read file.")
            return False

        problems = Chem.DetectChemistryProblems(mol)
        bad_atoms = {p.GetAtomIdx() for p in problems if p.GetType() == "AtomValenceException"}
        isolated_h = {a.GetIdx() for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 1 and a.GetDegree() == 0}
        to_remove = bad_atoms | isolated_h

        if to_remove:
            if bad_atoms:
                logger.info(f"  - removed {len(bad_atoms)} valence-violating atoms: {sorted(bad_atoms)}")
            if isolated_h:
                logger.info(f"  - removed {len(isolated_h)} initially isolated hydrogens")
            edit = Chem.RWMol(mol)
            for idx in sorted(to_remove, reverse=True):
                edit.RemoveAtom(idx)
            mol = edit.GetMol()

        orphaned_h = {a.GetIdx() for a in mol.GetAtoms()
                      if a.GetAtomicNum() == 1 and a.GetDegree() == 0}
        if orphaned_h:
            logger.info(f"  - removed {len(orphaned_h)} further hydrogens orphaned by the removal")
            edit = Chem.RWMol(mol)
            for idx in sorted(orphaned_h, reverse=True):
                edit.RemoveAtom(idx)
            mol = edit.GetMol()

        Chem.MolToPDBFile(mol, output_pdb)
        return True


    def pdb_to_pdbqt(self, pdb_path: str, pdbqt_path: str, pdb_fixed_path: str) -> bool:
        logger.info(f"[ProteinPreparation] converting PDB → PDBQT...")

        import subprocess
        import os


        self.fix_protein_valence(pdb_path, pdb_fixed_path)

        result = subprocess.run(
            ['unidocktools', 'proteinprep',
            '-r', pdb_fixed_path,
            '-o', pdbqt_path],
            capture_output=True,
            text=True
        )


        if result.returncode == 0 and os.path.exists(pdbqt_path):
            logger.info(f"  - PDBQT conversion complete: {pdbqt_path}")

        else:
            logger.error(f"    ✗ UniDockTools failed: {result.stderr}")

        if os.path.exists(pdb_fixed_path):
            os.remove(pdb_fixed_path)
