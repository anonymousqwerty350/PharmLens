import os
import logging
import subprocess
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class UniDockRunner:
    def __init__(self, work_dir: str = "./docking"):
        self.work_dir = work_dir

    def find_ligand_binding_site(self, pdb_path: str) -> Optional[Dict[str, List[float]]]:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_path)

        BLACKLIST = [
            'HOH', 'WAT', 'CL', 'NA', 'MG', 'ZN', 'CA', 'K',
            'SO4', 'PO4',

            'EDO', 'PEG', 'GOL', 'DMS', 'ACT', 'FMT',

            'NAG', 'MAN', 'BMA', 'CLR', '3PE', 'OLC', 'P4C',
            'CHO', 'DLP', 'PLM', 'CDL',

            'ATP', 'ADP', 'AMP', 'ANP', 'HEM', 'HEC', 'FAD',
            'FMN', 'NAD', 'NDP', 'COH', 'NAP',
            'AGS',

            'VO4', 'CHS', 'GDN',

            'GSH',

            'UNL',

            'Y01', 'PIP', 'PI4', 'PLC', 'PLQ',
            'PT5',

            'OLA', 'MYR', 'STR', 'LDA', 'LBN',

            'LMT', 'LMU', 'BOG', 'LHG', 'C10', 'SDS',

            'MPD', 'TRS', 'EPE', 'MES', 'IPA', 'P33', '12P', '1PE',

            'GTP', 'GDP', 'GNP', 'GSP',

            'DA', 'DT', 'DG', 'DC', 'DU',

            'CRO',
        ]

        candidates = []
        for residue in structure.get_residues():
            resname = residue.resname.strip()

            if residue.id[0] != ' ' and resname not in BLACKLIST:
                atoms = list(residue.get_atoms())

                if 5 < len(atoms) < 100:
                    candidates.append(residue)

        if not candidates:
            return None

        best_ligand = max(candidates, key=lambda r: len(list(r.get_atoms())))

        coords_array = np.array([atom.coord for atom in best_ligand])
        center = np.mean(coords_array, axis=0).tolist()
        mins = coords_array.min(axis=0)
        maxs = coords_array.max(axis=0)

        size = (maxs - mins + 12).tolist()
        size = [max(15.0, min(s, 25.0)) for s in size]

        logger.info(f"[ligand found]: {best_ligand.resname} ({len(list(best_ligand.get_atoms()))} atoms)")
        logger.info(f"- Pocket Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        logger.info(f"- Pocket Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]")

        return {'center': center, 'size': size}


    def run_fpocket(self, pdb_path: str) -> Optional[Dict[str, List[float]]]:
        try:
            pdb_dir = os.path.dirname(pdb_path)
            pdb_name = Path(pdb_path).stem

            result = subprocess.run(
                ['fpocket', '-f', pdb_path],
                cwd=pdb_dir,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                logger.error(f"    ✗ fpocket run failed")
                logger.error(f"      {result.stderr}")
                return None

            fpocket_output_dir = os.path.join(pdb_dir, f"{pdb_name}_out")
            pocket1_pdb = os.path.join(fpocket_output_dir, "pockets", "pocket1_atm.pdb")

            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            pocket_structure = parser.get_structure('pocket', pocket1_pdb)

            pocket_coords = []
            for atom in pocket_structure.get_atoms():
                pocket_coords.append(atom.coord)

            if not pocket_coords:
                logger.error(f"    ✗ no pocket coordinates")
                return None

            coords_array = np.array(pocket_coords)
            center = np.mean(coords_array, axis=0).tolist()

            mins = coords_array.min(axis=0)
            maxs = coords_array.max(axis=0)
            size = (maxs - mins + 12).tolist()

            size = [max(15.0, min(s, 25.0)) for s in size]

            logger.info(f"[Pocket] found")
            logger.info(f"- Pocket Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            logger.info(f"- Pocket Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]")

            return {'center': center, 'size': size}

        except Exception as e:
            logger.error(f"    ✗ fpocket run error: {e}")
            return None


    def estimate_binding_site(self, pdb_path: str) -> Dict[str, List[float]]:
        result = self.find_ligand_binding_site(pdb_path)
        if result:
            return result

        result = self.run_fpocket(pdb_path)
        if result:
            return result


    def run_batch(
        self,
        receptor_pdbqt: str,
        ligand_sdfs: List[str],
        output_dir: str,
        center: List[float],
        size: List[float],
        search_mode: str = "balance",
        num_modes: int = 9,
        exhaustiveness: int = 8,
    ) -> Dict[str, Optional[str]]:
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "unidock",
            "--receptor",       receptor_pdbqt,
            "--gpu_batch",      *ligand_sdfs,
            "--center_x",       str(center[0]),
            "--center_y",       str(center[1]),
            "--center_z",       str(center[2]),
            "--size_x",         str(size[0]),
            "--size_y",         str(size[1]),
            "--size_z",         str(size[2]),
            "--dir",            output_dir,
            "--search_mode",    search_mode,
            "--num_modes",      str(num_modes),
            "--exhaustiveness", str(exhaustiveness),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Batch docking failed:\n  STDERR: {result.stderr}")

        mapping: Dict[str, Optional[str]] = {}
        for sdf in ligand_sdfs:
            stem = Path(sdf).stem
            out_sdf = os.path.join(output_dir, f"{stem}_out.sdf")
            mapping[sdf] = out_sdf if os.path.exists(out_sdf) else None

        return mapping

    def run(
        self,
        receptor_pdbqt: str,
        ligand_sdf: str,
        output_dir: str,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        search_mode: str = "balance",
        num_modes: int = 9,
        exhaustiveness: int = 8
    ) -> Optional[str]:
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, f"docking_result.sdf")

        cmd = [
            'unidock',
            '--receptor', receptor_pdbqt,
            '--ligand', ligand_sdf,
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(size[0]),
            '--size_y', str(size[1]),
            '--size_z', str(size[2]),
            '--out', output_file,
            '--search_mode', search_mode,
            '--num_modes', str(num_modes),
            '--exhaustiveness', str(exhaustiveness)
        ]

        logger.info(f"  - Receptor: {receptor_pdbqt}")
        logger.info(f"  - Ligand: {ligand_sdf}")
        logger.info(f"  - Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        logger.info(f"  - Size: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]")
        logger.info(f"  - Search mode: {search_mode}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and os.path.exists(output_file):
            return output_file
        else:
            logger.error(f"Docking failed:")
            logger.error(f"    STDOUT: {result.stdout}")
            logger.error(f"    STDERR: {result.stderr}")
            return None


    def docking_complex_reusult(
        self,
        receptor_pdbqt: str,
        ligand_sdf: str,
        output_dir: str,
        protein_pdb: str,
        center: Optional[List[float]] = None,
        size: Optional[List[float]] = None,
        num_top_poses: int = 3,
        search_mode: str = "balance",
        exhaustiveness: int = 8
    ) -> List[Dict]:
        docking_result = self.run(
            receptor_pdbqt=receptor_pdbqt,
            ligand_sdf=ligand_sdf,
            output_dir=output_dir,
            center=center,
            size=size,
            search_mode=search_mode,
            exhaustiveness=exhaustiveness
        )

        if not docking_result:
            logger.error(f"[docking_complex_reusult] docking failed")
            return []

        scores = self.parse_scores(docking_result)

        converter = FormatConverter()
        results = []

        for i in range(1, min(num_top_poses + 1, len(scores) + 1)):
            ligand_pdb = os.path.join(output_dir, f"ligand_docking_pose{i}.pdb")

            if not converter.sdf_to_pdb(docking_result, ligand_pdb, model_num=i):
                continue

            complex_pdb = os.path.join(output_dir, f"complex_pose{i}.pdb")
            if not converter.create_complex(
                protein_pdb=protein_pdb,
                ligand_pdb=ligand_pdb,
                output_complex_pdb=complex_pdb
            ):
                continue

            results.append({
                'model': i,
                'score': scores[i-1] if i-1 < len(scores) else None,
                'complex_pdb': complex_pdb,
                'ligand_pdb': ligand_pdb
            })


        return results


    def parse_scores(self, result_path: str) -> List[float]:
        scores = []
        try:
            with open(result_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if '<Uni-Dock RESULT>' in line and (i + 1) < len(lines):
                        next_line = lines[i+1]
                        if 'ENERGY=' in next_line:
                            score_part = next_line.split('ENERGY=')[1].split()[0]
                            scores.append(float(score_part))
        except Exception as e:
            logger.error(f"  ✗ score parsing failed: {e}")

        return scores


class FormatConverter:
    @staticmethod
    def sdf_to_pdb(sdf_path: str, pdb_path: str, model_num: int = 1) -> bool:
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
        mol = supplier[model_num - 1]

        writer = Chem.PDBWriter(pdb_path)
        writer.write(mol)
        writer.close()

        return True


    @staticmethod
    def create_complex(
        protein_pdb: str,
        ligand_pdb: str,
        output_complex_pdb: str
    ) -> bool:
        with open(output_complex_pdb, 'w') as f_out:
            last_line = ''
            with open(protein_pdb, 'r') as f_in:
                for line in f_in:
                    if line.startswith(('ATOM', 'TER')):
                        f_out.write(line)
                        last_line = line

            if not last_line.startswith('TER'):
                f_out.write('TER\n')

            with open(ligand_pdb, 'r') as f_in:
                for line in f_in:
                    if line.startswith(('ATOM', 'HETATM')):
                        if line.startswith('ATOM'):
                            line = 'HETATM' + line[6:]
                        f_out.write(line)

            f_out.write('END\n')

        return True


if __name__ == "__main__":
    test_pdb = "path/to/protein_6C0V.pdb"

    runner = UniDockRunner(work_dir="./debug_docking")

    result = runner.find_ligand_binding_site(test_pdb)
    print(result)
