import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Interaction:
    type: str
    residue: str
    distance: float
    description: str


LIGAND_HETIDS = {"UNL", "LIG", "UNK"}

_XML_TAGS = {
    'hydrogen_bonds':  ('hydrogen_bond',           ['dist_d-a', 'dist_h-a']),
    'hydrophobic':     ('hydrophobic_interaction', ['dist']),
    'pi_stacking':     ('pi_stack',                ['centdist', 'dist']),
    'pi_cation':       ('pi_cation_interaction',   ['dist']),
    'salt_bridges':    ('salt_bridge',             ['dist']),
    'halogen_bonds':   ('halogen_bond',            ['dist']),
    'metal_complexes': ('metal_complex',           ['dist']),
    'water_bridges':   ('water_bridge',            ['dist_a-w', 'dist_d-w']),
}


class PLIPAnalyzer:
    def __init__(self, plip_path: str = "plip"):
        self.plip_path = plip_path


    def analyze(
        self,
        complex_pdb: str,
        output_dir: str = "."
    ) -> Optional[Dict]:
        os.makedirs(output_dir, exist_ok=True)

        import glob
        plipfixed_files = glob.glob(os.path.join(output_dir, "plipfixed.*.pdb"))
        for f in plipfixed_files:
            os.remove(f)

        cmd = [
            self.plip_path,
            '-f', complex_pdb,
            '-o', output_dir,
            '-x',
            '-t',
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"PLIP run failed: {result.stderr}")
            return None

        xml_files = list(Path(output_dir).glob("*.xml"))
        analysis_result = self._parse_xml(str(xml_files[0]))

        return analysis_result


    def _ligand_bindingsite(self, root):
        sites = root.findall('.//bindingsite')
        for site in sites:
            hetid = site.find('.//hetid')
            if hetid is not None and (hetid.text or '').strip().upper() in LIGAND_HETIDS:
                return site
        return sites[0] if len(sites) == 1 else None

    def _parse_xml(self, xml_path: str) -> Dict:
        root = ET.parse(xml_path).getroot()

        interactions = {key: [] for key in _XML_TAGS}

        site = self._ligand_bindingsite(root)
        if site is None:
            return interactions

        for key, (tag, dist_tags) in _XML_TAGS.items():
            for el in site.findall(f'.//{tag}'):
                resnr = el.find('resnr')
                restype = el.find('restype')

                distance = 0.0
                for dt in dist_tags:
                    d = el.find(dt)
                    if d is not None and d.text:
                        distance = float(d.text)
                        break

                interactions[key].append(Interaction(
                    type=key,
                    residue=resnr.text if resnr is not None else 'UNK',
                    distance=distance,
                    description=restype.text if restype is not None else 'UNK',
                ))

        return interactions


    def generate_report(self, interactions: Dict, scores: List[float]) -> str:
        report = "\n[ Interaction analysis ]\n"

        if scores:
            report += f"• Best docking score: {min(scores):.2f} kcal/mol\n"
            report += f"• Mean score: {sum(scores)/len(scores):.2f} kcal/mol\n"

        total = sum(len(v) for v in interactions.values())
        report += f"\n• Total interactions: {total}\n"

        for interaction_type, items in interactions.items():
            if items:
                report += f"\n  [{interaction_type.replace('_', ' ').title()}] ({len(items)})\n"
                for i, item in enumerate(items[:5], 1):
                    report += f"    {i}. {item.residue} - {item.distance:.2f}Å\n"
                if len(items) > 5:
                    report += f"    ... and {len(items)-5} more\n"

        return report
