from typing import Dict, List

from agent.prompts.base_templates import JSON_ONLY


def recognition_system() -> str:
    return (
        "You are a molecular pharmacologist. For a given protein you decide HOW it recognises "
        "its ligands: by a NARROW structural class (essentially all its ligands share one core "
        "pharmacophore) or by a BROAD, polyspecific pocket (structurally diverse ligands). This "
        "is a judgement about substrate-class breadth from established pharmacology, not about "
        "binding affinity."
    )


def _catalog_block(catalog: List[Dict[str, str]]) -> str:
    lines = []
    for e in catalog:
        desc = (e.get("description") or "").replace("Number of ", "").strip()
        lines.append(f"  {e['name']}: {desc}")
    return "\n".join(lines)


def recognition_user(protein: str, relation: str, examples: List[str],
                     catalog: List[Dict[str, str]]) -> str:
    ex = ", ".join(examples[:12]) if examples else "(none available — use your own knowledge)"
    return (
        f"Protein: {protein}\n"
        f"Its ligands in question are its {relation}s.\n"
        f"Known example {relation}s: {ex}\n\n"
        "1) narrow — true if its ligands belong to ONE constrained substrate class whose whole "
        "core is shared (amino-acid carriers like LAT1, peptide transporters like PEPT1, "
        "monocarboxylate transporters like MCT1, nucleoside transporters like ENT1, sugar "
        "transporters like GLUT1). false if polyspecific: it binds structurally DIVERSE scaffolds "
        "in a pocket — efflux pumps (P-gp, BCRP), kinases, ion channels, and essentially ALL "
        "enzymes, nuclear receptors and GPCRs.\n"
        "   CRITICAL: merely SHARING A FUNCTIONAL GROUP is NOT narrow — diverse NSAIDs all bear a "
        "carboxylate/sulfonamide yet COX-2 is a polyspecific pocket (narrow=false). Narrow means "
        "the ENTIRE core is one class.\n\n"
        "2) required_groups — if narrow=true, the functional groups that EVERY ligand of this "
        "protein MUST carry (its necessary recognition features). Choose ONLY exact names from "
        "this catalog (a molecule is filtered out if it lacks any one of them, so keep it MINIMAL "
        "— only the truly required groups; e.g. an amino-acid carrier -> [\"fr_NH2\",\"fr_COO\"], a "
        "monocarboxylate carrier -> [\"fr_COO\"], a peptide transporter -> [\"fr_amide\",\"fr_COO\"]).\n"
        "If narrow=false, use [].\n\n"
        f"Functional-group catalog (name: meaning):\n{_catalog_block(catalog)}\n\n"
        f"{JSON_ONLY}\n"
        "```json\n"
        '{"narrow": true, "required_groups": ["fr_..."], "evidence": "<one line>"}\n'
        "```"
    )
