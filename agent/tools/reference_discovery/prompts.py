from typing import Dict, List, Optional

from agent.prompts.base_templates import JSON_ONLY

_RELATION_GLOSS = {
    "substrate": ("molecules the protein TRANSPORTS across the membrane (carrier substrates) or ENZYMATICALLY PROCESSES - NOT inhibitors or blockers of it. "
                  "A molecule that binds and blocks the protein without being transported is the WRONG class here and must be excluded."),
    "inhibitor": ("molecules that INHIBIT or BLOCK the protein's activity (potent binders, typically reported with an IC50/Ki/pIC50) - NOT mere substrates or passively transported passengers."),
    "antagonist": ("molecules that ANTAGONISE the receptor (block its activation) - NOT agonists and NOT simple substrates."),
    "agonist": ("molecules that ACTIVATE the receptor (agonists, including partial agonists, typically reported with an EC50 and a measured functional response) - "
                "NOT antagonists, inverse agonists or blockers. Those occupy the same pocket but are the WRONG class here and must be excluded."),
}


def positive_system(task_description: str) -> str:
    return (
        "You are a professional AI molecular pharmacologist.\n"
        "Your job is to assemble a reference set of KNOWN ligands for a specific protein to serve as ground truth.\n\n"
        f"Downstream Task: {task_description}\n\n"

        "Follow these strict rules:\n"
        "1. NO EXTERNAL KNOWLEDGE: Work ONLY from the provided abstracts. Never assume or rely on memory.\n"
        "2. NAMES ONLY: Propose molecule names only. Never write chemical structures or SMILES.\n"
        "3. UNAMBIGUOUS NAMES: Use generic/INN names (e.g., 'L-DOPA' instead of 'DOPA').\n"
    )


def positive_user(protein: str, relation: str, abstracts: List[Dict[str, str]],
                   n_target: int = 15, n_floor: int = 10, relation_gloss: Optional[str] = None) -> str:
    gloss = relation_gloss or _RELATION_GLOSS.get(relation, f"molecules with a {relation} relationship to it")

    lit = "\n\n".join(
        f"[PMID {a.get('pmid', '?')}]\n{a.get('text', '')[:2000]}" for a in abstracts
    ) or "(no abstracts retrieved)"

    return (
        f"Protein: {protein}\n"
        f"Relationship wanted (relation = \"{relation}\"): {gloss}\n\n"

        f"### Retrieved Literature\n{lit}\n\n"

        f"Task: Extract {n_floor} to {n_target} molecules. The corpus above runs to dozens of "
        "abstracts and normally names more established ligands than a first reading notices.\n"
        "Follow this step-by-step extraction process:\n\n"

        "Step 1. Identify Candidate Molecules.\n"
        "- Find specific INDIVIDUAL MOLECULES.\n"
        "- EXCLUDE broad classes (e.g., 'flavonoids', 'indazole derivatives').\n\n"

        "Step 2. Verify the Relationship.\n"
        f"- The text MUST connect the molecule to {protein} as a {relation}.\n"
        "- Presupposed connections (e.g., resistance, therapy, or IC50) are valid evidence.\n\n"

        "Step 3. Apply Exclusion Filters. EXCLUDE the molecule if:\n"
        "- It belongs to the opposite functional class.\n"
        "- It is named only for contrast or reports a NEGATIVE result.\n"
        "- Its connection in the text is to a DIFFERENT protein.\n\n"

        "Step 4. Finalize Output & Grounding.\n"
        "- Spread choices across diverse chemical/mechanistic classes if multiple exist.\n"
        "- Ground every molecule: quote the exact sentence tying it to the protein, and never "
        "add one absent from the text to reach the count. Where the text supports fewer, "
        "fewer is correct.\n\n"

        f"{JSON_ONLY}\n"
        "```json\n"
        "{\n"
        '  "ligands": [\n'
        '    {"name": "<compound name>", "class": "<structural/mechanistic class>", '
        '"evidence": "<quoted sentence from the abstract that shows the connection>", '
        '"source_pmid": "<the PMID>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )


def negative_system(protein: str, relation: str) -> str:
    return (
        f"You are a professional AI molecular pharmacologist.\n"
        f"Your job is to extract NEGATIVE reference ligands (molecules experimentally shown NOT to be {relation}s of {protein}).\n\n"

        "Follow these strict rules:\n"
        "1. NO ASSUMPTIONS: Work ONLY from the provided abstracts. Never use external knowledge.\n"
        "2. EXPLICIT EVIDENCE REQUIRED: Absence of a claim is NOT a negative claim.\n"
        "3. NAMES ONLY: Never write chemical structures or SMILES.\n"
    )


def negative_user(protein: str, relation: str, abstracts: List[Dict[str, str]],
                  n_target: int = 15) -> str:
    lit = "\n\n".join(
        f"[PMID {a.get('pmid', '?')}]\n{a.get('text', '')[:1200]}" for a in abstracts
    ) or "(no abstracts retrieved)"

    return (
        f"Protein: {protein}\n"
        f"Wanted: molecules the literature below reports as NOT {relation}s of {protein}.\n\n"

        f"### Retrieved Literature\n{lit}\n\n"

        f"Task: Extract up to {n_target} non-interacting molecules.\n"
        "Follow this step-by-step extraction process:\n\n"

        "Step 1. Identify Candidate Molecules.\n"
        "- Find unambiguous compound names only.\n\n"

        "Step 2. Verify Explicit Non-interaction.\n"
        "- The text MUST explicitly state tested non-interaction (e.g., 'showed no inhibition', 'inactive', 'IC50 > 100 uM').\n"
        "- Weak/partial activity does NOT count as a negative.\n\n"

        f"Step 3. Apply Contradiction Filter. EXCLUDE the molecule if:\n"
        f"- The same text reports it as an actual {relation} elsewhere.\n\n"

        "Step 4. Finalize Output & Grounding.\n"
        "- Quote the exact sentence proving non-interaction.\n"
        "- If evidence is missing, return an empty list. NEVER pad the list.\n\n"

        f"{JSON_ONLY}\n"
        "```json\n"
        "{\n"
        '  "ligands": [\n'
        '    {"name": "<compound name>", "evidence": "<quoted sentence from the abstract>", '
        '"source_pmid": "<PMID>"}\n'
        "  ]\n"
        "}\n"
        "```"
    )
