from typing import List, Optional

from rdkit import DataStructs, RDLogger

from agent.tools.reference_ligands import _fp

RDLogger.DisableLog("rdApp.*")


def max_similarity(query_smiles: str, ref_smiles: List[str]) -> Optional[float]:
    qfp = _fp(str(query_smiles))
    if qfp is None:
        return None
    sims = [DataStructs.TanimotoSimilarity(qfp, f)
            for f in (_fp(str(s)) for s in ref_smiles) if f is not None]
    return round(max(sims), 2) if sims else None
