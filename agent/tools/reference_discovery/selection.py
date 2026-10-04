from typing import Callable, List, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

MAX_POOL = 1500

CLUSTER_CUT = 0.65

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

MAX_ATOMS = 100


def too_large(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return Chem.AddHs(mol).GetNumAtoms() > MAX_ATOMS


def representatives(smiles: List[str], k: int,
                    priority: Optional[Callable[[int], float]] = None) -> List[int]:
    pool = smiles[:MAX_POOL]
    mols = [Chem.MolFromSmiles(s) for s in pool]
    keep = [i for i, m in enumerate(mols) if m is not None]
    if len(keep) <= k:
        return keep

    fps = [_MORGAN.GetFingerprint(mols[i]) for i in keep]

    dists: List[float] = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)

    clusters = Butina.ClusterData(dists, len(fps), CLUSTER_CUT, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)

    def speaker(cluster) -> int:
        if priority is None:
            return keep[cluster[0]]
        return keep[max(cluster, key=lambda j: priority(keep[j]))]

    picked = [speaker(c) for c in clusters[:k]]

    if len(picked) < k:
        chosen = set(picked)
        for cluster in clusters:
            ordered = sorted(cluster, key=lambda j: -priority(keep[j])) if priority else cluster
            for j in ordered:
                if len(picked) >= k:
                    break
                if keep[j] not in chosen:
                    chosen.add(keep[j])
                    picked.append(keep[j])
            if len(picked) >= k:
                break
    return picked[:k]


def property_vector(smiles: str) -> Optional[tuple]:
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        return None
    return (Descriptors.MolWt(mol), Crippen.MolLogP(mol))


def property_matcher(reference: List[str]) -> Callable[[str], float]:
    vectors = [v for v in (property_vector(s) for s in reference) if v]
    if not vectors:
        return lambda _s: 0.0

    n = len(vectors)
    mu = [sum(v[i] for v in vectors) / n for i in range(2)]
    sd = []
    for i in range(2):
        var = sum((v[i] - mu[i]) ** 2 for v in vectors) / n
        sd.append(var ** 0.5 or (abs(mu[i]) or 1.0))

    def score(smiles: str) -> float:
        vec = property_vector(smiles)
        if vec is None:
            return float("-inf")
        distance = sum(((vec[i] - mu[i]) / sd[i]) ** 2 for i in range(2)) ** 0.5
        return -distance

    return score
