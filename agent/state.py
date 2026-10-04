from typing import Any, Dict, List, Optional, TypedDict


class ProteinEvidence(TypedDict, total=False):
    protein: str
    pdb_id: str
    category: str
    logic: str
    stage: str

    ml_label: Optional[str]
    ml_proba: Optional[float]
    band: str

    docking_used: bool
    docking_score: Optional[float]
    docking_quartile: Optional[str]
    docking_stats: Optional[Dict[str, Any]]
    plip: Optional[Dict[str, List[str]]]

    relation: Optional[str]
    relation_source: str
    relation_confidence: Optional[float]
    relation_evidence: Optional[str]


class AgentState(TypedDict, total=False):
    smiles: str
    task: str
    knowledge: Dict[str, Any]
    label: Optional[int]
    split: str

    physchem_values: Dict[str, Any]
    physchem_verdict: Dict[str, Any]

    proteins: List[ProteinEvidence]

    sufficiency: List[Dict[str, Any]]
    escalated: List[str]

    prediction: Optional[int]
    prediction_text: str

    ablation: Dict[str, bool]
    errors: List[str]
