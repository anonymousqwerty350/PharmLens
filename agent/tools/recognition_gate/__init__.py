from typing import Any, Dict

from agent.tools.recognition_gate import fgroups
from agent.tools.recognition_gate.analysis import decide_recognition, known_actives
from agent.tools.recognition_gate.motif_match import max_similarity

__all__ = ["gate", "decide_recognition"]

SIM_CUTOFF = 0.25


def gate(protein: str, query_smiles: str, task: str = None) -> Dict[str, Any]:
    rec = decide_recognition(protein)

    if not rec.get("narrow"):
        return {"decision": "proceed", "narrow": False, "mode": "not_narrow",
                "has_motif": None, "evidence": rec.get("evidence")}

    mode = rec.get("motif_mode")
    if mode == "groups":
        groups = rec.get("required_groups") or []
        hit = fgroups.has_all_groups(query_smiles, groups)
        detail = "+".join(groups)
    else:
        sim = max_similarity(query_smiles, known_actives(rec["protein"]))
        hit = None if sim is None else (sim >= SIM_CUTOFF)
        mode = "similarity"
        detail = None if sim is None else f"maxTanimoto={sim} (cutoff {SIM_CUTOFF})"

    decision = "terminate_negative" if hit is False else "proceed"
    return {"decision": decision, "narrow": True, "mode": mode, "has_motif": hit,
            "detail": detail, "required_groups": rec.get("required_groups"),
            "evidence": rec.get("evidence")}
