from typing import Any, Dict, List

from agent import config
from agent.agents.relation_evidence import format_evidence
from agent.prompts import base_templates as T
from agent.utils import parse_json_response


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    proteins: List[Dict[str, Any]] = state.get("proteins", [])
    knowledge = state["knowledge"]

    forced = {p["protein"] for p in proteins if p["band"] in ("no_model", "ambiguous")}

    if state.get("ablation", {}).get("no_pce"):
        escalated = [p["protein"] for p in proteins if p["docking_score"] is not None]
        for p in proteins:
            if p["protein"] not in escalated and p["relation"] is None:
                p["relation"] = "uncertain"
                p["relation_source"] = "no_evidence"
                p["relation_confidence"] = 0.0
                p["relation_evidence"] = (
                    f"{p['protein']}: target-relationship prediction disabled and no "
                    f"docking pose available.")
        return {"sufficiency": [{"protein": p["protein"], "sufficient": False,
                                 "trigger": "pce_disabled",
                                 "reason": "target-relationship prediction disabled"}
                                for p in proteins],
                "escalated": escalated, "proteins": proteins}

    if state.get("ablation", {}).get("no_refcomp"):
        return {"sufficiency": [{"protein": p["protein"], "sufficient": True,
                                 "trigger": None, "reason": "reference comparison disabled"}
                                for p in proteins],
                "escalated": []}

    llm = config.get_llm("relation")
    raw = llm.invoke([
        ("system", T.relation_system(knowledge.get("task_description", state["task"]))),
        ("user", T.relation_user(state["smiles"], format_evidence(proteins))),
    ]).content
    parsed = parse_json_response(raw)

    assessments = parsed.get("assessments") or []
    by_name = {a.get("protein"): a for a in assessments if isinstance(a, dict)}

    sufficiency, escalated = [], []
    for p in proteins:
        a = by_name.get(p["protein"], {})
        sufficient = bool(a.get("sufficient", True))
        trigger = a.get("trigger")

        if p["protein"] in forced:
            sufficient = False
            trigger = trigger or ("no_ml_model" if p["band"] == "no_model"
                                  else "ambiguity_band")

        sufficiency.append({
            "protein": p["protein"], "sufficient": sufficient, "trigger": trigger,
            "reason": a.get("reason", "no assessment returned"),
        })

        if not sufficient:
            if p["docking_score"] is not None:
                escalated.append(p["protein"])
            elif p["relation"] is None:
                p["relation"] = "uncertain"
                p["relation_source"] = "no_evidence"
                p["relation_confidence"] = 0.0
                p["relation_evidence"] = (
                    f"{p['protein']}: QSAR {p['band']} and no docking pose available.")

    return {"sufficiency": sufficiency, "escalated": escalated,
            "proteins": proteins}
