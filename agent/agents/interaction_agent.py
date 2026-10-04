from typing import Any, Dict, List

from agent import config
from agent.prompts import base_templates as T
from agent.tools import evidence as ev
from agent.tools import reference_complex
from agent.utils import parse_json_response


def _query_text(p: Dict[str, Any]) -> str:
    lines = [f"  docking score: {p['docking_score']:.2f} kcal/mol"
             + (f"  [{p['docking_quartile']}]" if p.get("docking_quartile") else "")]
    dist = (ev.format_score_stats(p.get("docking_stats"))
            if p["docking_score"] < 0 else None)
    if dist:
        lines.append(f"  quartile is against ALL task train-set molecules docked into "
                     f"this protein: {dist}")
    lines.append("  contacts:")
    lines.append(ev.format_plip(p.get("plip")))
    return "\n".join(lines)


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    escalated = set(state.get("escalated", []))
    proteins: List[Dict[str, Any]] = state.get("proteins", [])
    if not escalated:
        return {"proteins": proteins}

    knowledge = state["knowledge"]
    task_desc = knowledge.get("task_description", state["task"])
    llm = config.get_llm("interaction")

    use_gate = not state.get("ablation", {}).get("no_recognition_gate")

    for p in proteins:
        if p["protein"] not in escalated:
            continue

        if use_gate:
            from agent.tools import recognition_gate
            g = recognition_gate.gate(p["protein"], state["smiles"], state.get("task"))
            if g["decision"] == "terminate_negative":
                rel = p.get("ml_label") or "substrate"
                p["relation"] = f"non-{rel}"
                p["relation_source"] = "recognition_gate"
                p["relation_confidence"] = 0.9
                p["relation_evidence"] = (
                    f"recognition_gate: {p['protein']} recognises a narrow class requiring "
                    f"{g.get('detail')}, which this molecule lacks — not a {rel}.")
                p["docking_used"] = False
                continue

        profile = reference_complex.load(p["pdb_id"])
        usable = (profile or {}).get("discriminability", {})

        if profile is None or not usable.get("usable"):
            why = usable.get("reason") if profile else "no reference profile built"
            p["relation"] = "uncertain"
            p["relation_source"] = "reference_comparison"
            p["relation_confidence"] = 0.0
            p["relation_evidence"] = (
                f"Binding mode cannot settle this relationship for {p['protein']}: {why}")
            continue

        relation = p["ml_label"] or profile.get("relation") or "binder"
        raw = llm.invoke([
            ("system", T.interaction_system(task_desc)),
            ("user", T.interaction_user(state["smiles"], p["protein"], relation,
                                      p.get("logic", ""), _query_text(p), profile)),
        ]).content
        parsed = parse_json_response(raw)

        p["relation"] = parsed.get("relation", "uncertain")
        p["relation_source"] = "reference_comparison"
        p["relation_confidence"] = parsed.get("confidence")
        p["relation_evidence"] = parsed.get("evidence")

        if str(p["relation"]).startswith("non-"):
            p["docking_used"] = False

    return {"proteins": proteins}
