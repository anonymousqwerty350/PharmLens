import json
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from agent import config
from agent.agents import (interaction_agent, physchem_agent, prediction_agent,
                               relation_agent, relation_evidence)
from agent.state import AgentState


def _route_after_relation(state: AgentState) -> str:
    return "interaction" if state.get("escalated") else "prediction"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("physchem", physchem_agent.run)
    g.add_node("relation_evidence", relation_evidence.run)
    g.add_node("relation", relation_agent.run)
    g.add_node("interaction", interaction_agent.run)
    g.add_node("prediction", prediction_agent.run)

    g.set_entry_point("physchem")
    g.add_edge("physchem", "relation_evidence")
    g.add_edge("relation_evidence", "relation")
    g.add_conditional_edges("relation", _route_after_relation,
                            {"interaction": "interaction",
                             "prediction": "prediction"})
    g.add_edge("interaction", "prediction")
    g.add_edge("prediction", END)
    return g.compile()


def load_knowledge(task: str) -> Dict[str, Any]:
    with open(config.TASKS[task]["knowledge"]) as f:
        return json.load(f)


_ABLATION_IMPLIES = {
    "no_lce": ("no_physchem",),
    "no_ie":  ("no_refcomp",),
}


def expand_ablation(ablation: Optional[Dict[str, bool]]) -> Dict[str, bool]:
    out = dict(ablation or {})
    for layer, implied in _ABLATION_IMPLIES.items():
        if out.get(layer):
            for flag in implied:
                out.setdefault(flag, True)
    return out


def run_one(graph, task: str, knowledge: Dict[str, Any], smiles: str,
            label: Optional[int] = None,
            ablation: Optional[Dict[str, bool]] = None,
            split: str = "test") -> AgentState:
    init: AgentState = {
        "smiles": smiles, "task": task, "knowledge": knowledge, "label": label,
        "split": split, "ablation": expand_ablation(ablation), "errors": [],
    }
    return graph.invoke(init)


def trace(state: AgentState) -> Dict[str, Any]:
    return {
        "smiles": state.get("smiles"),
        "label": state.get("label"),
        "prediction": state.get("prediction"),
        "physchem_verdict": state.get("physchem_verdict"),
        "physchem_values": state.get("physchem_values"),
        "proteins": [
            {k: v for k, v in p.items() if k != "logic"}
            for p in state.get("proteins", [])
        ],
        "sufficiency": state.get("sufficiency"),
        "escalated": state.get("escalated"),
        "prediction_text": state.get("prediction_text"),
        "errors": state.get("errors"),
    }
