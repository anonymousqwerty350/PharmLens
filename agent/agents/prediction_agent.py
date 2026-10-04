import re
from typing import Any, Dict, Optional

from agent import config
from agent.agents import physchem_agent, relation_evidence
from agent.prompts import base_templates as T


def _answer_words(knowledge: Dict[str, Any]) -> list:
    q = knowledge.get("prediction_question", "")
    return re.findall(r"Final answer:\s*([A-Za-z][\w-]*)", q)


def _positive_word(knowledge: Dict[str, Any]) -> str:
    for h in _answer_words(knowledge):
        if not h.lower().startswith("non"):
            return h
    return "Yes"


def parse_answer(text: str, knowledge: Dict[str, Any]) -> Optional[int]:
    if not text:
        return None
    pos = _positive_word(knowledge)

    m = re.findall(r"final answer\s*[::]\s*([^\n\.]+)", text, re.IGNORECASE)
    candidate = m[-1].strip() if m else text[:200]
    c = candidate.lower()

    if re.search(rf"non[-\s]?{re.escape(pos.lower())}", c):
        return 0
    if re.search(rf"\b{re.escape(pos.lower())}\b", c):
        return 1

    if re.search(r"\bnon[-\s]?tox", c) or re.search(r"\bno\b", c):
        return 0
    if re.search(r"\btox", c) or re.search(r"\byes\b", c):
        return 1
    return None


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    knowledge = state["knowledge"]

    if state.get("ablation", {}).get("no_knowledge"):
        knowledge = {**knowledge, "pathway_rules": [], "molecular_context": {}}

    physchem_text = physchem_agent.format_verdict(state)
    proteins_text = relation_evidence.format_evidence(state.get("proteins", []))
    llm = config.get_llm("prediction")

    messages = [
        ("system", T.prediction_system(knowledge)),
        ("user", T.prediction_user(
            state["smiles"], physchem_text, proteins_text,
            state["knowledge"].get("prediction_question", "Is the answer YES or NO?"))),
    ]

    raw = llm.invoke(messages).content

    pred = parse_answer(raw, state["knowledge"])
    return {"prediction": -1 if pred is None else pred, "prediction_text": raw}
