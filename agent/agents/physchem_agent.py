import json
import os
from typing import Any, Dict

from agent import config
from agent.prompts import base_templates as T
from agent.tools import rdkit_tools
from agent.utils import parse_json_response


def select_tools(task: str, knowledge: Dict[str, Any], force: bool = False) -> Dict[str, Any]:
    path = os.path.join(config.TOOL_SELECTION_DIR, f"{task}.json")
    if os.path.exists(path) and not force:
        with open(path) as f:
            return json.load(f)

    catalogs = rdkit_tools.load_catalogs()
    task_description = knowledge.get("task_description", task)
    llm = config.get_llm("tool_selection")
    raw = llm.invoke([
        ("system", T.tool_selection_system(task_description)),
        ("user", T.tool_selection_user(
            task_description,
            knowledge.get("molecular_context", {}),
            catalogs["functional_groups"], catalogs["descriptors"])),
    ]).content

    sel = parse_json_response(raw)
    sel["descriptors"] = [d for d in sel.get("descriptors", []) if d in rdkit_tools.DESC_FUNCS]
    sel["functional_groups"] = [f for f in sel.get("functional_groups", [])
                                if f in rdkit_tools.FG_FUNCS]

    os.makedirs(config.TOOL_SELECTION_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(sel, f, indent=2, ensure_ascii=False)
    return sel


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    task, knowledge, smiles = state["task"], state["knowledge"], state["smiles"]

    if state.get("ablation", {}).get("no_physchem"):
        return {"physchem_values": {}, "physchem_verdict": {"verdict": "disabled"}}

    sel = select_tools(task, knowledge)
    values = rdkit_tools.compute(smiles, sel["functional_groups"], sel["descriptors"])

    if not values.get("valid"):
        return {"physchem_values": values,
                "physchem_verdict": {"verdict": "neutral", "confidence": 0.0,
                                     "rationale": "invalid SMILES"}}

    llm = config.get_llm("physchem")
    raw = llm.invoke([
        ("system", T.physchem_system(knowledge.get("task_description", task))),
        ("user", T.physchem_user(smiles, rdkit_tools.format_for_prompt(values),
                                 knowledge.get("molecular_context", {}))),
    ]).content

    return {"physchem_values": values, "physchem_verdict": parse_json_response(raw)}


def format_verdict(state: Dict[str, Any]) -> str:
    v = state.get("physchem_verdict") or {}
    if v.get("verdict") == "disabled":
        return "  (physicochemical evidence disabled for this run)"

    lines = [f"  Assessment: {v.get('verdict', 'unknown')} "
             f"(confidence {v.get('confidence', '?')})"]
    for e in v.get("key_evidence", []):
        lines.append(f"    - {e}")
    if v.get("rationale"):
        lines.append(f"  Rationale: {v['rationale']}")
    lines.append("")
    lines.append("  Measured values:")
    lines.append(rdkit_tools.format_for_prompt(state.get("physchem_values", {})))
    return "\n".join(lines)
