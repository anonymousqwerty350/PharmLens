import json
import os
from typing import Any, Dict, List, Optional

from agent import config
from agent.registry.proteins import lookup, relation_for
from agent.tools.recognition_gate import fgroups, prompts
from agent.tools.reference_ligands import _CURATED_SUBSTRATES
from agent.utils import parse_json_response

CACHE_DIR = os.path.join(config.CACHE_DIR, "recognition_gate")
_DISCOVERY_DIR = os.path.join(config.CACHE_DIR, "reference_ligands")


def _discovery_cache(canonical: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(_DISCOVERY_DIR, f"{canonical}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _example_ligands(canonical: str) -> List[str]:
    rec = _discovery_cache(canonical)
    if rec:
        names = [r.get("name") for r in rec.get("resolved", []) if r.get("name")]
        if names:
            return names[:12]
        if rec.get("actives"):
            return rec["actives"][:12]
    return [name for name, _ in _CURATED_SUBSTRATES.get(canonical, [])][:12]


def known_actives(canonical: str) -> List[str]:
    rec = _discovery_cache(canonical)
    if rec and rec.get("actives"):
        return rec["actives"]
    return [smi for _, smi in _CURATED_SUBSTRATES.get(canonical, [])]


def decide_recognition(protein: str, relation: Optional[str] = None,
                       force: bool = False) -> Dict[str, Any]:
    entry = lookup(protein)
    canonical = entry["canonical"] if entry else protein
    relation = relation or relation_for(canonical)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{canonical}.json")
    if os.path.exists(cache_path) and not force:
        with open(cache_path) as f:
            return json.load(f)

    examples = _example_ligands(canonical)
    acts = known_actives(canonical)
    llm = config.get_llm("recognition_gate")

    raw = llm.invoke([
        ("system", prompts.recognition_system()),
        ("user", prompts.recognition_user(canonical, relation, examples, fgroups.catalog())),
    ]).content
    parsed = parse_json_response(raw) or {}

    narrow = bool(parsed.get("narrow"))
    groups = fgroups.normalize(parsed.get("required_groups") or []) if narrow else []
    recall = fgroups.groups_recall(groups, acts)

    GROUPS_MIN_RECALL = 0.8
    if narrow and groups and recall is not None and recall >= GROUPS_MIN_RECALL:
        motif_mode = "groups"
    elif narrow and acts:
        motif_mode = "similarity"
        groups = []
    else:
        motif_mode = "none" if narrow else "not_narrow"
        groups = []

    record = {
        "protein": canonical,
        "relation": relation,
        "narrow": narrow,
        "required_groups": groups,
        "groups_self_recall": recall,
        "motif_mode": motif_mode,
        "n_known_actives": len(acts),
        "evidence": parsed.get("evidence"),
        "examples_used": examples,
    }
    with open(cache_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record
