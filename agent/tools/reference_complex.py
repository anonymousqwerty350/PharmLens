import json
import os
from typing import Any, Dict, Optional

from agent import config

MIN_ACTIVES = 10
MAX_CONTACT_OVERLAP = 0.8
MIN_SCORE_GAP = 0.5


def _residues(side: Dict[str, Any]) -> set:
    out = set()
    for contacts in (side.get("top_contacts") or {}).values():
        for c in contacts:
            out.add(c.split("(")[0])
    return out


def discriminability(profile: Dict[str, Any]) -> Dict[str, Any]:
    act, ina = profile.get("actives_reference", {}), profile.get("inactives_reference", {})
    n_act, n_ina = act.get("n", 0), ina.get("n", 0)

    if n_act < MIN_ACTIVES:
        return {"usable": False, "reason":
                f"only {n_act} known actives docked (need >= {MIN_ACTIVES})"}

    ra, ri = _residues(act), _residues(ina)
    overlap = len(ra & ri) / len(ra | ri) if (ra | ri) else 1.0

    sa, si = act.get("score_p50"), ina.get("score_p50")
    gap = (si - sa) if (sa is not None and si is not None) else None

    if n_ina == 0:
        reason = "no inactive reference set; judge on the actives' signature alone"
    elif overlap >= MAX_CONTACT_OVERLAP and not (gap is not None and gap >= MIN_SCORE_GAP):
        reason = (f"actives and inactives touch largely the same residues (contact overlap "
                  f"{overlap:.0%}, actives out-bind inactives by "
                  f"{0.0 if gap is None else gap:.2f} kcal/mol). Read the per-residue "
                  f"frequencies rather than the residue lists; if those match too, binding "
                  f"mode carries no information for this protein.")
    else:
        reason = "actives and inactives are separable"

    return {"usable": True, "contact_overlap": round(overlap, 2),
            "score_gap": None if gap is None else round(gap, 2), "reason": reason}


def load(pdb_id: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(config.REFERENCE_DIR, f"{pdb_id}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        profile = json.load(f)
    profile["discriminability"] = discriminability(profile)
    return profile
