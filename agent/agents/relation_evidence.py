from typing import Any, Dict, List

from agent import config
from agent.registry.proteins import resolve_proteins
from agent.tools import cypreact
from agent.tools import evidence as ev
from agent.tools import ml_predictor


def run(state: Dict[str, Any]) -> Dict[str, Any]:
    task, knowledge, smiles = state["task"], state["knowledge"], state["smiles"]
    ablation = state.get("ablation", {})
    no_gate = ablation.get("no_gate", False)
    no_pce = ablation.get("no_pce", False)
    no_ie = ablation.get("no_ie", False)

    stats = {} if no_ie else ev.load_score_stats(task)
    cache = {} if no_ie else ev.load_cache(smiles, task, state.get("split", "test"))
    proteins: List[Dict[str, Any]] = []

    for p in resolve_proteins(knowledge):
        proba = None
        cyp_substrate = None

        if no_pce:
            pass
        elif p["category"] == "cyp_substrate":
            try:
                cyp_substrate = cypreact.predict(smiles, (p["protein"],))[p["protein"]]
            except Exception as exc:                        # noqa: BLE001
                state.setdefault("errors", []).append(f"{p['protein']} CypReact: {exc}")
        elif p["ml_bundle"]:
            try:
                proba = ml_predictor.predict_proba(p["ml_bundle"], smiles)
            except Exception as exc:                        # noqa: BLE001
                state.setdefault("errors", []).append(f"{p['protein']} ML: {exc}")

        if no_pce:
            band = "pce_disabled"
        elif p["category"] == "cyp_substrate":
            band = {True: "positive", False: "negative", None: "no_model"}[cyp_substrate]
        else:
            band = config.band_of(proba)

        docking_used = True if (no_gate or no_pce) else (band != "negative")

        entry: Dict[str, Any] = {
            "protein": p["protein"], "pdb_id": p["pdb_id"], "category": p["category"],
            "stage": p["stage"], "logic": p["logic"],
            "ml_label": p["ml_label"], "ml_proba": proba, "band": band,
            "docking_used": docking_used, "ie_disabled": no_ie,
            "docking_score": None, "docking_quartile": None, "docking_stats": None,
            "plip": None,
            "relation": None, "relation_source": None,
            "relation_confidence": None, "relation_evidence": None,
        }

        hit = cache.get(p["pdb_id"])
        if hit and hit.get("docking_score") is not None:
            score = hit["docking_score"]
            entry["docking_score"] = score
            entry["docking_quartile"] = ev.quartile_label(score, stats.get(p["pdb_id"], {}))
            entry["docking_stats"] = stats.get(p["pdb_id"]) or None
            entry["plip"] = ev.extract_plip(hit)

        if band in ("positive", "negative"):
            entry["relation"] = (p["ml_label"] if band == "positive"
                                 else f"non-{p['ml_label']}")
            if p["category"] == "cyp_substrate":
                entry["relation_source"] = "cypreact"
                entry["relation_confidence"] = None
            else:
                entry["relation_source"] = "ml_gate"
                entry["relation_confidence"] = (proba if band == "positive"
                                                else round(1.0 - proba, 3))

        proteins.append(entry)

    return {"proteins": proteins}


def format_evidence(proteins: List[Dict[str, Any]], include_plip: bool = True) -> str:
    if not proteins:
        return "  (no proteins in panel)"

    blocks = []
    for p in proteins:
        head = f"[{p['protein']}] ({p['pdb_id']}) — {p['stage'] or p['category']}"
        lines = [head]

        if p.get("band") == "pce_disabled":
            lines.append("  Target-relationship prediction: disabled for this run")
        elif p.get("category") == "cyp_substrate":
            call = {"positive": "R (substrate — this enzyme metabolises it)",
                    "negative": "N (not a substrate of this enzyme)",
                    "no_model": "no call returned"}.get(p["band"], p["band"])
            lines.append(f"  CypReact substrate prediction: {call}")
        elif p["ml_proba"] is not None:
            lines.append(f"  QSAR model: P({p['ml_label']}) = {p['ml_proba']:.3f}  "
                         f"-> {p['band']}")
        else:
            lines.append(f"  QSAR model: none trained for this protein "
                         f"(relationship in question: {p['ml_label'] or p['category']})")

        if p.get("relation"):
            src = {"ml_gate": "settled by the QSAR model",
                   "cypreact": "settled by CypReact (reports R/N, no probability)",
                   "reference_comparison": "settled by comparison against known ligands",
                   "no_evidence": "not settled — no evidence available"}
            conf = p.get("relation_confidence")
            conf_txt = "" if conf is None else f", confidence {conf}"
            lines.append(f"  Relationship: {p['relation']}  "
                         f"({src.get(p['relation_source'], p['relation_source'])}{conf_txt})")
            if p.get("relation_evidence"):
                lines.append(f"    Basis: {p['relation_evidence']}")
        elif p.get("band") == "pce_disabled" and p.get("ie_disabled"):
            lines.append("  Relationship: not assessed — this protein's evidence layers "
                         "are disabled for this run.")
        else:
            lines.append("  Relationship: UNDETERMINED — the evidence did not settle it. "
                         "Do not assume either direction.")

        if p.get("ie_disabled"):
            lines.append("  Docking and contacts: disabled for this run")
        elif p["docking_score"] is None:
            lines.append("  Docking: not available for this molecule")
        elif not p["docking_used"]:
            if p["relation_source"] == "ml_gate" and p["ml_proba"] is not None:
                why = (f"The QSAR model calls this molecule a confident "
                       f"non-{p['ml_label']} (P={p['ml_proba']:.3f})")
            elif p["relation_source"] == "cypreact":
                why = "CypReact calls this molecule a non-substrate of this enzyme"
            else:
                why = (f"Its binding mode does not match known {p['ml_label'] or 'ligand'}s "
                       f"of this protein")
            lines.append(
                f"  Docking: WITHHELD. {why}, so its pose ({p['docking_score']:.2f} kcal/mol) "
                f"describes binding that does not occur. This is negative evidence for this "
                f"mechanism, not missing evidence.")
        else:
            lines.append(f"  Docking score: {p['docking_score']:.2f} kcal/mol"
                         + (f"  [{p['docking_quartile']}]" if p["docking_quartile"] else ""))
            dist = (ev.format_score_stats(p.get("docking_stats"))
                    if p["docking_score"] < 0 else None)
            if dist:
                lines.append(f"    Train-set scores for this protein: {dist}")
            if include_plip:
                lines.append("  Contacts (best pose):")
                lines.append(ev.format_plip(p["plip"]))

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
