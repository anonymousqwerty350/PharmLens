import json
from typing import Any, Dict, List

JSON_ONLY = ("Respond with a single JSON object and nothing else.")


def tool_selection_system(task_description: str) -> str:
    return f"""You are a medicinal chemist choosing which molecular measurements are important for: {task_description}

You do not compute anything. You only decide what is worth computing."""


def tool_selection_user(task_description: str, molecular_context: Dict[str, List[str]],
                        fg_catalog: List[Dict[str, str]],
                        desc_catalog: List[Dict[str, str]]) -> str:
    positive_features = "\n".join(f"  + {s}" for s in molecular_context.get("yes_features", []))
    negative_features = "\n".join(f"  - {s}" for s in molecular_context.get("no_features", []))
    fg_list = "\n".join(f"  {d['name']}: {d['description']}" for d in fg_catalog)
    descriptor_list = "\n".join(f"  {d['name']}: {d['description']}" for d in desc_catalog)

    return f"""Task: {task_description}

=== Structural features associated with each class ===
Features associated with the POSITIVE class:
{positive_features or "  (none given)"}

Features associated with the NEGATIVE class:
{negative_features or "  (none given)"}

=== AVAILABLE DESCRIPTORS ===
{descriptor_list}

=== AVAILABLE FUNCTIONAL GROUPS ===
{fg_list}

=== Your task ===
Below are the RDKit tools available. Select ONLY those whose value would actually change your judgement for this task.

There is no target count. Select every tool that is genuinely diagnostic for this task.

Think carefully before writing your answer.

{JSON_ONLY}
```json
{{
  "descriptors": ["exact names from the descriptor list above"],
  "functional_groups": ["exact fr_* names from the functional group list above"],
  "rationale": "Why these, tied to the mechanism of this specific task. Name which features you expect to separate the classes and in which direction."
}}
```"""


def physchem_system(task_description: str) -> str:
    return f"""You are a medicinal chemist assessing the ligand-intrinsic evidence for: {task_description}

You are given measurements only — RDKit computed them, they are exact, and none of them is pre-labelled as good or bad. Interpreting them is your job.

Judge only what the structure itself supports. A molecule whose physicochemistry is favourable can still be ruled out downstream, and that is fine — say what the structure says, no more."""


def physchem_user(smiles: str, values_text: str,
                  molecular_context: Dict[str, List[str]]) -> str:
    positive_features = "\n".join(f"  + {s}" for s in molecular_context.get("yes_features", []))
    negative_features = "\n".join(f"  - {s}" for s in molecular_context.get("no_features", []))

    return f"""Molecule: {smiles}

=== Measured properties ===
{values_text}

=== Structural features associated with each class (domain knowledge) ===
POSITIVE class:
{positive_features or "  (none given)"}

NEGATIVE class:
{negative_features or "  (none given)"}

=== Your task ===
Decide whether the measured structure favours the positive class, the negative class, or neither.

Cite the specific numbers that drive your call — a claim with no number behind it is not evidence.

Reason from two sources, and use both:
1. The domain-knowledge features listed above.
2. Your own medicinal-chemistry knowledge of how these measured properties govern this task.

The list above is a starting point, not a closed set — it does not exhaust what matters here:
- If a measured value is diagnostic for a reason the list never names, say so and use it.
- If your own knowledge contradicts the list, say that too rather than deferring to the list.
- Either way, the number still has to be on the table. An appeal to your own knowledge is held to the same standard as any other claim.

Think carefully before writing your answer.

{JSON_ONLY}
```json
{{
  "verdict": "favors_positive" | "favors_negative" | "neutral",
  "confidence": 0.0-1.0,
  "key_evidence": ["each entry names a measured value and what it implies, e.g. 'TPSA=32.7, far below the ~90 ceiling for passive crossing'"],
  "matched_context": {{"positive": ["which POSITIVE-class features this molecule actually matches"], "negative": ["which NEGATIVE-class features it actually matches"]}},
  "rationale": "2-3 sentences."
}}
```"""


def relation_system(task_description: str) -> str:
    return f"""You are auditing the ligand-protein evidence assembled for: {task_description}

Your primary task for each protein is to evaluate and determine if the provided evidence is sufficient to establish a definitive relationship with the molecule.

Escalate a protein when:
- The relation-evaluator probability is ambiguous.
- No relation-evaluator exists for it.
- The relation-evaluator call and the docking evidence point in opposite directions.

Escalation is not failure; it routes the protein to a structural comparison against ligands whose relationship to it is known."""


def relation_user(smiles: str, proteins_text: str) -> str:
    return f"""Molecule: {smiles}

=== Ligand-protein evidence ===
{proteins_text}

Assess each protein listed above. Return one entry per protein, in the same order.

{JSON_ONLY}
```json
{{
  "assessments": [
    {{
      "protein": "protein name exactly as given",
      "sufficient": true | false,
      "trigger": null | "ambiguity_band" | "no_ml_model" | "ml_docking_conflict",
      "reason": "One sentence. If sufficient, state the relationship the evidence establishes and what settles it. If not, state precisely what is missing."
    }}
  ]
}}
```"""


def interaction_system(task_description: str) -> str:
    return f"""You determine what relationship a molecule has with one protein, by comparing how it binds against how known ligands of that protein bind. Context: {task_description}

You are here because the QSAR evidence could not settle it. Recognition is the CONTACT PATTERN — the specific residues a real ligand must engage, which follow from the protein's binding mechanism.

Otherwise, compare the contacts first and the score second:
- A molecule that reproduces the known actives' contact signature is likely a real ligand even with a mediocre score.
- A molecule that makes none of those contacts is likely not one, however good its score looks.

If the reference set is small or the evidence genuinely does not separate, answer "uncertain"."""


def _contact_frequencies(side: Dict[str, Any]) -> Dict[tuple, int]:
    out: Dict[tuple, int] = {}
    for kind, items in (side.get("top_contacts") or {}).items():
        for item in items:
            residue, _, rest = str(item).partition("(")
            percent = rest.rstrip(")").rstrip("%")
            if residue and percent.isdigit():
                out[(kind, residue.strip())] = int(percent)
    return out


def contact_contrast(profile: Dict[str, Any], limit: int = 18) -> str:
    act = profile.get("actives_reference") or {}
    ina = profile.get("inactives_reference") or {}
    a_freq = _contact_frequencies(act)
    if not a_freq:
        return "  (no recurring contacts among the known actives)"

    header = f"  {'type':<14} {'residue':<10} {'act':>5} {'non-act':>8} {'diff':>6}"
    if not ina.get("n"):
        rows = sorted(a_freq.items(), key=lambda kv: -kv[1])[:limit]
        body = "\n".join(f"  {k:<14} {r:<10} {p:>4}% {'n/a':>8} {'n/a':>6}" for (k, r), p in rows)
        return f"{header}\n{body}\n  (no non-active reference set for this protein)"

    i_freq = _contact_frequencies(ina)
    keys = sorted(set(a_freq) | set(i_freq),
                  key=lambda k: -abs(a_freq.get(k, 0) - i_freq.get(k, 0)))[:limit]
    body = "\n".join(
        f"  {k[0]:<14} {k[1]:<10} {a_freq.get(k, 0):>4}% {i_freq.get(k, 0):>7}% "
        f"{a_freq.get(k, 0) - i_freq.get(k, 0):>+6}"
        for k in keys)
    return f"{header}\n{body}"


def interaction_user(smiles: str, protein: str, relation: str, logic: str,
                   query_text: str, profile: Dict[str, Any]) -> str:
    def fmt(side: str) -> str:
        p = profile.get(side, {})
        if not p.get("n"):
            return "  (no reference data)"
        lines = [f"  n = {p['n']} molecules"]
        if p.get("score_p50") is not None:
            lines.append(f"  docking score  p25/p50/p75 = "
                         f"{p.get('score_p25')} / {p.get('score_p50')} / {p.get('score_p75')} "
                         f"kcal/mol  (over these {p['n']} reference molecules only)")
        contacts = p.get("top_contacts", {})
        if contacts:
            for key, items in contacts.items():
                lines.append(f"  {key}: {', '.join(items)}")
        else:
            lines.append("  (no recurring contacts)")
        return "\n".join(lines)

    return f"""Molecule: {smiles}
Protein: {protein}
Relationship in question: is this molecule a {relation} of {protein}?

=== What recognition means for this protein (domain knowledge) ===
{logic or "(none given)"}

=== How THIS molecule binds ===
{query_text}

=== How KNOWN {relation.upper()}S of {protein} bind (reference set, source: {profile.get('active_source')}) ===
{fmt("actives_reference")}

=== How KNOWN NON-{relation.upper()}S bind (reference set, source: {profile.get('inactive_source')}) ===
{fmt("inactives_reference")}

=== Contact frequency, actives vs non-actives (sorted by the gap) ===
{contact_contrast(profile)}

Percentages are the fraction of reference molecules making that contact.
How separated the two halves are overall: {(profile.get('discriminability') or {}).get('reason', 'not measured')}

=== Your task ===
Two different score baselines appear above, and they are not interchangeable:
- The query's quartile is cut from the whole task train set.
- The reference p25/p50/p75 are cut from these reference molecules alone.

Compare the query's raw kcal/mol against the reference numbers — never its quartile, which says nothing about where it sits among known ligands.

Compare the contact pattern first, then the score. State which known-active contacts this molecule reproduces and which it misses.

{JSON_ONLY}
```json
{{
  "relation": "{relation}" | "non-{relation}" | "uncertain",
  "confidence": 0.0-1.0,
  "evidence": "Name the specific contacts reproduced and missed, and the score relative to the reference distribution. Then say what that pattern means mechanistically."
}}
```"""


def prediction_system(knowledge: Dict[str, Any]) -> str:
    rules = "\n".join(f"• {r}" for r in knowledge.get("pathway_rules", []))
    ctx = knowledge.get("molecular_context", {})
    positive_features = "\n".join(f"  + {s}" for s in ctx.get("yes_features", []))
    negative_features = "\n".join(f"  - {s}" for s in ctx.get("no_features", []))

    return f"""You are an expert pharmacologist making the final call on: {knowledge.get('task_description', '')}

Three specialists have already reported, and their findings are given to you as evidence:

  (1) A physicochemical assessment of the ligand itself.
  (2) Relationship verdict per protein.
  (3) Docking scores and PLIP contacts, for the proteins where that evidence was retained.

Read the evidence honestly.

=== Reasoning through each per-protein finding, then explaining it ===
Each protein has a relationship in question for this task (substrate, agonist, antagonist,
inhibitor, and so on). The relationship, and what it means for this task, is fixed by the
decision rules and the protein's mechanism below — read it off the rules for each protein; it
is not the same across proteins. A finding tells you whether that relationship holds, does not
hold, or is uncertain. Reason through it in three steps.

  Step 0 — Is this protein on the causal path at all? From the decision rules, decide whether
  this protein's mechanism acts on the process the question asks about. A mechanism that
  operates downstream of, or beside, that process does not bear on the answer, and no finding
  about it — either way — changes your call. Say so and move on to the next protein.

  Step 1 — Direction if the relationship holds. From the decision rules and the protein's
  mechanism, determine which task class it points to when its relationship holds. Derive this
  from each protein's own mechanism: two proteins sharing a relationship type can point to
  opposite classes (an efflux substrate and an influx substrate are both substrates yet oppose
  each other).

  Step 2 — Apply the actual finding.
  - Holds → contributes in the Step 1 direction, weighted by how strong the evidence is.
  - Does not hold → this pathway is not operating for this molecule. That REMOVES its
    contribution; it does not create one in the opposite direction. Absence of a mechanism is
    not evidence for the opposite class.
    Then judge how much the absence even tells you: if engagement would have been expected
    from the structure, its absence is informative about that pathway; if molecules like this
    one are non-engagers by default, it tells you almost nothing.
  - Uncertain → little weight.

=== When the panel says little ===
These proteins are a subset of the pathways that determine the answer, not all of them. If
none of them is engaged, that is not a case for the negative class — it means the mechanistic
panel is silent for this molecule, and the call rests on the physicochemical evidence and on
what the structural features below indicate. Do not convert silence into a verdict in either
direction.

In the RATIONALE, for each protein, explain in mechanistic terms whether it bears on this
process at all, which class holding its relationship would point to, and what the actual
finding changes. Give the reason, not the label.

=== Weighing physicochemistry against mechanism ===
The two carry independent weight; neither overrules the other by default. A molecule with an
ideal profile can still be blocked by a transporter, and a molecule with a poor profile can
still get in through a carrier.

=== Mechanistic decision rules for this task ===
{rules or "(none given)"}

=== Structural features associated with each class ===
POSITIVE class:
{positive_features or "  (none given)"}

NEGATIVE class:
{negative_features or "  (none given)"}"""


def prediction_user(smiles: str, physchem_text: str, proteins_text: str,
                    prediction_question: str) -> str:
    return f"""Molecule: {smiles}

=== Evidence (1): Physicochemical assessment ===
{physchem_text}

=== Evidence (2, 3): Ligand-protein relationships, docking and contacts ===
{proteins_text}

=== Question ===
{prediction_question}

=== Your task ===
Write your reasoning as connected prose: what the physicochemistry implies and what each
protein contributes. For each protein, explain in mechanistic terms whether it bears on this
process at all, which class holding its relationship would point to, and what the actual
finding changes — remembering that a relationship which does not hold removes that pathway's
contribution rather than creating one in the opposite direction. Give the reason, not the
label.

Then, on a new line at the very end, give the final answer in exactly the format the question
specifies. Write the exact phrase "final answer" nowhere except this last line."""


def format_json_block(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)
