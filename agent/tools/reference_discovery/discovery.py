import csv
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from agent import config
from agent.registry.proteins import lookup, relation_for
from agent.tools.reference_discovery import dataset_actives
from agent.tools.reference_discovery import literature
from agent.tools.reference_discovery import prompts
from agent.tools.reference_discovery import selection
from agent.tools.reference_ligands import tasks_using, K
from agent.utils import parse_json_response

CACHE_DIR = os.path.join(config.CACHE_DIR, "reference_ligands")

N_FLOOR = 10

_QUERIES = {
    "substrate":  ['"{p}"[tiab] AND (substrate[tiab] OR transported[tiab] OR uptake[tiab])',
                   '"{p}"[tiab] AND (prodrug[tiab] OR "drug delivery"[tiab])',
                   '"{p}"[tiab] AND substrate[tiab] AND review[pt]',
                   '"{p}"[tiab] AND (Km[tiab] OR affinity[tiab] OR kinetics[tiab]) '
                   'AND (substrate[tiab] OR transport[tiab])'],
    "inhibitor":  ['"{p}"[tiab] AND (inhibitor[tiab] OR IC50[tiab] OR Ki[tiab])',
                   '"{p}"[tiab] AND (blocker[tiab] OR blockade[tiab] OR inhibition[tiab])',
                   '"{p}"[tiab] AND inhibitor[tiab] AND review[pt]',
                   '"{p}"[tiab] AND (potency[tiab] OR "structure-activity"[tiab]) '
                   'AND inhibitor[tiab]'],
    "antagonist": ['"{p}"[tiab] AND (antagonist[tiab] OR antagonism[tiab])',
                   '"{p}"[tiab] AND antagonist[tiab] AND (IC50[tiab] OR Ki[tiab])',
                   '"{p}"[tiab] AND antagonist[tiab] AND review[pt]',
                   '"{p}"[tiab] AND (potency[tiab] OR "structure-activity"[tiab]) '
                   'AND antagonist[tiab]'],
    "agonist":    ['"{p}"[tiab] AND (agonist[tiab] OR activation[tiab])',
                   '"{p}"[tiab] AND agonist[tiab] AND (EC50[tiab] OR "functional response"[tiab])',
                   '"{p}"[tiab] AND agonist[tiab] AND review[pt]',
                   '"{p}"[tiab] AND (potency[tiab] OR "structure-activity"[tiab]) '
                   'AND agonist[tiab]'],
}


def _search_names(protein: str) -> List[str]:
    from agent.registry.synonyms import synonyms_for

    return synonyms_for(protein, _pdb_for(protein))

_NEG_PHRASES = ('("without affecting"[tiab] OR unaffected[tiab] OR "no effect"[tiab] '
                'OR inactive[tiab] OR "showed no"[tiab])')
_NEG_QUERIES = {
    rel: [f'"{{p}}"[tiab] AND {_NEG_PHRASES}',
          f'"{{p}}"[tiab] AND (selectivity[tiab] OR selective[tiab]) AND '
          f'(inactive[tiab] OR "no effect"[tiab] OR unaffected[tiab])']
    for rel in ("substrate", "inhibitor", "antagonist", "agonist")
}


def _gather_abstracts(protein: str, relation: str, per_query: int = 8,
                      cap: int = 32, queries: Optional[Dict[str, List[str]]] = None
                      ) -> List[Dict[str, str]]:
    names = _search_names(protein)
    templates = (queries or _QUERIES).get(relation, ['"{p}"[tiab] AND ligand[tiab]'])
    quota = max(1, cap // len(templates))

    seen: set = set()
    picked: List[List[Dict[str, str]]] = [[] for _ in templates]

    def run(idx: int, limit: int) -> None:
        for name in names:
            if len(picked[idx]) >= limit:
                return
            for hit in literature.search_pubmed(templates[idx].format(p=name), per_query):
                pmid = hit.get("pmid") or hit.get("text", "")[:40]
                if pmid in seen:
                    continue
                seen.add(pmid)
                picked[idx].append(hit)
                if len(picked[idx]) >= limit:
                    return

    for i in range(len(templates)):
        run(i, quota)

    spare = cap - sum(len(p) for p in picked)
    for i in range(len(templates)):
        if spare <= 0:
            break
        before = len(picked[i])
        run(i, before + spare)
        spare -= len(picked[i]) - before

    return [hit for group in picked for hit in group][:cap]


def _extract_names(protein: str, relation: str,
                   abstracts: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not abstracts:
        return []

    llm = config.get_llm("reference_discovery")
    raw = llm.invoke([
        ("system", prompts.positive_system(f"reference {relation}s of {protein}")),
        ("user", prompts.positive_user(protein, relation, abstracts,
                                       n_target=K, n_floor=N_FLOOR)),
    ]).content
    parsed = parse_json_response(raw)
    ligands = parsed.get("ligands") if isinstance(parsed, dict) else None
    if not isinstance(ligands, list):
        return []

    corpus_norm = " ".join(" ".join(a.get("text", "") for a in abstracts).lower().split())

    def in_corpus(text: str, floor: int = 4) -> bool:
        t = " ".join(str(text).lower().split())
        if len(t) >= floor and t in corpus_norm:
            return True
        head = t.split("(")[0].split("/")[0].strip()
        return len(head) >= floor and head in corpus_norm

    out = []
    for g in ligands:
        if not isinstance(g, dict):
            continue
        name = str(g.get("name", "")).strip()
        if not name or not in_corpus(name):
            continue
        quote = str(g.get("evidence", "")).strip()
        g["evidence_verified"] = bool(quote) and in_corpus(quote[:60], floor=20)
        out.append(g)
    return out


def _resolve(names: List[Dict[str, str]]) -> Dict[str, Any]:
    smiles, resolved, unresolved, oversized, seen = [], [], [], [], set()
    for g in names:
        name = str(g["name"]).strip()
        smi = literature.resolve_name_to_smiles(name)
        if smi is None:
            unresolved.append(name)
            continue
        if smi in seen:
            continue
        seen.add(smi)
        if selection.too_large(smi):
            oversized.append(name)
            continue
        smiles.append(smi)
        rec = {"name": name, "smiles": smi, "source": "llm_discovery",
               "class": g.get("class", ""), "source_pmid": g.get("source_pmid", "")}
        if g.get("evidence"):
            rec["evidence"] = str(g["evidence"]).strip()
        if "evidence_verified" in g:
            rec["evidence_verified"] = bool(g["evidence_verified"])
        resolved.append(rec)
    return {"smiles": smiles, "resolved": resolved, "unresolved": unresolved,
            "oversized": oversized}


def _top_up(actives: List[str], resolved: List[Dict[str, Any]], canonical: str,
            relation: str) -> Dict[str, Any]:
    have = set(actives)
    merged_actives = list(actives)
    merged_resolved = list(resolved)
    counts = Counter({"llm_discovery": len(actives)})

    def absorb(smiles: List[str], records: List[Dict[str, Any]], tag: str) -> None:
        room = K - len(merged_actives)
        if room <= 0:
            return
        by_smiles = {r["smiles"]: r for r in records}
        for smi in _without_test_overlap(smiles, canonical):
            if room <= 0:
                break
            if smi in have:
                continue
            have.add(smi)
            merged_actives.append(smi)
            if smi in by_smiles:
                merged_resolved.append(by_smiles[smi])
            counts[tag] += 1
            room -= 1

    if len(merged_actives) < K and dataset_actives.is_allowed(canonical):
        got = dataset_actives.dataset_actives(canonical, limit=K + len(merged_actives))
        absorb(got["actives"], got["resolved"], "ml_dataset")

    return {"actives": merged_actives, "resolved": merged_resolved,
            "active_sources": {k: v for k, v in counts.items() if v}}


def _diagnostics(abstracts: List[Dict[str, str]], proposed: List[Dict[str, str]],
                 resolved: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes = [str(r.get("class", "")).strip().lower() for r in resolved]
    classes = [c for c in classes if c]
    top = max(Counter(classes).values()) if classes else 0
    return {
        "n_abstracts": len(abstracts),
        "n_proposed": len(proposed),
        "n_evidence_verified": sum(1 for g in proposed if g.get("evidence_verified")),
        "class_concentration": round(top / len(classes), 2) if classes else None,
        "n_classes": len(set(classes)),
    }


_TEST_CANON: Dict[str, set] = {}


def _test_canonical_smiles(task: str) -> set:
    if task not in _TEST_CANON:
        t = config.TASKS[task]
        path = os.path.join(t["dataset_dir"], "test.csv")
        out = set()
        if os.path.exists(path):
            with open(path) as f:
                for row in csv.DictReader(f):
                    smi = literature._canonical(row.get(t["smiles_col"], ""))
                    if smi:
                        out.add(smi)
        _TEST_CANON[task] = out
    return _TEST_CANON[task]


def _pdb_for(canonical: str) -> Optional[str]:
    from agent.registry.proteins import lookup as _lookup, resolve_proteins

    for task in tasks_using(canonical):
        with open(config.TASKS[task]["knowledge"]) as f:
            for p in resolve_proteins(json.load(f)):
                hit = _lookup(p["protein"])
                name = hit["canonical"] if hit else p["protein"]
                if name == canonical and p.get("pdb_id"):
                    return p["pdb_id"]
    return None


def _without_test_overlap(smiles_list: List[str], canonical: str) -> List[str]:
    test = set()
    for task in tasks_using(canonical):
        test |= _test_canonical_smiles(task)
    return [s for s in smiles_list if s not in test]


def discover_positive(protein: str, relation: Optional[str] = None,
                     force: bool = False) -> Dict[str, Any]:
    entry = lookup(protein)
    canonical = entry["canonical"] if entry else protein
    relation = relation or relation_for(canonical)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{canonical}.json")
    previous = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            previous = json.load(f)
        if not force:
            return previous

    abstracts = _gather_abstracts(canonical, relation)
    names = _extract_names(canonical, relation, abstracts)
    res = _resolve(names)
    actives = _without_test_overlap(res["smiles"], canonical)[:K]
    filled = _top_up(actives, res["resolved"], canonical, relation)

    record = {
        "protein": canonical,
        "relation": relation,
        "source": "+".join(filled["active_sources"]) or "llm_discovery",
        "active_sources": filled["active_sources"],
        "actives": filled["actives"],
        "inactives": [],
        "resolved": filled["resolved"],
        "unresolved": res["unresolved"],
        "oversized": res["oversized"],
        "pmids": sorted({a["pmid"] for a in abstracts if a.get("pmid")}),
        **_diagnostics(abstracts, names, res["resolved"]),
    }

    if "inactive_sources" in previous:
        kept = [s for s in previous.get("inactives", []) if s not in set(filled["actives"])]
        record["inactives"] = kept
        record["inactives_resolved"] = [r for r in previous.get("inactives_resolved", [])
                                        if r.get("smiles") in set(kept)]
        for field in ("inactive_source", "inactive_sources", "neg_pmids"):
            if field in previous:
                record[field] = previous[field]
    with open(cache_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def discover_negative(protein: str, force: bool = False) -> Dict[str, Any]:
    entry = lookup(protein)
    canonical = entry["canonical"] if entry else protein

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{canonical}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            record = json.load(f)
    else:
        record = discover_positive(canonical)

    if "inactive_sources" in record and not force:
        return record

    relation = record.get("relation") or relation_for(canonical)

    abstracts = _gather_abstracts(canonical, relation, queries=_NEG_QUERIES)

    names: List[Dict[str, str]] = []
    if abstracts:
        llm = config.get_llm("reference_discovery")
        raw = llm.invoke([
            ("system", prompts.negative_system(canonical, relation)),
            ("user", prompts.negative_user(canonical, relation, abstracts, n_target=K)),
        ]).content
        parsed = parse_json_response(raw)
        proposed = parsed.get("ligands") if isinstance(parsed, dict) else None
        names = [g for g in (proposed or [])
                 if isinstance(g, dict) and str(g.get("name", "")).strip()
                 and str(g.get("evidence", "")).strip()]

    res = _resolve(names)
    actives = record.get("actives") or []
    actives_set = set(actives)
    from_literature = [s for s in res["smiles"] if s not in actives_set]

    matcher = selection.property_matcher(actives)
    from_literature = _without_test_overlap(from_literature, canonical)
    from_literature.sort(key=matcher, reverse=True)
    from_literature = from_literature[:K]

    resolved = [r for r in res["resolved"] if r["smiles"] in set(from_literature)]
    for entry_record in resolved:
        entry_record["source"] = "llm_discovery"
    counts = Counter({"llm_discovery": len(from_literature)} if from_literature else {})

    inactives = list(from_literature)
    if len(inactives) < K and dataset_actives.is_allowed(canonical):
        got = dataset_actives.dataset_inactives(canonical, limit=K + len(inactives),
                                                actives=actives)
        have = set(inactives) | actives_set
        by_smiles = {r["smiles"]: r for r in got["resolved"]}
        for smi in _without_test_overlap(got["inactives"], canonical):
            if len(inactives) >= K:
                break
            if smi in have:
                continue
            have.add(smi)
            inactives.append(smi)
            if smi in by_smiles:
                resolved.append(by_smiles[smi])
            counts["ml_dataset"] += 1

    record["inactives"] = inactives
    record["inactives_resolved"] = resolved
    record["inactive_source"] = "+".join(counts) or None
    record["inactive_sources"] = dict(counts)
    record["neg_pmids"] = sorted({a["pmid"] for a in abstracts if a.get("pmid")})
    record["neg_oversized"] = res["oversized"]

    with open(cache_path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def as_source(protein: str) -> Dict[str, List[str]]:
    rec = discover_positive(protein)
    return {"actives": rec.get("actives", []), "inactives": rec.get("inactives", [])}
