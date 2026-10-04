import argparse
import json
import os

from agent import config
from agent.registry.proteins import panel_proteins, resolve_proteins
from agent.tools.reference_discovery import discover_negative, discover_positive

CACHE_DIR = os.path.join(config.CACHE_DIR, "reference_ligands")

_SKIP_CATEGORIES = {"unknown", "cyp_substrate"}


def run_positive(task: str, force: bool = False):
    knowledge = json.load(open(config.TASKS[task]["knowledge"]))
    seen = set()

    for p in resolve_proteins(knowledge):
        name, category = p["protein"], p["category"]
        if name in seen:
            continue
        seen.add(name)

        if category in _SKIP_CATEGORIES:
            print(f"[{task}] {name} — {category} → skip (not a reference-comparison target)")
            continue

        print(f"\n[{task}] {name} ligand discovery (relation={p.get('ml_label')})")
        rec = discover_positive(name, force=force)
        n_act = len(rec.get("actives", []))
        n_unres = len(rec.get("unresolved", []))
        conc = rec.get("class_concentration")
        print(f"  actives={n_act}  relation={rec.get('relation')}  "
              f"resolve failed={n_unres}  abstracts={rec.get('n_abstracts')} "
              f"pmids={len(rec.get('pmids', []))}  "
              f"classes={rec.get('n_classes')}  top share={conc}")
        for r in rec.get("resolved", []):
            print(f"    - {r['name']:<28} [{r.get('class', '')}]  {r['smiles']}")
            if r.get("evidence"):
                print(f"        └ PMID {r.get('source_pmid', '?')}: \"{r['evidence'][:110]}\"")
        if rec.get("unresolved"):
            print(f"  ⚠ DB lookup failed (excluded): {', '.join(rec['unresolved'])}")
        if rec.get("oversized"):
            print(f"  ⚠ atom count exceeded (cannot dock, excluded): {', '.join(rec['oversized'])}")
        if conc is not None and conc > 0.5:
            print(f"  ⚠ class skew {int(conc * 100)}% — fine if this protein recognizes a single class, "
                  f"but a collapse if cross-class recognition is the point. Check it by eye")
        if n_act and n_act < 10:
            print(f"  ⚠ actives {n_act} — below the reference_complex floor (10), profile rejected")


def run_negative(proteins, force: bool = False):
    rows = []
    for name in proteins:
        rec = discover_negative(name, force=force)
        inact = rec.get("inactives", [])
        resolved = rec.get("inactives_resolved", [])
        rows.append((name, rec.get("relation", "?"), len(rec.get("actives", [])),
                     len(inact), len(rec.get("neg_pmids", []))))
        print(f"\n[{name}] relation={rec.get('relation')}  "
              f"actives={len(rec.get('actives', []))}  inactives={len(inact)}  "
              f"neg_pmids={len(rec.get('neg_pmids', []))}")
        for r in resolved[:15]:
            print(f"    - {r['name']:<28} {r.get('evidence', '')[:90]}")
        if rec.get("neg_oversized"):
            print(f"    ⚠ atom count exceeded (cannot dock, excluded): {', '.join(rec['neg_oversized'])}")
        if not inact:
            print("    (no measured negatives in the literature -> actives-only profile)")

    print(f"\n{'protein':<10}{'relation':<12}{'actives':>8}{'inactives':>10}{'neg_pmids':>11}")
    for r in rows:
        print(f"{r[0]:<10}{r[1]:<12}{r[2]:>8}{r[3]:>10}{r[4]:>11}")
    empty = [r[0] for r in rows if r[3] == 0]
    print(f"\nproteins with zero negatives {len(empty)}/{len(rows)}: {', '.join(empty) or 'none'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="side", required=True)

    pos = sub.add_parser("positive")
    pos.add_argument("--tasks", nargs="+", default=None)
    pos.add_argument("--all", action="store_true",
                     help="every registered task (config.TASKS)")
    pos.add_argument("--force", action="store_true", help="ignore the cache and re-discover")

    neg = sub.add_parser("negative")
    neg.add_argument("--proteins", nargs="*", default=None)
    neg.add_argument("--all", action="store_true",
                     help="every panel protein of the registered tasks (registry.panel_proteins)")
    neg.add_argument("--force", action="store_true",
                     help="re-discover negatives even where they are already filled in")

    args = ap.parse_args()

    if args.side == "positive":
        if args.all:
            tasks = list(config.TASKS)
        elif args.tasks:
            tasks = args.tasks
        else:
            pos.error("specify either --all or --tasks")

        print(f"[positive] {len(tasks)} task(s): {', '.join(tasks)}", flush=True)
        for t in tasks:
            run_positive(t, force=args.force)
    else:
        if args.all:
            names = panel_proteins()
        elif args.proteins:
            names = args.proteins
        elif os.path.isdir(CACHE_DIR):
            names = sorted(f[:-5] for f in os.listdir(CACHE_DIR) if f.endswith(".json"))
        else:
            neg.error(f"specify --all or --proteins (no cache: {CACHE_DIR})")

        print(f"[negative] {len(names)} protein(s)", flush=True)
        run_negative(names, force=args.force)
