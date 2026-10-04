import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

from rdkit import Chem, RDLogger

from agent import config

RDLogger.DisableLog("rdApp.*")

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"

_USER_AGENT = ("agent-reference-discovery/1.0"
               + (f" (mailto:{config.CONTACT_EMAIL})" if config.CONTACT_EMAIL else ""))


def _get(url: str, *, timeout: int = 30, retries: int = 3) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:                     # noqa: PERF203
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                return None
        except Exception:                                       # noqa: BLE001
            if attempt == retries - 1:
                return None
        time.sleep(1.5 * (attempt + 1))
    return None


def _get_json(url: str, **kw) -> Optional[dict]:
    raw = _get(url, **kw)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def search_pubmed(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    term = urllib.parse.quote(query)
    ids_doc = _get_json(f"{_EUTILS}/esearch.fcgi?db=pubmed&term={term}"
                        f"&retmax={max_results}&retmode=json")
    ids = (((ids_doc or {}).get("esearchresult") or {}).get("idlist")) or []
    if not ids:
        return []

    time.sleep(0.34)
    raw = _get(f"{_EUTILS}/efetch.fcgi?db=pubmed&id={','.join(ids)}"
               f"&rettype=abstract&retmode=xml")
    if raw is None:
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []

    out = []
    for record in root:
        pmid = (record.findtext("./MedlineCitation/PMID")
                or record.findtext("./BookDocument/PMID") or "")
        title = (record.findtext(".//ArticleTitle")
                 or record.findtext(".//BookTitle") or "")

        parts = []
        for node in record.iter("AbstractText"):
            body = "".join(node.itertext()).strip()
            if not body:
                continue
            label = (node.get("Label") or "").strip()
            parts.append(f"{label}: {body}" if label else body)
        if not parts:
            continue

        out.append({"pmid": pmid, "text": f"{title.strip()}\n" + " ".join(parts)})
    return out


_ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}


def _canonical(smiles: str) -> Optional[str]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    if any(a.GetSymbol() not in _ORGANIC for a in mol.GetAtoms()):
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


_SMILES_KEYS = ("IsomericSMILES", "SMILES", "CanonicalSMILES", "ConnectivitySMILES")


def _from_pubchem(name: str) -> Optional[str]:
    n = urllib.parse.quote(name, safe="")
    time.sleep(0.25)
    doc = _get_json(f"{_PUBCHEM}/compound/name/{n}/property/IsomericSMILES/JSON")
    props = (((doc or {}).get("PropertyTable") or {}).get("Properties")) or []
    if not props:
        return None
    for key in _SMILES_KEYS:
        if props[0].get(key):
            smi = _canonical(props[0][key])
            if smi:
                return smi
    return None


def _from_chembl(name: str) -> Optional[str]:
    n = urllib.parse.quote(name, safe="")
    doc = _get_json(f"{_CHEMBL}/molecule?pref_name__iexact={n}&format=json&limit=1")
    mols = (doc or {}).get("molecules") or []
    if mols:
        struct = mols[0].get("molecule_structures") or {}
        return _canonical(struct.get("canonical_smiles", ""))
    return None


_PAPER_LOCAL = re.compile(
    r"^(compound|cpd|analog|analogue|derivative|example|entry|inhibitor)s?\s*[-_ ]?\d+[a-z]?$"
    r"|^\d+[a-z]?$",
    re.I,
)


_CLASS_NAME = re.compile(
    r"(derivatives?|analogues?|analogs?|hybrids?|congeners?|series|scaffolds?)\s*$", re.I)

_SALT_SUFFIX = re.compile(
    r"\s+(calcium|sodium|potassium|magnesium|lithium|hydrochloride|hcl|dihydrochloride|"
    r"mesylate|maleate|besylate|tosylate|citrate|acetate|sulfate|sulphate|succinate|"
    r"fumarate|tartrate|phosphate|bromide|chloride|monohydrate|dihydrate|hydrate)$", re.I)

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _as_peptide(name: str) -> Optional[str]:
    s = (name or "").strip()
    if not (3 <= len(s) <= 15) or not s.isupper() or not set(s) <= _AMINO_ACIDS:
        return None
    mol = Chem.MolFromSequence(s)
    return Chem.MolToSmiles(mol) if mol else None


def _name_variants(name: str) -> List[str]:
    out: List[str] = []

    def add(candidate: str) -> None:
        candidate = candidate.strip(" -,;:")
        if candidate and candidate not in out:
            out.append(candidate)

    n = (name or "").strip()
    add(n)

    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", n)
    if m:
        add(m.group(1))
        add(m.group(2))
    if "/" in n:
        for part in n.split("/"):
            add(part)
    add(re.sub(r"^\[(\d+)\]-", r"\1-", n))
    add(_SALT_SUFFIX.sub("", n))
    return out


def resolve_name_to_smiles(name: str) -> Optional[str]:
    name = (name or "").strip()
    if not name or _CLASS_NAME.search(name):
        return None

    peptide = _as_peptide(name)
    if peptide:
        return peptide

    for variant in _name_variants(name):
        if _PAPER_LOCAL.match(variant):
            continue
        smi = _from_pubchem(variant) or _from_chembl(variant)
        if smi:
            return smi
    return None
