import csv
import functools
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

from agent import config

BUNDLE_DIR = os.path.join(config.PKG_ROOT, "tools", "jars", "CypReactBundle")
JAR = os.path.join(BUNDLE_DIR, "cypreact.jar")
JAVA = os.environ.get("CYPREACT_JAVA") or shutil.which("java")

SUPPORTED = {
    "CYP1A2": "1A2", "CYP2A6": "2A6", "CYP2B6": "2B6", "CYP2C8": "2C8",
    "CYP2C9": "2C9", "CYP2C19": "2C19", "CYP2D6": "2D6", "CYP2E1": "2E1",
    "CYP3A4": "3A4",
}

TIMEOUT = 180


def available() -> bool:
    return bool(JAVA) and os.path.exists(JAR) and os.path.exists(JAVA)


@functools.lru_cache(maxsize=4096)
def predict(smiles: str, cyps: tuple) -> Dict[str, Optional[bool]]:
    codes = [SUPPORTED[c] for c in cyps if c in SUPPORTED]
    if not codes or not available():
        return {c: None for c in cyps}

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        out_path = tmp.name
    try:
        proc = subprocess.run(
            [JAVA, "-jar", JAR, BUNDLE_DIR + "/", f"SMILES={smiles}",
             out_path, ",".join(codes)],
            capture_output=True, text=True, timeout=TIMEOUT, cwd=BUNDLE_DIR,
        )
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            return {c: None for c in cyps}

        with open(out_path) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return {c: None for c in cyps}

        header = [h.strip().replace("|", "") for h in rows[0]]
        values = [v.strip() for v in rows[1]]
        by_code = dict(zip(header, values))

        out: Dict[str, Optional[bool]] = {}
        for name in cyps:
            code = SUPPORTED.get(name)
            v = by_code.get(code) if code else None
            out[name] = None if v in (None, "", "null") else (v.upper() == "R")
        return out
    except (subprocess.TimeoutExpired, OSError, csv.Error):
        return {c: None for c in cyps}
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)
