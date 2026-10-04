import functools
import sys
from typing import Optional

from agent import config

if config.RELATION_EVALUATOR_DIR not in sys.path:
    sys.path.insert(0, config.RELATION_EVALUATOR_DIR)

from qsar_core.deploy import load_bundle, predict_bundle  # noqa: E402


@functools.lru_cache(maxsize=32)
def _bundle(path: str):
    return load_bundle(path)


def predict_proba(bundle_path: str, smiles: str) -> Optional[float]:
    df = predict_bundle(_bundle(bundle_path), [smiles])
    p = df.loc[0, "proba"]
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return None if p != p else round(p, 3)
