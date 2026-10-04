import os

from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_ROOT = os.path.join(PROJECT_ROOT, "agent")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

RELATION_EVALUATOR_DIR = os.path.join(PROJECT_ROOT, "relation_evaluator")
MODEL_DIRS = {
    "transporter": os.path.join(RELATION_EVALUATOR_DIR, "transporter", "models"),
    "inhibitor":   os.path.join(RELATION_EVALUATOR_DIR, "inhibitor", "models"),
    "antagonist":  os.path.join(RELATION_EVALUATOR_DIR, "antagonist", "models"),
    "agonist":     os.path.join(RELATION_EVALUATOR_DIR, "agonist", "models"),
}
DATASET_DIRS = {
    "transporter": os.path.join(RELATION_EVALUATOR_DIR, "transporter", "dataset"),
    "inhibitor":   os.path.join(RELATION_EVALUATOR_DIR, "inhibitor", "dataset"),
    "antagonist":  os.path.join(RELATION_EVALUATOR_DIR, "antagonist", "dataset"),
    "agonist":     os.path.join(RELATION_EVALUATOR_DIR, "agonist", "dataset"),
}
DOCKING_RESULTS = os.environ.get(
    "PHARMLENS_DOCKING_RESULTS", os.path.join(PROJECT_ROOT, "docking_pipeline", "results"))
DATASET_ROOT = os.environ.get(
    "PHARMLENS_DATASET_DIR", os.path.join(PROJECT_ROOT, "dataset"))
KNOWLEDGE_DIR = os.path.join(PROJECT_ROOT, "knowledge")
CONTACT_EMAIL = os.environ.get("PHARMLENS_CONTACT_EMAIL", "").strip()

CACHE_DIR = os.path.join(PKG_ROOT, "cache")
TOOL_SELECTION_DIR = os.path.join(CACHE_DIR, "tool_selection")
REFERENCE_DIR = os.path.join(CACHE_DIR, "reference")
RESULTS_DIR = os.path.join(PKG_ROOT, "results")
CATALOG_DIR = os.path.join(PKG_ROOT, "tools", "catalogs")

_COLLECTIONS = {
    "tdc":         ("admet_group",  "tdc",         "Drug",   "Y", "tdc"),
    "unitox_cliff": ("unitox_cliff", "unitox_cliff", "smiles", "y", "unitox"),
}

_TASKS = {
    "bbbp":               ("tdc", "bbbp",  "bbbp",  "knowledge_bbbp.json"),
    "hia":                ("tdc", "hia",   "hia",   "knowledge_hia.json"),
    "bioav":              ("tdc", "bioav", "bioav", "knowledge_bioav.json"),
    "dili":               ("tdc", "dili",  "dili",  "knowledge_dili.json"),
    "cardiotoxicity_cliff":     ("unitox_cliff", "cardiotoxicity", "cardiotoxicity",
                                 "knowledge_cardiotoxicity.json"),
    "dermatological_cliff":     ("unitox_cliff", "dermatological_toxicity", "dermato",
                                 "knowledge_dermatological.json"),
    "hematological_cliff":      ("unitox_cliff", "hematological", "hemato",
                                 "knowledge_hematological.json"),
    "infertility_cliff":        ("unitox_cliff", "infertility", "infertility",
                                 "knowledge_infertility.json"),
    "liver_toxicity_cliff":     ("unitox_cliff", "liver_toxicity", "liver_toxicity",
                                 "knowledge_liver_toxicity.json"),
    "ototoxicity_cliff":        ("unitox_cliff", "ototoxicity", "ototoxicity",
                                 "knowledge_ototoxicity.json"),
    "pulmonary_toxicity_cliff": ("unitox_cliff", "pulmonary_toxicity", "pulmonary_toxicity",
                                 "knowledge_pulmonary_toxicity.json"),
    "renal_toxicity_cliff":     ("unitox_cliff", "renal_toxicity", "renal_toxicity",
                                 "knowledge_renal_toxicity.json"),
}


def knowledge_path(collection: str, knowledge: str) -> str:
    path = os.path.join(KNOWLEDGE_DIR, collection, knowledge)
    if os.path.exists(path):
        return path
    directory = os.path.dirname(path)
    if os.path.isdir(directory):
        wanted = os.path.basename(knowledge).lower()
        for f in os.listdir(directory):
            if f.lower() == wanted:
                return os.path.join(directory, f)
    return path


def _task_entry(collection: str, dataset_subdir: str, docking_subdir: str, knowledge: str):
    root, docking_collection, smiles_col, label_col, knowledge_subdir = _COLLECTIONS[collection]
    return {
        "group":          collection,
        "collection":     docking_collection,
        "docking_task":   docking_subdir,
        "knowledge_group": knowledge_subdir,
        "knowledge":      knowledge_path(knowledge_subdir, knowledge),
        "dataset_dir":    os.path.join(DATASET_ROOT, root, dataset_subdir),
        "smiles_col":     smiles_col,
        "label_col":      label_col,
    }


TASKS = {name: _task_entry(*spec) for name, spec in _TASKS.items()}


def docking_cache_dir(task: str, split: str = "test") -> str:
    t = TASKS[task]
    return os.path.join(DOCKING_RESULTS, t["collection"], t["docking_task"], split, "cache")


def run_dir(task: str) -> str:
    return os.path.join(RESULTS_DIR, TASKS[task]["group"], task)


BAND_POSITIVE = 0.5
BAND_NEGATIVE = 0.3


def band_of(proba):
    if proba is None:
        return "no_model"
    if proba >= BAND_POSITIVE:
        return "positive"
    if proba >= BAND_NEGATIVE:
        return "ambiguous"
    return "negative"


# The backbone a run uses when --model is not given. It must be a key of MODEL_BACKENDS
# below, or an OpenAI model name, which falls through to ChatOpenAI and OPENAI_API_KEY.
DEFAULT_MODEL = "deepseek-v4-pro-nothink"
OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"

MODEL_BACKENDS = {
    "deepseek-v4-pro-nothink": {
        "transport":   "openai",
        "api_model":   "deepseek-v4-pro",
        "base_url":    "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "extra_body":  {"thinking": {"type": "disabled"}},
        "timeout":     300,
    },
    "claude-4.5-sonnet": {
        "transport":   "anthropic",
        "api_model":   "claude-sonnet-4-5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens":  8192,
        "timeout":     300,
    },
    "qwen3.7-max-nothink": {
        "transport":   "openai",
        "api_model":   "qwen3.7-max",
        "base_url":    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "extra_body":  {"enable_thinking": False},
        "max_tokens":  8192,
        "timeout":     300,
    },
    "llama3.1": {
        "transport":   "openai",
        "api_model":   "llama3.1",
        "base_url_env": "OLLAMA_BASE_URL",
        "api_key":     "ollama",
        "timeout":     600,
    },
    "gemma4-31b-nothink": {
        "transport":    "openai",
        "api_model":    "gemma4:31b",
        "base_url_env": "OLLAMA_BASE_URL",
        "api_key":      "ollama",
        "extra_body":   {"reasoning_effort": "none"},
        "max_tokens":   8192,
        "timeout":      900,
    },
}
def backend_url(backend) -> str:
    if backend.get("base_url"):
        return backend["base_url"]
    if backend.get("base_url_env"):
        return os.getenv(backend["base_url_env"]) or OLLAMA_BASE_URL
    return OLLAMA_BASE_URL


AGENT_TEMPERATURE = {
    "tool_selection":       0.0,
    "physchem":             0.0,
    "relation":             0.0,
    "recognition_gate":     0.0,
    "interaction":          0.0,
    "prediction":           0.0,
}


LLM_USAGE = {}
_LLM_USAGE_LOCK = None


def _record_usage(label: str, in_tokens: int, out_tokens: int, truncated: bool):
    global _LLM_USAGE_LOCK
    if _LLM_USAGE_LOCK is None:
        import threading
        _LLM_USAGE_LOCK = threading.Lock()
    with _LLM_USAGE_LOCK:
        e = LLM_USAGE.setdefault(label, {"n": 0, "max_out": 0, "sum_out": 0,
                                         "sum_in": 0, "truncated": 0})
        e["n"] += 1
        e["sum_in"] += in_tokens
        e["sum_out"] += out_tokens
        e["max_out"] = max(e["max_out"], out_tokens)
        e["truncated"] += int(truncated)


def usage_report(limit_of=None) -> str:
    if not LLM_USAGE:
        return ""
    lines = ["", "  tokens per LLM call (output max/mean, and the run's billed totals):",
             f"    {'agent':<36} {'calls':>5} {'out max':>8} {'out mean':>9} {'cap':>7} "
             f"{'used':>6}  cut {'tok in':>10} {'tok out':>9}"]
    tot_in = tot_out = 0
    for label in sorted(LLM_USAGE):
        e = LLM_USAGE[label]
        cap = (limit_of or {}).get(label)
        mean = e["sum_out"] / e["n"] if e["n"] else 0
        pct = f"{100 * e['max_out'] / cap:5.1f}%" if cap else "    —"
        tot_in += e["sum_in"]
        tot_out += e["sum_out"]
        lines.append(f"    {label:<36} {e['n']:>5} {e['max_out']:>8} {mean:>9.0f} "
                     f"{str(cap or '—'):>7} {pct:>6}  {e['truncated']} "
                     f"{e['sum_in']:>10,} {e['sum_out']:>9,}")
    lines.append(f"    {'TOTAL':<36} {'':>5} {'':>8} {'':>9} {'':>7} {'':>6}    "
                 f"{tot_in:>10,} {tot_out:>9,}")
    return "\n".join(lines) + "\n"


def _response_warner(label: str, limit):
    from langchain_core.callbacks import BaseCallbackHandler

    cut = {"max_tokens": "truncated at the output cap", "length": "truncated at the output cap"}
    declined = {"refusal": "declined by the model", "content_filter": "declined by the model"}

    def verdict(result):
        metas = []
        out = getattr(result, "llm_output", None) or {}
        if out:
            metas.append(out)
        for gen_list in getattr(result, "generations", None) or []:
            for gen in gen_list:
                meta = getattr(getattr(gen, "message", None), "response_metadata", None)
                if meta:
                    metas.append(meta)
                info = getattr(gen, "generation_info", None)
                if info:
                    metas.append(info)

        for meta in metas:
            for key in ("stop_reason", "finish_reason"):
                reason = meta.get(key)
                if reason in cut:
                    return "truncated", reason, None
                if reason in declined:
                    detail = (meta.get("stop_details") or {}).get("category")
                    return "refused", reason, detail
        return None

    def tokens(result):
        for gen_list in getattr(result, "generations", None) or []:
            for gen in gen_list:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if usage and usage.get("output_tokens"):
                    return int(usage.get("input_tokens") or 0), int(usage["output_tokens"])
        out = getattr(result, "llm_output", None) or {}
        usage = out.get("token_usage") or out.get("usage") or {}
        got_in = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        for key in ("completion_tokens", "output_tokens"):
            if usage.get(key):
                return int(got_in), int(usage[key])
        return int(got_in), 0

    class ResponseWarner(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            v = verdict(response)
            _in, _out = tokens(response)
            _record_usage(label, _in, _out, bool(v) and v[0] == "truncated")
            if not v:
                return
            kind, reason, detail = v
            if kind == "truncated":
                print(f"  !! [{label}] response TRUNCATED at the output cap "
                      f"(max_tokens={limit}, stop_reason={reason}) — this molecule's verdict "
                      f"will not parse. Raise max_tokens in config.MODEL_BACKENDS.")
            else:
                print(f"  !! [{label}] response REFUSED by the model "
                      f"(stop_reason={reason}"
                      + (f", category={detail}" if detail else "") + ") — empty content, so "
                      "this agent's verdict will not parse. Not a config problem: this "
                      "molecule is one the backbone declines, and its n will differ from "
                      "the other backbones' runs on the same split.")

    return ResponseWarner()


def get_llm(agent: str, model: str = None, temperature: float = None):
    from langchain_openai import ChatOpenAI

    temp = AGENT_TEMPERATURE.get(agent, 0.0) if temperature is None else temperature
    name = model or DEFAULT_MODEL

    backend = MODEL_BACKENDS.get(name)

    if backend and backend.get("transport") == "anthropic":
        from langchain_anthropic import ChatAnthropic

        max_tokens = backend.get("max_tokens", 8192)
        return ChatAnthropic(
            model=backend["api_model"],
            temperature=temp,
            api_key=os.getenv(backend["api_key_env"]),
            max_tokens=max_tokens,
            max_retries=5,
            timeout=backend.get("timeout", 120),
            callbacks=[_response_warner(f"{name}/{agent}", max_tokens)],
        )

    if backend:
        key = (backend["api_key"] if backend.get("api_key")
               else os.getenv(backend["api_key_env"]))
        max_tokens = backend.get("max_tokens")
        extra = dict(backend.get("extra_body") or {})
        if max_tokens:
            extra[backend.get("max_tokens_field", "max_tokens")] = max_tokens
        return ChatOpenAI(
            model=backend["api_model"],
            temperature=temp,
            api_key=key,
            base_url=backend_url(backend),
            extra_body=extra or None,
            max_retries=5,
            timeout=backend.get("timeout", 120),
            callbacks=[_response_warner(f"{name}/{agent}", max_tokens)],
        )

    return ChatOpenAI(
        model=name,
        temperature=temp,
        api_key=os.getenv("OPENAI_API_KEY"),
        max_retries=5,
        timeout=120,
        callbacks=[_response_warner(f"{name}/{agent}", None)],
    )

