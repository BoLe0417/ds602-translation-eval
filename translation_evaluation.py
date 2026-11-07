from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 0. Configuration. Everything that affects results is set here and saved to
#    run_manifest.json, so the experiment can be reproduced and audited.
# ----------------------------------------------------------------------------
SEED = 602
N_BOOT = 5000                 # bootstrap resamples for confidence intervals
MAX_ATTEMPTS = 4              # retries per request (exponential backoff + jitter)
REQUEST_TIMEOUT_S = 60

# Two LLMs that can both be used for free: Google's Gemini 3.5 Flash-Lite (proprietary) and
# OpenAI's open-weight gpt-oss-120b served by Groq. Gemini 3.x *Flash* models allow only
# 20 free requests per day, far too few for 96 test items; Flash-Lite allows about 500 per day.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GPTOSS_MODEL = os.getenv("GPTOSS_MODEL", "openai/gpt-oss-120b")
NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
# Free tiers limit requests (and tokens) per minute. Minimum seconds between calls:
# Gemini 6 s = 10 requests/min. Groq 6 s = 10 requests/min: its free tier allows 30 requests but
# only 8,000 tokens per minute for gpt-oss-120b, and each request (with reasoning) can use several
# hundred tokens. Set to 0 on a paid tier.
MIN_INTERVAL_S = {"gemini": float(os.getenv("GEMINI_MIN_INTERVAL_S", "6")),
                  "gptoss": float(os.getenv("GROQ_MIN_INTERVAL_S", "6"))}

# Prices in USD. Defaults are list prices known when this script was written.
# VERIFY them on the official pricing pages on the day you run the experiment,
# override via environment variables if they changed, and set PRICES_VERIFIED_ON.
PRICE = {
    # Gemini 3.5 Flash-Lite paid-tier list price. Output price includes any "thinking" tokens.
    # Report list-price cost even if you use the free tier.
    "gemini_in_per_m_tok": float(os.getenv("GEMINI_INPUT_USD_PER_MILLION_TOKENS", "0.30")),
    "gemini_out_per_m_tok": float(os.getenv("GEMINI_OUTPUT_USD_PER_MILLION_TOKENS", "2.50")),
    # gpt-oss-120b on Groq, paid (Developer) tier list price. Output includes reasoning tokens.
    "gptoss_in_per_m_tok": float(os.getenv("GPTOSS_INPUT_USD_PER_MILLION_TOKENS", "0.15")),
    "gptoss_out_per_m_tok": float(os.getenv("GPTOSS_OUTPUT_USD_PER_MILLION_TOKENS", "0.60")),
    "gpu_per_hour": float(os.getenv("GPU_USD_PER_HOUR", "0.35")),  # self-hosted NLLB on a T4-class GPU
}
PRICES_VERIFIED_ON = os.getenv("PRICES_VERIFIED_ON", "NOT VERIFIED")

# Deployment thresholds, fixed BEFORE looking at results (see project plan section 8).
# All are measured automatically by 'evaluate'.
THRESHOLDS = {
    "failure_rate_max": 0.02,          # share of test items with no usable output
    "all_tokens_ok_min": 0.95,         # share of items (with numbers/codes/URLs) where ALL were kept
    "format_fidelity_min": 0.95,       # mean formatting fidelity on items with formatting
    "latency_p95_max_s": 5.0,          # automatic
    "cost_per_1k_chars_max_usd": float(os.getenv("BUDGET_USD_PER_1K_CHARS", "0.05")),
}

PROMPT = """Translate the English text below into Simplified Chinese.
Requirements:
1. Preserve the complete meaning, including negation and conditions.
2. Preserve numbers, dates, percentages, currencies, units, URLs, email addresses, code, and Markdown structure.
3. Use clear, professional Simplified Chinese.
4. Output only the translation, with no commentary.

TEXT:
{text}"""

AI_SYSTEMS = ["gemini", "gptoss", "nllb"]
LLM_SYSTEMS = ["gemini", "gptoss"]
BASELINES = ["copy"]
DEFAULT_SYSTEMS = "gemini,gptoss,nllb,copy"
REQUIRED_SECRETS = {"gemini": ["Gemini_API_Key"], "gptoss": ["GROQ_API_KEY"], "nllb": [], "copy": []}

DATA_COLUMNS = ["id", "split", "domain", "difficulty", "phenomena", "source_en",
                "reference_zh", "glossary", "protected_zh", "notes"]



def prefix(dry_run: bool) -> str:
    return "DRYRUN_" if dry_run else ""


def load_secret(name: str) -> str | None:
    """Read a key from the environment, or from Colab Secrets when run with %run."""
    if os.getenv(name):
        return os.getenv(name)
    try:
        from google.colab import userdata  # only exists inside Colab
        value = userdata.get(name)
        if value:
            os.environ[name] = value
            return value
    except Exception:
        pass
    return None


# ----------------------------------------------------------------------------
# 1. Data loading and validation
# ----------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path: str, split: str | None = "test") -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    missing = set(DATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if split:
        df = df[df.split == split].reset_index(drop=True)
    return df


def nfkc(text: str) -> str:
    """Unify full-width/half-width characters so '２３：５９' and '23:59' compare equal."""
    return unicodedata.normalize("NFKC", str(text))


def glossary_recall(glossary: str, translation: str) -> float:
    """Share of expected target terms that appear in the translation (NaN if no glossary)."""
    pairs = [p.split("=", 1) for p in str(glossary).split(";") if "=" in p]
    if not pairs:
        return np.nan
    return float(np.mean([target.strip() in translation for _, target in pairs]))


def protected_fidelity(protected: str, translation: str) -> float:
    """Share of must-keep tokens (numbers, codes, URLs...) present in the translation.
    Each group may list acceptable alternatives separated by '|' (e.g. 120万|1,200,000)."""
    groups = [g for g in str(protected).split(";") if g.strip()]
    if not groups:
        return np.nan
    out = nfkc(translation)
    return float(np.mean([any(nfkc(alt) in out for alt in g.split("|")) for g in groups]))


def format_features(text: str) -> dict:
    text = str(text)
    return {"newlines": text.count("\n"),
            "bullets": len(re.findall(r"(?m)^\s*- ", text)),
            "bold": text.count("**"),
            "backticks": text.count("`"),
            "pipes": text.count("|")}


def format_fidelity(source: str, translation: str) -> float:
    """Share of formatting features in the source (line breaks, bullets, bold, code, tables)
    that the translation reproduces with the same count. NaN if the source has none."""
    s, t = format_features(source), format_features(translation)
    checks = [t[k] == v for k, v in s.items() if v > 0]
    return float(np.mean(checks)) if checks else np.nan


def cjk_ratio(text: str) -> float:
    cjk = len(re.findall(r"[一-鿿]", str(text)))
    latin = len(re.findall(r"[A-Za-z]", str(text)))
    return cjk / (cjk + latin) if cjk + latin else 0.0


def check_data(path: str) -> bool:
    all_df = load_data(path, split=None)
    ok = True
    print(f"Dataset: {path}\nSHA-256: {sha256_file(path)}")
    print(all_df.groupby(["split", "domain"]).size().unstack(fill_value=0), "\n")
    print("Difficulty x domain (test):")
    print(pd.crosstab(all_df.query("split=='test'").domain, all_df.query("split=='test'").difficulty), "\n")
    problems = []
    if not all_df.id.is_unique:
        problems.append("duplicate ids")
    dup = all_df[all_df.duplicated("source_en", keep=False)]
    if len(dup):
        problems.append(f"duplicate source sentences: {dup.id.tolist()}")
    for col in ["source_en", "reference_zh"]:
        if (all_df[col].str.strip() == "").any():
            problems.append(f"empty {col}")
        if all_df[col].str.contains("\\n", regex=False).any():
            problems.append(f"literal '\\n' in {col} (use real line breaks)")
    # Reference self-test: a correct human reference must score 100% on every automatic check.
    for r in all_df.itertuples():
        for name, value in [("glossary", glossary_recall(r.glossary, r.reference_zh)),
                            ("protected tokens", protected_fidelity(r.protected_zh, r.reference_zh)),
                            ("format", format_fidelity(r.source_en, r.reference_zh))]:
            if not np.isnan(value) and value < 1:
                problems.append(f"{r.id}: reference fails its own {name} check ({value:.2f})")
        for pair in [p for p in r.glossary.split(";") if "=" in p]:
            if pair.split("=", 1)[0].strip().lower() not in r.source_en.lower():
                problems.append(f"{r.id}: glossary source term not in source: {pair}")
    if problems:
        ok = False
        print("PROBLEMS FOUND:")
        for p in problems:
            print("  -", p)
    else:
        print("All checks passed: unique ids, no duplicates, every reference passes its own checks.")
    return ok


# ----------------------------------------------------------------------------
# 2. System callers. Each returns the same dictionary so the evaluation code
#    treats every system identically.
# ----------------------------------------------------------------------------
_clients: dict = {}
DRY_RUN = False


def result(translation, t0, model, input_tokens=np.nan, output_tokens=np.nan):
    return dict(translation=str(translation).strip(), latency_s=time.perf_counter() - t0,
                input_tokens=input_tokens, output_tokens=output_tokens, model_returned=model)


_pace_lock = threading.Lock()
_last_call: dict = {}


def pace(system: str):
    """Wait so that calls to a free-tier API stay under its requests-per-minute limit.
    Latency is measured after this wait, so pacing does not inflate the latency results."""
    with _pace_lock:
        wait = MIN_INTERVAL_S.get(system, 0) - (time.monotonic() - _last_call.get(system, 0.0))
        if wait > 0 and not DRY_RUN:
            time.sleep(wait)
        _last_call[system] = time.monotonic()


def call_gemini(text):
    """Gemini 3 models are designed for the default temperature (1.0) and think by default;
    both are left unchanged, so outputs may vary between runs (see the stability mode)."""
    if "gemini" not in _clients:
        from google import genai
        _clients["gemini"] = genai.Client(api_key=load_secret("Gemini_API_Key"))
    pace("gemini")
    t0 = time.perf_counter()
    r = _clients["gemini"].models.generate_content(model=GEMINI_MODEL, contents=PROMPT.format(text=text))
    u = r.usage_metadata
    out_tokens = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return result(r.text or "", t0, GEMINI_MODEL, getattr(u, "prompt_token_count", np.nan), out_tokens)


def call_gptoss(text):
    """gpt-oss-120b is a reasoning model. Groq's defaults (temperature, reasoning effort) are kept,
    mirroring Gemini. Only the final answer is requested; reasoning tokens are still billed and
    are included in completion_tokens."""
    if "gptoss" not in _clients:
        from groq import Groq
        _clients["gptoss"] = Groq(api_key=load_secret("GROQ_API_KEY"), timeout=REQUEST_TIMEOUT_S)
    pace("gptoss")
    t0 = time.perf_counter()
    r = _clients["gptoss"].chat.completions.create(
        model=GPTOSS_MODEL, include_reasoning=False, max_completion_tokens=2048,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}])
    return result(r.choices[0].message.content or "", t0, getattr(r, "model", GPTOSS_MODEL),
                  r.usage.prompt_tokens, r.usage.completion_tokens)


def call_nllb(text):
    """Free open-source MT model. Runs on the Colab GPU (Runtime -> Change runtime type -> T4 GPU)."""
    if "nllb" not in _clients:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="eng_Latn")
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).to(device).eval()
        _clients["nllb"] = (tok, model, device, torch)
    tok, model, device, torch = _clients["nllb"]
    t0 = time.perf_counter()
    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("zho_Hans"),
                             num_beams=4, max_new_tokens=512)
    if device == "cuda":
        torch.cuda.synchronize()
    return result(tok.batch_decode(out, skip_special_tokens=True)[0], t0, NLLB_MODEL)


def call_copy(text):
    t0 = time.perf_counter()
    return result(text, t0, "copy-source baseline")


CALLERS = {"gemini": call_gemini, "gptoss": call_gptoss, "nllb": call_nllb, "copy": call_copy}


def make_dry_run_caller(system: str, row_lookup: dict):
    """Simulated outputs for testing the pipeline without keys or cost. Each fake system
    corrupts the reference in its own way so every metric and chart has something to show."""
    profile = {"nllb": (0.45, 0.25), "gemini": (0.12, 2.2), "gptoss": (0.14, 0.9)}

    def fake(text):
        if system == "copy":
            return call_copy(text)
        row = row_lookup[text]
        rng = random.Random(f"{system}-{row['id']}")
        err_rate, median_latency = profile[system]
        t0 = time.perf_counter()
        out = row["reference_zh"]
        if rng.random() < err_rate:
            kind = rng.choice(["term", "number", "format", "omission"])
            if kind == "term" and "=" in row["glossary"]:
                out = out.replace(row["glossary"].split(";")[0].split("=", 1)[1], "某项内容")
            elif kind == "number":
                out = re.sub(r"\d", lambda m: str((int(m.group()) + 1) % 10), out, count=1)
            elif kind == "format":
                out = out.replace("`", "").replace("**", "").replace("\n", " ")
            else:
                out = out[: max(1, int(len(out) * 0.6))]
        if rng.random() < 0.02:
            raise TimeoutError("simulated timeout")
        res = result(out, t0, f"DRYRUN-{system}", len(text) // 3 + 90, len(out))
        res["latency_s"] = float(np.exp(np.log(median_latency) + rng.gauss(0, 0.4)))
        return res
    return fake


class DailyQuotaExceeded(Exception):
    """The API's free requests-per-day quota is used up; retrying today is pointless."""


def is_daily_quota(err: Exception) -> bool:
    msg = str(err)
    return ("PerDay" in msg or "per day" in msg.lower()) and ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                                                              or "rate_limit" in msg.lower())


def suggested_wait(err: Exception) -> float | None:
    """Rate-limit errors often say how long to wait, e.g. 'Please retry in 49.7s' (Gemini)
    or 'Please try again in 7.66s' (Groq)."""
    m = re.search(r"(?:retry|try again) in ([\d.]+)\s*s|retryDelay'?\"?:\s*'?\"?(\d+)s", str(err), re.IGNORECASE)
    return float(m.group(1) or m.group(2)) if m else None


def with_retry(fn, text):
    """Call fn(text); on failure wait (the API's suggested delay, else exponential backoff + jitter)
    and try again, up to MAX_ATTEMPTS. Returns (result, attempts). A used-up DAILY quota is raised
    immediately as DailyQuotaExceeded, because waiting a few seconds cannot fix it."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(text), attempt
        except Exception as e:
            if is_daily_quota(e):
                raise DailyQuotaExceeded(str(e)[:300]) from e
            if attempt == MAX_ATTEMPTS:
                raise
            if not DRY_RUN:
                wait = suggested_wait(e)
                time.sleep(min(90, wait + 1) if wait else min(30, 2 ** attempt) + random.random())


def preflight(systems, dry_run):
    if dry_run:
        return
    missing = {s: [k for k in REQUIRED_SECRETS[s] if not load_secret(k)] for s in systems}
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        sys.exit(f"Missing credentials: {missing}. Add them to Colab Secrets, or drop those "
                 f"systems with --systems (e.g. --systems gemini,nllb,copy).")


# ----------------------------------------------------------------------------
# 3. Running systems (resumable). One request per item, no conversation history,
#    randomised system order within each item, failures saved rather than dropped.
# ----------------------------------------------------------------------------
def run_systems(data_path, output_path, systems, split, dry_run):
    df = load_data(data_path, split)
    preflight(systems, dry_run)
    lookup = {r["source_en"]: r for r in df.to_dict("records")}
    callers = {s: make_dry_run_caller(s, lookup) if dry_run else CALLERS[s] for s in systems}
    old = pd.read_csv(output_path, keep_default_na=False) if Path(output_path).exists() else pd.DataFrame()
    records = old.to_dict("records") if len(old) else []
    done = {(r["id"], r["system"]) for r in records if r.get("status") == "ok"}
    rng = np.random.default_rng(SEED)
    exhausted = set()   # systems whose daily free quota ran out during this run
    todo = sum((row["id"], s) not in done for row in df.to_dict("records") for s in systems)
    print(f"{len(df)} items x {len(systems)} systems; {todo} requests to run.")
    for i, row in enumerate(df.to_dict("records"), 1):
        order = list(systems)
        rng.shuffle(order)
        for system in order:
            if (row["id"], system) in done or system in exhausted:
                continue
            base = {k: row[k] for k in ["id", "split", "domain", "difficulty"]}
            try:
                ans, attempts = with_retry(callers[system], row["source_en"])
                rec = {**base, "system": system, "status": "ok", "error": "", "attempts": attempts, **ans}
            except DailyQuotaExceeded as e:
                # Not the system's fault: record nothing, skip it for the rest of this run, resume later.
                exhausted.add(system)
                print(f"\n!! {system}: free daily quota used up at item {row['id']}. Skipping {system} "
                      f"for now; run the same command again after the quota resets (usually next day).\n   {e}")
                continue
            except Exception as e:
                rec = {**base, "system": system, "status": "error", "attempts": MAX_ATTEMPTS,
                       "error": f"{type(e).__name__}: {str(e)[:300]}", "translation": "",
                       "latency_s": np.nan, "input_tokens": np.nan, "output_tokens": np.nan,
                       "model_returned": ""}
            rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            # keep only the latest attempt for each (item, system)
            records = [r for r in records if (r["id"], r["system"]) != (row["id"], system)] + [rec]
            pd.DataFrame(records).to_csv(output_path, index=False, encoding="utf-8-sig")
        if i % 10 == 0 or i == len(df):
            print(f"  {i}/{len(df)} items done")
    out = pd.DataFrame(records)
    if len(out):
        print(out.groupby(["system", "status"]).size().unstack(fill_value=0))
    if exhausted:
        print(f"Unfinished because of daily quota: {sorted(exhausted)} -> re-run this command later to resume.")
    write_manifest(data_path, systems, split, dry_run)


def package_versions():
    from importlib import metadata
    names = ["pandas", "numpy", "scipy", "sacrebleu", "google-genai", "groq",
             "transformers", "torch", "unbabel-comet"]
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            pass
    return out


def write_manifest(data_path, systems, split, dry_run):
    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "packages": package_versions(), "dataset": data_path, "dataset_sha256": sha256_file(data_path),
        "split": split, "systems": systems, "seed": SEED, "max_attempts": MAX_ATTEMPTS,
        "models": {"gemini": f"{GEMINI_MODEL} (default temperature, thinking on by default)",
                   "gptoss": f"{GPTOSS_MODEL} via Groq (default temperature and reasoning effort, "
                             "include_reasoning=False, max_completion_tokens 2048)",
                   "nllb": f"{NLLB_MODEL} (beam 4)"},
        "min_interval_s": MIN_INTERVAL_S,
        "llm_prompt": PROMPT,
        "prices_usd": PRICE, "prices_verified_on": PRICES_VERIFIED_ON, "thresholds": THRESHOLDS,
    }
    path = f"{prefix(dry_run)}run_manifest.json"
    Path(path).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved {path}")


def stability_check(data_path, systems, dry_run, pre, repeats=3, per_domain=2):
    """Run each LLM several times on the same small stratified sample. Both LLMs sample at their
    default temperature, so this shows how much results move between identical runs."""
    from sacrebleu.metrics import CHRF
    chrf = CHRF()
    df = load_data(data_path, "test").groupby("domain").sample(n=per_domain, random_state=SEED)
    llms = [s for s in systems if s in LLM_SYSTEMS]
    preflight(llms, dry_run)
    lookup = {r["source_en"]: r for r in df.to_dict("records")}
    rows = []
    for s in llms:
        fn = make_dry_run_caller(s, lookup) if dry_run else CALLERS[s]
        for rep_no in range(1, repeats + 1):
            for r in df.to_dict("records"):
                try:
                    out, _ = with_retry(fn, r["source_en"])
                    text = out["translation"]
                except DailyQuotaExceeded:
                    sys.exit(f"{s}: free daily quota used up during the stability check; run it again tomorrow.")
                except Exception:
                    text = ""
                if dry_run and rep_no > 1 and random.Random(f"{s}{r['id']}{rep_no}").random() < 0.3:
                    text = text[:-1]  # simulate small run-to-run variation
                rows.append({"system": s, "repeat": rep_no, "id": r["id"], "translation": text,
                             "chrF": chrf.sentence_score(text, [r["reference_zh"]]).score if text else 0.0})
    raw = pd.DataFrame(rows)
    raw.to_csv(f"{pre}stability_outputs.csv", index=False, encoding="utf-8-sig")
    per_item = raw.groupby(["system", "id"]).agg(identical=("translation", lambda t: t.nunique() == 1),
                                                   chrF_range=("chrF", lambda c: c.max() - c.min()))
    summary = per_item.groupby("system").agg(items=("identical", "size"), identical_rate=("identical", "mean"),
                                             mean_chrF_range=("chrF_range", "mean"), max_chrF_range=("chrF_range", "max"))
    summary.to_csv(f"{pre}stability_summary.csv", encoding="utf-8-sig")
    print(f"{repeats} repeats x {len(df)} items per LLM")
    print(summary.round(3).to_string())


def measure_throughput(data_path, systems, dry_run, out_path):
    """Requests/second when sending the dev items one at a time vs 4 in parallel."""
    df = load_data(data_path, "dev")
    preflight(systems, dry_run)
    lookup = {r["source_en"]: r for r in df.to_dict("records")}
    rows = []
    for s in [s for s in systems if s not in BASELINES]:
        fn = make_dry_run_caller(s, lookup) if dry_run else CALLERS[s]
        fn(df.source_en.iloc[0])  # warm-up (model loading, connection setup)
        for workers in [1, 4]:
            texts = list(df.source_en) * 2
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as ex:
                outs = list(ex.map(lambda t: _safe(fn, t), texts))
            wall = time.perf_counter() - t0
            if dry_run:  # simulated latencies are not real sleeps; estimate wall time instead
                wall = sum(o["latency_s"] for o in outs if o) / workers
            rows.append({"system": s, "concurrency": workers, "requests": len(texts),
                         "failures": sum(o is None for o in outs), "req_per_s": len(texts) / wall})
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(out.round(2).to_string(index=False))


def _safe(fn, text):
    try:
        return fn(text)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 4. Evaluation: automatic metrics, statistics, charts, error analysis
# ----------------------------------------------------------------------------
def cost_usd(row) -> float:
    s = row["system"]
    if row["status"] != "ok":
        return 0.0
    if s in LLM_SYSTEMS:
        key = s
        return (row["input_tokens"] * PRICE[f"{key}_in_per_m_tok"]
                + row["output_tokens"] * PRICE[f"{key}_out_per_m_tok"]) / 1e6
    if s == "nllb":
        return row["latency_s"] * PRICE["gpu_per_hour"] / 3600
    return 0.0


def bootstrap_mean_ci(values, n_boot=N_BOOT, seed=SEED):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    idx = np.random.default_rng(seed).integers(0, len(v), (n_boot, len(v)))
    sims = v[idx].mean(axis=1)
    return float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


def holm(pvalues):
    """Holm-Bonferroni correction for multiple comparisons."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired_tests(data, metric, systems, binary=False):
    """Every pair of AI systems on the same items: paired bootstrap CI of the mean
    difference, Wilcoxon signed-rank (continuous) or exact McNemar (binary) p-value."""
    from scipy.stats import binomtest, wilcoxon
    wide = data.pivot(index="id", columns="system", values=metric)
    rows = []
    for i, a in enumerate(systems):
        for b in systems[i + 1:]:
            pair = wide[[a, b]].dropna()
            d = (pair[b] - pair[a]).to_numpy()
            lo, hi = bootstrap_mean_ci(d)
            if binary:
                only_a = int(((pair[a] == 1) & (pair[b] == 0)).sum())
                only_b = int(((pair[a] == 0) & (pair[b] == 1)).sum())
                p = binomtest(only_b, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
                test = f"McNemar exact (only {a}: {only_a}, only {b}: {only_b})"
            else:
                try:
                    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p = 1.0
                test = "Wilcoxon signed-rank"
            rows.append({"metric": metric, "comparison": f"{b} - {a}", "n": len(d),
                         "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
                         "test": test, "p_value": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p_value)
        out["ci_excludes_0"] = (out.ci95_low > 0) | (out.ci95_high < 0)
        out["significant_holm_5pct"] = out.p_holm < 0.05
    return out


def score_outputs(outputs, data, use_comet):
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU(tokenize="zh", effective_order=True)
    chrf = CHRF()  # character n-grams; chrF++ word n-grams are meaningless for unsegmented Chinese
    x = outputs.merge(data[["id", "phenomena", "source_en", "reference_zh", "glossary", "protected_zh"]],
                      on="id", validate="many_to_one")
    x["translation"] = x.translation.fillna("").astype(str)
    x["ok"] = (x.status == "ok").astype(int)
    x["BLEU"] = [bleu.sentence_score(h, [r]).score if h else 0.0 for h, r in zip(x.translation, x.reference_zh)]
    x["chrF"] = [chrf.sentence_score(h, [r]).score if h else 0.0 for h, r in zip(x.translation, x.reference_zh)]
    x["terminology"] = [glossary_recall(g, t) for g, t in zip(x.glossary, x.translation)]
    x["token_fidelity"] = [protected_fidelity(p, t) for p, t in zip(x.protected_zh, x.translation)]
    x["all_tokens_ok"] = [np.nan if np.isnan(v) else float(v == 1.0) for v in x.token_fidelity]
    x["format_fidelity"] = [format_fidelity(s, t) for s, t in zip(x.source_en, x.translation)]
    if use_comet:
        from comet import download_model, load_from_checkpoint
        import torch
        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        pred = model.predict([{"src": s, "mt": t, "ref": r} for s, t, r in
                              zip(x.source_en, x.translation, x.reference_zh)],
                             batch_size=32, gpus=1 if torch.cuda.is_available() else 0, progress_bar=False)
        x["COMET"] = list(pred.scores)
    # Automatic error flags: cheap signals that point the team to likely problems.
    length_ratio = [len(nfkc(t)) / max(1, len(nfkc(r))) for t, r in zip(x.translation, x.reference_zh)]
    x["flag_failed_or_empty"] = (x.translation.str.strip() == "").astype(int)
    x["flag_untranslated"] = [int(bool(t.strip()) and cjk_ratio(t) < 0.5 * cjk_ratio(r))
                              for t, r in zip(x.translation, x.reference_zh)]
    x["flag_possible_omission"] = [int(bool(t.strip()) and lr < 0.6) for t, lr in zip(x.translation, length_ratio)]
    x["flag_possible_addition"] = [int(lr > 1.6) for lr in length_ratio]
    x["flag_token_error"] = (x.token_fidelity < 1).astype(int)
    x["flag_terminology_miss"] = (x.terminology < 1).astype(int)
    x["flag_format_broken"] = (x.format_fidelity < 1).astype(int)
    flags = [c for c in x.columns if c.startswith("flag_")]
    x["any_flag"] = x[flags].max(axis=1)
    x["cost_usd"] = x.apply(cost_usd, axis=1)
    x["source_chars"] = x.source_en.str.len()
    return x, flags


def summarise(x, throughput_path):
    def q(p):
        return lambda s: s.dropna().quantile(p) if s.notna().any() else np.nan
    g = x.groupby("system")
    s = g.agg(n_items=("id", "count"), failure_rate=("ok", lambda v: 1 - v.mean()),
              chrF=("chrF", "mean"), BLEU_sent=("BLEU", "mean"),
              terminology=("terminology", "mean"), token_fidelity=("token_fidelity", "mean"),
              all_tokens_ok=("all_tokens_ok", "mean"), format_fidelity=("format_fidelity", "mean"),
              any_auto_flag=("any_flag", "mean"),
              latency_p50_s=("latency_s", q(0.50)), latency_p95_s=("latency_s", q(0.95)),
              latency_p99_s=("latency_s", q(0.99)), mean_latency_s=("latency_s", "mean"),
              input_tokens_per_req=("input_tokens", "mean"), output_tokens_per_req=("output_tokens", "mean"),
              cost_total_usd=("cost_usd", "sum"), cost_per_request_usd=("cost_usd", "mean"),
              source_chars=("source_chars", "sum"))
    from sacrebleu.metrics import BLEU, CHRF
    for system, grp in g:
        s.loc[system, "BLEU_corpus"] = BLEU(tokenize="zh").corpus_score(list(grp.translation), [list(grp.reference_zh)]).score
        s.loc[system, "chrF_corpus"] = CHRF().corpus_score(list(grp.translation), [list(grp.reference_zh)]).score
        s.loc[system, "chrF_ci95_low"], s.loc[system, "chrF_ci95_high"] = bootstrap_mean_ci(grp.chrF)
        if "COMET" in grp:
            s.loc[system, "COMET"] = grp.COMET.mean()
            s.loc[system, "COMET_ci95_low"], s.loc[system, "COMET_ci95_high"] = bootstrap_mean_ci(grp.COMET)
    s["cost_per_1k_chars_usd"] = s.cost_total_usd / s.source_chars * 1000
    s["throughput_sequential_req_s"] = 1 / s.mean_latency_s
    if Path(throughput_path).exists():
        tp = pd.read_csv(throughput_path).pivot(index="system", columns="concurrency", values="req_per_s")
        for c in tp.columns:
            s[f"throughput_c{c}_req_s"] = tp[c]
    s["prices_verified_on"] = PRICES_VERIFIED_ON
    return s


def make_charts(x, summary, ai_systems, pre):
    import matplotlib.pyplot as plt
    systems = [s for s in summary.index if s in ai_systems] + [s for s in summary.index if s not in ai_systems]
    colors = plt.cm.tab10(np.arange(len(systems)))
    files = []
    # 4.1 Quality with 95% CI
    metrics = ["chrF"] + (["COMET"] if "COMET" in summary else [])
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        v = summary.loc[systems, m]
        err = np.vstack([v - summary.loc[systems, f"{m}_ci95_low"], summary.loc[systems, f"{m}_ci95_high"] - v])
        ax.bar(systems, v, yerr=err, capsize=5, color=colors)
        for i, val in enumerate(v):
            top = summary.loc[systems[i], f"{m}_ci95_high"]
            ax.text(i + 0.12, top, f"{val:.1f}" if m == "chrF" else f"{val:.3f}", ha="left", va="bottom", fontsize=9)
        ax.set_title(f"{m} (mean, 95% bootstrap CI)")
        ax.set_ylabel(m)
    fig.tight_layout(); files.append(f"{pre}chart_quality_ci.png"); fig.savefig(files[-1], dpi=200); plt.close(fig)
    # 4.2 Latency distribution
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ai_only = [s for s in systems if s in ai_systems]   # the copy baseline has ~0 latency
    data = [x.loc[(x.system == s) & (x.status == "ok"), "latency_s"].dropna() for s in ai_only]
    try:
        ax.boxplot(data, tick_labels=ai_only)      # matplotlib >= 3.9
    except TypeError:
        ax.boxplot(data, labels=ai_only)           # older matplotlib
    ax.set_yscale("log"); ax.set_ylabel("seconds (log scale)")
    ax.set_title("End-to-end latency per request")
    fig.tight_layout(); files.append(f"{pre}chart_latency.png"); fig.savefig(files[-1], dpi=200); plt.close(fig)
    # 4.3 Domain x system heat map
    heat = x.pivot_table(index="domain", columns="system", values="chrF", aggfunc="mean")[systems]
    fig, ax = plt.subplots(figsize=(1.4 * len(systems) + 2, 4.2))
    im = ax.imshow(heat.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(systems)), systems); ax.set_yticks(range(len(heat)), heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.values[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean chrF"); ax.set_title("chrF by domain")
    fig.tight_layout(); files.append(f"{pre}chart_domain_heatmap.png"); fig.savefig(files[-1], dpi=200); plt.close(fig)
    # 4.4 Quality-cost Pareto chart (AI systems only)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, c in zip(systems, colors):
        if s in ai_systems:
            ax.scatter(summary.loc[s, "cost_per_1k_chars_usd"], summary.loc[s, "chrF"], s=90, color=c)
            ax.annotate(s, (summary.loc[s, "cost_per_1k_chars_usd"], summary.loc[s, "chrF"]),
                        textcoords="offset points", xytext=(6, 6))
    ax.axvline(THRESHOLDS["cost_per_1k_chars_max_usd"], ls="--", color="grey", lw=1)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel("variable cost per 1,000 source characters (USD)"); ax.set_ylabel("mean chrF")
    ax.set_title("Quality vs cost (upper-left is better; dashed = budget)")
    fig.tight_layout(); files.append(f"{pre}chart_pareto.png"); fig.savefig(files[-1], dpi=200); plt.close(fig)
    # 4.5 Automatic error flags
    flags = [c for c in x.columns if c.startswith("flag_")]
    ai_only = [s for s in systems if s in ai_systems]   # baseline rates are in the csv/table
    rates = x.groupby("system")[flags].mean().loc[ai_only] * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(ai_only)
    for k, (s, c) in enumerate(zip(ai_only, colors)):
        ax.bar(np.arange(len(flags)) + k * width, rates.loc[s], width, label=s, color=c)
    ax.set_xticks(np.arange(len(flags)) + 0.4 - width / 2, [f.replace("flag_", "") for f in flags], rotation=25, ha="right")
    ax.set_ylabel("% of outputs"); ax.set_title("Automatic error flags"); ax.legend(fontsize=8)
    fig.tight_layout(); files.append(f"{pre}chart_auto_errors.png"); fig.savefig(files[-1], dpi=200); plt.close(fig)
    return files


def evaluate(data_path, outputs_path, throughput_path, use_comet, pre):
    data = load_data(data_path, "test")
    outputs = pd.read_csv(outputs_path, keep_default_na=False,
                          dtype={"translation": str, "error": str, "model_returned": str})
    for c in ["latency_s", "input_tokens", "output_tokens"]:
        outputs[c] = pd.to_numeric(outputs[c], errors="coerce")
    outputs = outputs[outputs.id.isin(data.id)]
    systems = sorted(outputs.system.unique(), key=lambda s: (s in BASELINES, s))
    # Every system must have attempted every test item ("run ALL test examples").
    attempted = outputs.groupby("system").id.nunique()
    not_run = attempted[attempted < len(data)]
    if len(not_run):
        raise SystemExit(f"Not every test item was attempted: {not_run.to_dict()}. Run 'run' again to resume.")
    if PRICES_VERIFIED_ON == "NOT VERIFIED":
        print("WARNING: prices not verified. Check official pricing pages and set PRICES_VERIFIED_ON.\n")

    x, flags = score_outputs(outputs, data, use_comet)
    ai = [s for s in systems if s in AI_SYSTEMS]
    summary = summarise(x, throughput_path)
    summary = summary.loc[systems]
    x.to_csv(f"{pre}results_per_item.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(f"{pre}results_summary.csv", encoding="utf-8-sig")

    # Paired statistics between AI systems (failed requests stay in, scored as empty output)
    tests = [paired_tests(x, "chrF", ai), paired_tests(x, "terminology", ai),
             paired_tests(x, "token_fidelity", ai), paired_tests(x, "all_tokens_ok", ai, binary=True)]
    if use_comet:
        tests.insert(1, paired_tests(x, "COMET", ai))
    tests = pd.concat(tests, ignore_index=True)
    tests.to_csv(f"{pre}results_pairwise_tests.csv", index=False, encoding="utf-8-sig")

    # Breakdowns for error analysis (small subgroups are exploratory)
    by = {}
    for col in ["domain", "difficulty"]:
        by[col] = x.pivot_table(index=col, columns="system", values="chrF", aggfunc="mean")[systems]
    ph = x.assign(phenomenon=x.phenomena.str.split(";")).explode("phenomenon")
    by["phenomenon"] = ph.pivot_table(index="phenomenon", columns="system", values="any_flag", aggfunc="mean")[systems]
    with open(f"{pre}results_breakdowns.txt", "w", encoding="utf-8") as f:
        for k, v in by.items():
            label = "share of outputs with any automatic flag" if k == "phenomenon" else "mean chrF"
            f.write(f"== {label} by {k} ==\n{v.round(3).to_string()}\n\n")
    flag_rates = x.groupby("system")[flags].mean().loc[systems]
    flag_rates.to_csv(f"{pre}results_auto_error_rates.csv", encoding="utf-8-sig")

    # Cases for manual error coding: every flagged output plus the 5 lowest-chrF per AI system
    worst = x[x.system.isin(ai)].sort_values("chrF").groupby("system").head(5)
    review = pd.concat([x[(x.any_flag == 1) & x.system.isin(ai)], worst]).drop_duplicates(["id", "system"])
    review = review[["id", "system", "domain", "difficulty", "phenomena", "source_en", "reference_zh",
                     "translation", "chrF"] + flags].sort_values(["id", "system"])
    for c in ["primary_error_type", "business_impact", "suspected_cause", "mitigation"]:
        review[c] = ""
    review.to_csv(f"{pre}error_cases_for_review.csv", index=False, encoding="utf-8-sig")

    charts = make_charts(x, summary, ai, pre)
    primary = "COMET" if use_comet else "chrF"
    decision, recommendation = deployment_decision(summary, tests, ai, primary)
    decision.to_csv(f"{pre}results_decision.csv", encoding="utf-8-sig")

    cols = ["n_items", "failure_rate", "chrF", "chrF_ci95_low", "chrF_ci95_high", "BLEU_corpus"] + \
           (["COMET"] if use_comet else []) + \
           ["terminology", "token_fidelity", "format_fidelity", "latency_p50_s", "latency_p95_s",
            "latency_p99_s", "cost_per_1k_chars_usd"]
    print(summary[cols].round(3).to_string())
    print("\nPairwise tests (chrF):")
    print(tests.query("metric=='chrF'")[["comparison", "mean_diff", "ci95_low", "ci95_high", "p_holm"]].round(4).to_string(index=False))
    print(f"\nAutomatic error flag rates:\n{(flag_rates * 100).round(1).to_string()}")
    print(f"\nSaved: {pre}results_summary.csv, {pre}results_per_item.csv, {pre}results_pairwise_tests.csv, "
          f"{pre}results_breakdowns.txt, {pre}error_cases_for_review.csv, charts {charts}")
    show = [primary, "failure_rate", "all_tokens_ok", "format_fidelity", "latency_p95_s", "cost_per_1k_chars_usd"] + \
           [col for col in decision.columns if col.startswith("pass")] + ["pareto_efficient"]
    print(f"\nDeployment thresholds (saved to {pre}results_decision.csv):\n{decision[show].round(4).to_string()}")
    print(f"\nRECOMMENDATION: {recommendation}")


# ----------------------------------------------------------------------------
# 5. Deployment decision (pre-registered thresholds, automatic metrics only)
# ----------------------------------------------------------------------------
def deployment_decision(summary, tests, ai, primary):
    """1) Drop every AI system that fails any threshold. 2) Among the rest, take the one with the
    best primary quality metric. 3) If a cheaper passing system is NOT significantly worse on that
    metric (Holm-corrected p >= 0.05), recommend the cheapest such system instead."""
    t = THRESHOLDS
    d = summary.loc[ai].copy()
    d["pass_failure_rate"] = d.failure_rate <= t["failure_rate_max"]
    d["pass_numbers_codes"] = d.all_tokens_ok >= t["all_tokens_ok_min"]
    d["pass_format"] = d.format_fidelity >= t["format_fidelity_min"]
    d["pass_latency"] = d.latency_p95_s <= t["latency_p95_max_s"]
    d["pass_cost"] = d.cost_per_1k_chars_usd <= t["cost_per_1k_chars_max_usd"]
    d["passes_all"] = d.filter(like="pass_").all(axis=1)
    q, c = d[primary], d.cost_per_1k_chars_usd
    d["pareto_efficient"] = [not (((q >= q[s]) & (c <= c[s])) & ((q > q[s]) | (c < c[s]))).any() for s in d.index]
    passing = d[d.passes_all]
    if passing.empty:
        msg = ("No system passes every threshold -> do not deploy unattended; keep human post-editing. "
               "See the pass_ columns for what blocks each system.")
        return d, msg
    best = passing[primary].idxmax()
    choice, reason = best, f"highest mean {primary} among systems that pass every threshold"
    for s in passing.sort_values("cost_per_1k_chars_usd").index:
        if passing.loc[s, "cost_per_1k_chars_usd"] >= passing.loc[best, "cost_per_1k_chars_usd"] or s == best:
            break
        row = tests[(tests.metric == primary) & tests.comparison.isin([f"{s} - {best}", f"{best} - {s}"])]
        if len(row) and row.p_holm.iloc[0] >= 0.05:
            choice = s
            reason = (f"cheaper than '{best}' and not significantly worse on {primary} "
                      f"(Holm p = {row.p_holm.iloc[0]:.3f})")
            break
    return d, f"Deploy '{choice}': {reason}."


# ----------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["check-data", "pilot", "run", "throughput", "stability", "evaluate"])
    p.add_argument("--data", default="translation_eval_dataset.csv")
    p.add_argument("--systems", default=DEFAULT_SYSTEMS, help=f"comma list from {AI_SYSTEMS + BASELINES}")
    p.add_argument("--comet", action="store_true", help="also compute COMET (pip install unbabel-comet; GPU recommended)")
    p.add_argument("--dry-run", action="store_true", help="simulated outputs to test the pipeline (never report these)")
    a = p.parse_args(argv)
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    unknown = set(systems) - set(CALLERS)
    if unknown:
        sys.exit(f"Unknown systems: {unknown}")
    global DRY_RUN
    DRY_RUN = a.dry_run
    pre = prefix(a.dry_run)
    if a.dry_run:
        print("*** DRY RUN: simulated translations and latencies. Do NOT report these numbers. ***\n")
    if a.mode == "check-data":
        sys.exit(0 if check_data(a.data) else 1)
    elif a.mode == "pilot":
        run_systems(a.data, f"{pre}pilot_outputs.csv", systems, "dev", a.dry_run)
        print(pd.read_csv(f"{pre}pilot_outputs.csv")[["id", "system", "status", "translation"]].to_string())
    elif a.mode == "run":
        run_systems(a.data, f"{pre}translation_outputs.csv", systems, "test", a.dry_run)
    elif a.mode == "throughput":
        measure_throughput(a.data, systems, a.dry_run, f"{pre}throughput.csv")
    elif a.mode == "stability":
        stability_check(a.data, systems, a.dry_run, pre)
    else:
        evaluate(a.data, f"{pre}translation_outputs.csv", f"{pre}throughput.csv", a.comet, pre)


if __name__ == "__main__":
    main()
