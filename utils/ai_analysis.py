"""
ai_analysis.py
--------------
GenAI (Hugging Face Inference API) integration for CampusSense AI.

This module is responsible for:
  1. Analyzing individual feedback entries and extracting structured data
     (sentiment, category, severity, keywords, main issue, summary, action).
  2. Generating an overall AI summary of the (filtered) dataset.
  3. Answering natural-language questions about the dataset through the
     "CampusSense Assistant" chatbot.

Uses open-source chat models hosted on Hugging Face via the official
`huggingface_hub` SDK. Models are tried in order (MODEL_CHAIN): if one is
unavailable, rate-limited, or gated, the next one is used automatically.

The Hugging Face token is NEVER hardcoded here - it is read from the
HF_TOKEN environment variable (loaded from a local .env file via
python-dotenv in app.py). If the token is missing, functions raise a
clear AIAnalysisError instead of crashing the app.
"""

import os
import re
import json
import time
import random
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from huggingface_hub import InferenceClient

from utils.local_classifier import local_classify
from utils.datasets import build_datasets, datasets_to_text, is_analyzed, SEVERITY_WEIGHT

# Allowed values the AI must choose from. Keeping these as constants means
# we can validate the AI's JSON output and fall back safely if it drifts.
VALID_SENTIMENTS = ["Positive", "Neutral", "Negative", "Mixed"]
VALID_CATEGORIES = [
    "Internet", "Hardware", "Software", "Facilities", "Cleanliness",
    "Food Service", "Transportation", "Safety", "Environment",
    "Academic", "Other",
]
VALID_SEVERITIES = ["Low", "Medium", "High"]

# Models are tried in this order; the first one that responds is used.
MODEL_CHAIN = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]
MODEL_NAME = MODEL_CHAIN[0]

# Index of the model in MODEL_CHAIN that last worked, so we don't keep
# retrying a model that already failed on every single row.
_active_model_index = 0


_TRANSIENT_MARKERS = ("429", "rate limit", "too many requests", "502", "503", "504", "timeout",
                      "timed out", "overloaded", "temporarily", "connection")


class AIAnalysisError(Exception):
    """Raised when the GenAI API cannot be reached or returns something unusable."""
    pass


def get_hf_token() -> str | None:
    """Return the Hugging Face token from the environment or Streamlit secrets (or None)."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if token:
        return token
    try:  # Streamlit Community Cloud secrets
        import streamlit as st
        return st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
    except Exception:
        return None


def get_client() -> InferenceClient:
    """
    Build a Hugging Face Inference client using the token from the environment.
    Raises AIAnalysisError with a friendly message if the token is missing.
    """
    token = get_hf_token()
    if not token:
        raise AIAnalysisError(
            "HF_TOKEN is not set. Add it to your .env file "
            "(see README.md) before running analysis."
        )
    return InferenceClient(api_key=token, timeout=120)


def _clean_model_text(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (Qwen3) and surrounding whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return text.strip()


def _chat(client: InferenceClient, system: str, user: str,
          max_tokens: int = 500, temperature: float = 0.2, history: list | None = None) -> str:
    """
    Send one chat request, walking down MODEL_CHAIN until a model answers.
    Returns the model's cleaned text. Raises AIAnalysisError if every model fails.
    """
    global _active_model_index
    errors = []

    for idx in range(_active_model_index, len(MODEL_CHAIN)):
        model = MODEL_CHAIN[idx]
        # Qwen3 is a "thinking" model - /no_think keeps answers fast and short.
        user_msg = user + " /no_think" if "Qwen3" in model else user
        for attempt in range(3):
            try:
                response = client.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        *(history or []),
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = _clean_model_text(response.choices[0].message.content)
                if not text:
                    raise ValueError("empty response")
                _active_model_index = idx  # remember the working model
                return text
            except Exception as exc:  # network, auth, gated model, rate limit, etc.
                msg = str(exc).lower()
                transient = any(k in msg for k in _TRANSIENT_MARKERS)
                if transient and attempt < 2:
                    # Rate-limited / busy: wait briefly and retry the SAME model instead of
                    # falling through to a weaker one (important when many requests run in parallel).
                    time.sleep(1.5 * (2 ** attempt) + random.random())
                    continue
                errors.append(f"{model}: {exc}")
                break

    raise AIAnalysisError("All Hugging Face models failed. " + " | ".join(errors))


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences / chatter)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return json.loads(text[first:last + 1])


ANALYSIS_SYSTEM_PROMPT = """You are CampusSense AI, an assistant that analyzes college campus
feedback for a Computer Science project. For each piece of feedback you are given, extract
structured information and respond ONLY with a single valid JSON object - no extra text,
no markdown code fences.

The JSON object must have exactly these keys:
- "sentiment": one of Positive, Neutral, Negative, Mixed
- "category": one of Internet, Hardware, Software, Facilities, Cleanliness, Food Service,
  Transportation, Safety, Environment, Academic, Other
- "severity": one of Low, Medium, High (how serious/urgent the problem is; Low if feedback
  is purely positive or not a problem)
- "keywords": a list of 2-5 short important keywords/phrases from the feedback
- "main_issue": a short phrase (under 10 words) naming the core issue, or "None" if the
  feedback is purely positive
- "summary": a one-sentence summary of the feedback
- "suggested_action": a short, practical suggested action for campus staff, or
  "No action needed" if the feedback is purely positive

Base your analysis only on the feedback text given. Do not invent details that are not
implied by the text."""


def _build_fallback_result(error_message: str, text: str = "") -> dict:
    """
    Result used when the AI call fails. Instead of a useless Neutral/Other/Low placeholder it
    classifies the text with simple keyword rules (utils/local_classifier.py), so filters and
    charts still work. `analysis_error` records why the AI was not used.
    """
    result = local_classify(text) if text and str(text).strip() else {
        "sentiment": "Neutral", "category": "Other", "severity": "Low", "keywords": [],
        "main_issue": "None", "summary": "Empty feedback.", "suggested_action": "No action needed",
    }
    result["analysis_error"] = error_message
    return result


def analyze_feedback(client: InferenceClient, feedback_text: str) -> dict:
    """
    Send a single feedback entry to the Hugging Face model and return a structured dict.
    Never raises for a single-row failure - instead returns a fallback result
    with an "analysis_error" key so the caller can surface it without crashing
    the whole batch.
    """
    if not feedback_text or not str(feedback_text).strip():
        return _build_fallback_result("Empty feedback text.", feedback_text)

    try:
        raw_content = _chat(
            client,
            system=ANALYSIS_SYSTEM_PROMPT,
            user=f"Feedback: {feedback_text}",
            max_tokens=300,
            temperature=0.1,
        )
        result = _extract_json(raw_content)
    except json.JSONDecodeError:
        return _build_fallback_result("AI returned an invalid (non-JSON) response.", feedback_text)
    except Exception as exc:  # covers network errors, auth errors, rate limits, etc.
        return _build_fallback_result(f"API error: {exc}", feedback_text)

    if not isinstance(result, dict):
        return _build_fallback_result("AI returned an unexpected response format.", feedback_text)

    # --- Validate / sanitize the AI's response before trusting it ---
    if result.get("sentiment") not in VALID_SENTIMENTS:
        result["sentiment"] = "Neutral"
    if result.get("category") not in VALID_CATEGORIES:
        result["category"] = "Other"
    if result.get("severity") not in VALID_SEVERITIES:
        result["severity"] = "Low"
    if not isinstance(result.get("keywords"), list):
        result["keywords"] = []
    result["keywords"] = [str(k).strip() for k in result["keywords"] if str(k).strip()]
    result["main_issue"] = str(result.get("main_issue", "None")).strip() or "None"
    result["summary"] = str(result.get("summary", "")).strip()
    result["suggested_action"] = str(result.get("suggested_action", "")).strip() or "No action needed"
    result["analysis_error"] = None

    return result


# ---------------------------------------------------------------------------
# Fast bulk analysis: parallel requests + on-disk cache
# ---------------------------------------------------------------------------
# Number of feedback rows analyzed at the same time. Raise it (e.g. 8-12) if your Hugging Face
# plan allows, lower it (e.g. 2) if you see rate-limit errors. Override with the ANALYSIS_WORKERS env var.
ANALYSIS_WORKERS = max(1, int(os.getenv("ANALYSIS_WORKERS", "4")))
CACHE_PATH = os.path.join(".cache", "analysis_cache.json")

# If the API says "no credits / bad token / no permission", every further call would fail too.
# After such an error the remaining rows go straight to the local keyword classifier (fast).
_HARD_FAILURE_MARKERS = ("402", "401", "403", "payment required", "credit", "depleted", "unauthorized",
                         "forbidden", "invalid token", "invalid credentials", "permission", "not authorized")
_api_disabled = threading.Event()
_api_disabled_reason: list = []


def _cache_key(text: str) -> str:
    """Cache key = prompt + feedback text, so editing the prompt automatically invalidates old results."""
    return hashlib.sha1((ANALYSIS_SYSTEM_PROMPT + "|" + text).encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE_PATH)
    except Exception:
        pass  # caching is only an optimization - never break the analysis because of it


def analyze_dataframe(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Run analyze_feedback() on every row of a cleaned feedback DataFrame and
    return a new DataFrame with the extracted columns appended.

    Speed-ups compared with the simple one-by-one loop:
      - rows are analyzed in parallel (ANALYSIS_WORKERS threads)
      - identical feedback text is only sent once
      - successful results are cached on disk, so re-running Analyze Feedback
        (or restarting the app) only pays for rows it has not seen before

    progress_callback, if given, is called with (done_rows, total_rows) from the
    calling thread so the Streamlit progress bar keeps working.
    """
    get_client()  # raises AIAnalysisError early if no API key at all
    token = get_hf_token()

    texts = [str(t) if t is not None else "" for t in df["feedback"].tolist()]
    total = len(texts)
    cache = _load_cache()

    results: dict[str, dict] = {}          # cache key -> analysis dict
    pending: dict[str, str] = {}           # cache key -> text still to analyze (deduplicated)
    for text in texts:
        key = _cache_key(text)
        if key in results or key in pending:
            continue
        if key in cache:
            results[key] = cache[key]
        else:
            pending[key] = text

    # Progress counts ROWS, so rows served from cache/duplicates are "done" immediately.
    done = sum(1 for t in texts if _cache_key(t) in results)
    if progress_callback:
        progress_callback(done, total)

    _api_disabled.clear()
    _api_disabled_reason.clear()

    def _work(text: str) -> dict:
        if _api_disabled.is_set():
            return _build_fallback_result(f"AI skipped after an earlier fatal API error: {_api_disabled_reason[0]}", text)
        # One client per task: cheap to create and avoids sharing a session between threads.
        result = analyze_feedback(InferenceClient(api_key=token, timeout=120), text)
        err = str(result.get("analysis_error") or "")
        if err and any(m in err.lower() for m in _HARD_FAILURE_MARKERS):
            if not _api_disabled.is_set():
                _api_disabled_reason.append(err[:300])
                _api_disabled.set()
        return result

    if pending:
        workers = min(ANALYSIS_WORKERS, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_work, text): key for key, text in pending.items()}
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    analysis = fut.result()
                except Exception as exc:  # analyze_feedback should not raise, but never lose the batch
                    analysis = _build_fallback_result(f"Unexpected error: {exc}", pending[key])
                results[key] = analysis
                if not analysis.get("analysis_error"):
                    cache[key] = analysis           # only cache real results, not placeholders
                done += sum(1 for t in texts if _cache_key(t) == key)
                if progress_callback:
                    progress_callback(min(done, total), total)
        _save_cache(cache)

    analysis_df = pd.DataFrame([dict(results[_cache_key(t)]) for t in texts])
    # Join keywords list into a readable comma-separated string for display/export
    if "keywords" in analysis_df.columns:
        analysis_df["keywords"] = analysis_df["keywords"].apply(
            lambda kws: ", ".join(kws) if isinstance(kws, list) else str(kws)
        )

    combined = pd.concat([df.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)
    return combined


def _dataset_stats_text(df: pd.DataFrame) -> str:
    """Build a compact plain-text summary of the analyzed dataset for prompting the AI."""
    if df.empty:
        return "The dataset is empty."

    lines = [f"Total feedback records: {len(df)}"]

    if "sentiment" in df.columns:
        lines.append("Sentiment counts: " + df["sentiment"].value_counts().to_dict().__str__())
    if "category" in df.columns:
        lines.append("Category counts: " + df["category"].value_counts().to_dict().__str__())
    if "severity" in df.columns:
        lines.append("Severity counts: " + df["severity"].value_counts().to_dict().__str__())
    if "location" in df.columns:
        lines.append("Reports per location: " + df["location"].value_counts().to_dict().__str__())

    return "\n".join(lines)


def generate_campus_summary(df: pd.DataFrame) -> str:
    """
    Generate a short natural-language summary of the (filtered, analyzed)
    dataset using a Hugging Face model. This is always derived from the current
    data - never a hardcoded string.
    """
    if df.empty:
        return "There is no feedback data available for the current filters."

    client = get_client()
    stats_text = _dataset_stats_text(df)

    # Give the model a handful of the highest-severity examples for grounding,
    # without sending the entire (potentially large) dataset.
    high_severity_samples = ""
    if "severity" in df.columns:
        high_rows = df[df["severity"] == "High"].head(5)
        if not high_rows.empty and "summary" in high_rows.columns:
            high_severity_samples = "\n".join(
                f"- ({row.location}) {row.summary}" for row in high_rows.itertuples(index=False)
            )

    prompt = f"""Using ONLY the statistics and examples below (from real campus feedback
analysis), write a concise 3-5 sentence "Campus Problem Summary" for a dashboard.
Mention the total number of reports, the most frequently reported category/categories,
and any notable high-severity issues. Do not invent numbers or issues not shown below.

Statistics:
{stats_text}

Sample high-severity issues:
{high_severity_samples if high_severity_samples else "None"}
"""

    try:
        return _chat(
            client,
            system="You write short, factual dashboard summaries.",
            user=prompt,
            max_tokens=400,
            temperature=0.3,
        )
    except Exception as exc:
        raise AIAnalysisError(f"Could not generate summary: {exc}")


CHATBOT_SYSTEM_PROMPT = """You are "CampusSense Assistant", a friendly chatbot that answers questions
about a specific set of campus feedback data. You are given 5 datasets: the full feedback
records plus summary datasets by location, category, sentiment and severity.

Rules:
- Answer using ONLY the provided datasets. Never invent facts, numbers or issues.
- Use the summary datasets for counts/rankings and the full records for specific examples.
- If the user greets you or asks what you can do, reply briefly and suggest a few questions.
- If the data is not sufficient to answer, say clearly:
  "The dataset does not contain enough information to answer that."
- Keep answers concise and specific (mention numbers, locations and categories).
- Use short bullet points when listing several items."""

MAX_CONTEXT_ROWS = 150

_STOPWORDS = {
    "what", "which", "where", "when", "who", "how", "the", "and", "are", "is", "there", "this",
    "that", "with", "have", "has", "any", "about", "does", "from", "for", "give", "show", "tell",
    "list", "all", "campus", "feedback", "problem", "problems", "issue", "issues", "please", "can",
    "you", "your", "many", "much", "most", "some", "them", "they", "their", "into", "over",
}
_GENERIC_LOCATION_WORDS = {"area", "areas", "campus"}


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).lower().replace("-", ""))


def _row_line(row, with_action: bool = True) -> str:
    """One compact text line for a record (used in the AI prompt)."""
    parts = [
        str(getattr(row, "feedback_id", "")),
        str(getattr(row, "date", ""))[:10],
        str(getattr(row, "location", "")),
    ]
    if hasattr(row, "category"):
        parts += [str(row.category), str(row.sentiment), str(row.severity)]
    line = " | ".join(parts) + " | " + str(getattr(row, "feedback", ""))
    action = str(getattr(row, "suggested_action", "") or "")
    if with_action and action and action.lower() != "no action needed":
        line += f" | Action: {action}"
    return line


def build_chat_context(question: str, df: pd.DataFrame, datasets: dict | None = None) -> str:
    """Build the data context (all 5 datasets) that grounds the assistant's answer."""
    datasets = datasets or build_datasets(df)
    analyzed = is_analyzed(df)

    rows = df
    note = f"All {len(df)} records are listed."
    if len(df) > MAX_CONTEXT_ROWS:
        words = [w for w in re.findall(r"[a-z0-9]+", _norm_text(question)) if len(w) >= 4 and w not in _STOPWORDS]
        text = (df["feedback"].astype(str).map(_norm_text)
                + " " + df["location"].astype(str).map(_norm_text))
        score = sum(text.str.contains(w[:5], regex=False).astype(int) for w in words) if words else 0
        if analyzed:
            score = score * 10 + df["severity"].map(SEVERITY_WEIGHT).fillna(1)
        rows = df.assign(_score=score).sort_values("_score", ascending=False).head(MAX_CONTEXT_ROWS)
        note = f"Showing the {MAX_CONTEXT_ROWS} most relevant of {len(df)} records (summary datasets cover all {len(df)})."

    header = ("feedback_id | date | location | category | sentiment | severity | feedback | action"
              if analyzed else "feedback_id | date | location | feedback")
    record_lines = "\n".join(_row_line(r) for r in rows.itertuples(index=False))

    summary_text = datasets_to_text(datasets)
    if not analyzed:
        summary_text += ("\n(AI analysis has not been run yet, so category/sentiment/severity "
                         "datasets are not available.)")

    return (f"### Dataset: Feedback (Full) - {note}\n{header}\n{record_lines}\n\n{summary_text}")


# ----------------------------------------------------------------------------
# Local (no-AI) fallback - guarantees the assistant always answers from the data
# ----------------------------------------------------------------------------
def _sorted_by_severity(rows: pd.DataFrame) -> pd.DataFrame:
    if "severity" in rows.columns:
        return rows.assign(_w=rows["severity"].map(SEVERITY_WEIGHT).fillna(1)).sort_values(
            "_w", ascending=False)
    return rows


def _bullets(rows: pd.DataFrame, limit: int = 5) -> str:
    lines = []
    for r in rows.head(limit).itertuples(index=False):
        loc = getattr(r, "location", "")
        text = getattr(r, "summary", "") or getattr(r, "feedback", "")
        tag = f"{r.severity} severity, {r.category}" if hasattr(r, "category") else str(getattr(r, "date", ""))[:10]
        lines.append(f"- **{loc}** ({tag}): {text}")
    more = f"\n- ...and {len(rows) - limit} more." if len(rows) > limit else ""
    return "\n".join(lines) + more


def _mentioned_locations(q: str, df: pd.DataFrame) -> list:
    qn = _norm_text(q)
    found = []
    for loc in df["location"].dropna().unique():
        tokens = [t for t in _norm_text(loc).split() if t not in _GENERIC_LOCATION_WORDS and len(t) >= 3]
        if any(t in qn or (len(t) >= 6 and t[:5] in qn) for t in tokens):
            found.append(loc)
    return found


def local_answer(question: str, df: pd.DataFrame) -> str:
    """Rule-based answers computed directly from the DataFrame (used when the AI model is unavailable)."""
    if df is None or df.empty:
        return "The dataset does not contain enough information to answer that."

    q = question.lower().strip()
    analyzed = is_analyzed(df)
    problems = df[df["sentiment"].isin(["Negative", "Mixed"])] if analyzed else df
    has = lambda *words: any(w in q for w in words)

    # Greeting / help
    if q.rstrip("!?. ") in {"hi", "hello", "hey", "help"} or has("what can you", "how do you work"):
        return ("Hi! I answer questions using the campus feedback datasets. Try: "
                "*What is the most common problem?*, *Which location has the most complaints?*, "
                "*What are the major problems in the library?*, *What problems have high severity?*, "
                "or *What improvements are commonly suggested?*")

    locs = _mentioned_locations(q, df)
    if locs:
        rows = df[df["location"].isin(locs)]
        shown = _sorted_by_severity(rows[rows["sentiment"].isin(["Negative", "Mixed"])] if analyzed else rows)
        name = ", ".join(locs)
        if analyzed and shown.empty:
            return f"**{name}** has {len(rows)} report(s) and none of them are negative or mixed."
        extra = f" ({len(shown)} negative/mixed)" if analyzed else ""
        return f"**{name}** has {len(rows)} report(s){extra}:\n{_bullets(shown)}"

    if has("location", "where", "which place", "which area") and has("most", "top", "worst", "highest", "complaint"):
        counts = problems["location"].value_counts()
        top = counts.head(3)
        label = "negative/mixed reports" if analyzed else "reports"
        return f"**{top.index[0]}** has the most {label} ({top.iloc[0]}). Next: " + \
            ", ".join(f"{k} ({v})" for k, v in top.iloc[1:].items()) + "."

    if analyzed and has("high severity", "severe", "urgent", "serious", "critical", "high priority") or \
            (analyzed and "high" in q):
        rows = _sorted_by_severity(df[df["severity"] == "High"])
        if rows.empty:
            return "No feedback is marked High severity in the current data."
        return f"{len(rows)} report(s) are High severity:\n{_bullets(rows, 6)}"

    if has("improve", "suggest", "action", "solution", "recommend", "fix"):
        if not analyzed or "suggested_action" not in df.columns:
            return "Run **Analyze Feedback** first - suggested actions are generated by the AI analysis."
        acts = df["suggested_action"].astype(str)
        acts = acts[~acts.str.lower().isin(["no action needed", "review manually.", ""])]
        if acts.empty:
            return "The dataset does not contain enough information to answer that."
        top = acts.value_counts().head(5)
        return "Commonly suggested improvements:\n" + "\n".join(f"- {a}" for a in top.index)

    if has("negative", "complain", "summarize", "summary"):
        if not analyzed:
            return "Run **Analyze Feedback** first so I can tell negative feedback apart."
        neg = df[df["sentiment"] == "Negative"]
        if neg.empty:
            return "There is no negative feedback in the current data."
        cats = ", ".join(f"{k} ({v})" for k, v in neg["category"].value_counts().head(3).items())
        locs_top = ", ".join(f"{k} ({v})" for k, v in neg["location"].value_counts().head(3).items())
        return (f"There are {len(neg)} negative reports. Top categories: {cats}. "
                f"Top locations: {locs_top}.\n{_bullets(_sorted_by_severity(neg))}")

    if analyzed and has("positive", "praise", "compliment", "good"):
        pos = df[df["sentiment"] == "Positive"]
        return f"There are {len(pos)} positive reports:\n{_bullets(pos)}" if not pos.empty \
            else "There is no positive feedback in the current data."

    if analyzed and has("common", "most", "top", "main", "biggest", "major", "frequent", "category", "categories"):
        counts = problems["category"].value_counts()
        if not counts.empty:
            return f"The most common problem category is **{counts.index[0]}** ({counts.iloc[0]} reports). " \
                   "Next: " + ", ".join(f"{k} ({v})" for k, v in counts.iloc[1:4].items()) + "."

    if has("how many", "total", "number of", "count"):
        if analyzed:
            s = df["sentiment"].value_counts()
            return f"There are {len(df)} feedback records: " + ", ".join(f"{k} {v}" for k, v in s.items()) + "."
        return f"There are {len(df)} feedback records across {df['location'].nunique()} locations."

    if analyzed:
        cats = [c for c in df["category"].unique() if c.lower() in q]
        if cats:
            rows = _sorted_by_severity(df[df["category"].isin(cats)])
            return f"{len(rows)} report(s) in **{', '.join(cats)}**:\n{_bullets(rows)}"

    # Keyword search across the feedback text
    words = [w for w in re.findall(r"[a-z0-9]+", _norm_text(q)) if len(w) >= 4 and w not in _STOPWORDS]
    if words:
        text = df["feedback"].astype(str).map(_norm_text)
        score = sum(text.str.contains(w[:5], regex=False).astype(int) for w in words)
        hits = _sorted_by_severity(df[score > 0])
        if not hits.empty:
            return f"Found {len(hits)} related report(s):\n{_bullets(hits)}"

    return "The dataset does not contain enough information to answer that."


def chatbot_answer(question: str, df: pd.DataFrame, history: list | None = None,
                   datasets: dict | None = None) -> dict:
    """
    Answer a question about the datasets. Returns {"answer", "mode", "error"}.
    mode == "ai": answered by the Hugging Face model, grounded in all 5 datasets.
    mode == "local": AI unavailable (no token / API error) - answered directly from the data.
    The assistant therefore always responds.
    """
    if df is None or df.empty:
        return {"answer": "The dataset does not contain enough information to answer that.",
                "mode": "local", "error": None}

    prompt = f"{build_chat_context(question, df, datasets)}\n\nQuestion: {question}"
    recent = [{"role": m["role"], "content": m["content"]}
              for m in (history or [])[-6:] if m.get("role") in ("user", "assistant")]
    try:
        client = get_client()
        answer = _chat(client, CHATBOT_SYSTEM_PROMPT, prompt,
                       max_tokens=700, temperature=0.2, history=recent)
        return {"answer": answer, "mode": "ai", "error": None}
    except AIAnalysisError as exc:
        return {"answer": local_answer(question, df), "mode": "local", "error": str(exc)}