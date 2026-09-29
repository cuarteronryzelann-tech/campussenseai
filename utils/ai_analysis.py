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

import pandas as pd
from huggingface_hub import InferenceClient

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


class AIAnalysisError(Exception):
    """Raised when the GenAI API cannot be reached or returns something unusable."""
    pass


def get_hf_token() -> str | None:
    """Return the Hugging Face token from the environment (or None)."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


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
          max_tokens: int = 500, temperature: float = 0.2) -> str:
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
        try:
            response = client.chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
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
            errors.append(f"{model}: {exc}")

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


def _build_fallback_result(error_message: str) -> dict:
    """A safe, clearly-marked placeholder result used when the AI call fails."""
    return {
        "sentiment": "Neutral",
        "category": "Other",
        "severity": "Low",
        "keywords": [],
        "main_issue": "Analysis unavailable",
        "summary": "This feedback could not be analyzed automatically.",
        "suggested_action": "Review manually.",
        "analysis_error": error_message,
    }


def analyze_feedback(client: InferenceClient, feedback_text: str) -> dict:
    """
    Send a single feedback entry to the Hugging Face model and return a structured dict.
    Never raises for a single-row failure - instead returns a fallback result
    with an "analysis_error" key so the caller can surface it without crashing
    the whole batch.
    """
    if not feedback_text or not str(feedback_text).strip():
        return _build_fallback_result("Empty feedback text.")

    try:
        raw_content = _chat(
            client,
            system=ANALYSIS_SYSTEM_PROMPT,
            user=f"Feedback: {feedback_text}",
            max_tokens=400,
            temperature=0.1,
        )
        result = _extract_json(raw_content)
    except json.JSONDecodeError:
        return _build_fallback_result("AI returned an invalid (non-JSON) response.")
    except Exception as exc:  # covers network errors, auth errors, rate limits, etc.
        return _build_fallback_result(f"API error: {exc}")

    if not isinstance(result, dict):
        return _build_fallback_result("AI returned an unexpected response format.")

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


def analyze_dataframe(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Run analyze_feedback() on every row of a cleaned feedback DataFrame and
    return a new DataFrame with the extracted columns appended.

    progress_callback, if given, is called with (current_index, total_rows)
    after each row so the Streamlit UI can show a progress bar.
    """
    client = get_client()  # raises AIAnalysisError early if no API key at all

    results = []
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        analysis = analyze_feedback(client, getattr(row, "feedback", ""))
        results.append(analysis)
        if progress_callback:
            progress_callback(i, total)

    analysis_df = pd.DataFrame(results)
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


CHATBOT_SYSTEM_PROMPT = """You are "CampusSense Assistant", a chatbot that answers questions
about a specific set of analyzed campus feedback data. You will be given statistics and
sample records from the dataset. Answer the user's question using ONLY that information.

Rules:
- Never invent facts, numbers, or issues that are not present in the provided data.
- If the data provided is not sufficient to answer the question, say clearly:
  "The dataset does not contain enough information to answer that."
- Keep answers concise and specific (mention numbers/locations/categories when relevant).
"""


def chatbot_answer(question: str, df: pd.DataFrame) -> str:
    """
    Answer a natural-language question about the analyzed dataset.
    Grounds the model in aggregate statistics plus a small set of relevant
    sample rows so it cannot fabricate information beyond the dataset.
    """
    if df.empty:
        return "The dataset does not contain enough information to answer that."

    client = get_client()
    stats_text = _dataset_stats_text(df)

    # Pull a handful of rows whose location/category/feedback text loosely
    # match the question, to give the model concrete grounding examples.
    sample_rows = df
    lowered_question = question.lower()
    if "location" in df.columns:
        matches = df[df["location"].str.lower().apply(lambda loc: loc in lowered_question)]
        if not matches.empty:
            sample_rows = matches
    if "category" in df.columns and sample_rows is df:
        matches = df[df["category"].str.lower().apply(lambda cat: cat in lowered_question)]
        if not matches.empty:
            sample_rows = matches

    sample_rows = sample_rows.head(8)
    sample_lines = []
    for row in sample_rows.itertuples(index=False):
        location = getattr(row, "location", "Unknown")
        category = getattr(row, "category", "Unknown")
        sentiment = getattr(row, "sentiment", "Unknown")
        severity = getattr(row, "severity", "Unknown")
        summary = getattr(row, "summary", getattr(row, "feedback", ""))
        sample_lines.append(f"- [{location} | {category} | {sentiment} | {severity}] {summary}")

    prompt = f"""Dataset statistics:
{stats_text}

Relevant sample records:
{chr(10).join(sample_lines) if sample_lines else "None available"}

Question: {question}
"""

    try:
        return _chat(
            client,
            system=CHATBOT_SYSTEM_PROMPT,
            user=prompt,
            max_tokens=500,
            temperature=0.2,
        )
    except Exception as exc:
        raise AIAnalysisError(f"Chatbot could not respond: {exc}")
