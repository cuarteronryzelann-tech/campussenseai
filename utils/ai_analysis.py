"""
ai_analysis.py
--------------
GenAI (Google Gemini API) integration for CampusSense AI.

This module is responsible for:
  1. Analyzing individual feedback entries and extracting structured data
     (sentiment, category, severity, keywords, main issue, summary, action).
  2. Generating an overall AI summary of the (filtered) dataset.
  3. Answering natural-language questions about the dataset through the
     "CampusSense Assistant" chatbot.

Uses Google's Gemini 3.6 Flash model via the official `google-genai` SDK.
The Gemini API key is NEVER hardcoded here - it is read from the
GEMINI_API_KEY environment variable (loaded from a local .env file via
python-dotenv in app.py). If the key is missing, functions raise a
clear AIAnalysisError instead of crashing the app.
"""

import os
import json

import pandas as pd
from google import genai
from google.genai import types

# Allowed values the AI must choose from. Keeping these as constants means
# we can validate the AI's JSON output and fall back safely if it drifts.
VALID_SENTIMENTS = ["Positive", "Neutral", "Negative", "Mixed"]
VALID_CATEGORIES = [
    "Internet", "Hardware", "Software", "Facilities", "Cleanliness",
    "Food Service", "Transportation", "Safety", "Environment",
    "Academic", "Other",
]
VALID_SEVERITIES = ["Low", "Medium", "High"]

MODEL_NAME = "gemini-3.6-flash"  # Google's fast, cost-efficient Gemini 3 model - good fit for a classroom project

# Gemini 3 models use "thinking levels" instead of temperature/top_p/top_k.
# LOW keeps per-feedback classification fast and cheap since this is a
# straightforward extraction task, not deep multi-step reasoning.
CLASSIFICATION_THINKING = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)


class AIAnalysisError(Exception):
    """Raised when the GenAI API cannot be reached or returns something unusable."""
    pass


def get_client() -> genai.Client:
    """
    Build a Gemini API client using the API key from the environment.
    Raises AIAnalysisError with a friendly message if the key is missing.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise AIAnalysisError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see README.md) before running analysis."
        )
    return genai.Client(api_key=api_key)


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


def analyze_feedback(client: genai.Client, feedback_text: str) -> dict:
    """
    Send a single feedback entry to the Gemini API and return a structured dict.
    Never raises for a single-row failure - instead returns a fallback result
    with an "analysis_error" key so the caller can surface it without crashing
    the whole batch.
    """
    if not feedback_text or not str(feedback_text).strip():
        return _build_fallback_result("Empty feedback text.")

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=f"Feedback: {feedback_text}",
            config=types.GenerateContentConfig(
                system_instruction=ANALYSIS_SYSTEM_PROMPT,
                response_mime_type="application/json",
                thinking_config=CLASSIFICATION_THINKING,
            ),
        )
        raw_content = response.text
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        return _build_fallback_result("AI returned an invalid (non-JSON) response.")
    except Exception as exc:  # covers network errors, auth errors, rate limits, etc.
        return _build_fallback_result(f"API error: {exc}")

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
    dataset using the Gemini API. This is always derived from the current
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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You write short, factual dashboard summaries.",
                thinking_config=CLASSIFICATION_THINKING,
            ),
        )
        return response.text.strip()
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
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CHATBOT_SYSTEM_PROMPT,
                thinking_config=CLASSIFICATION_THINKING,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        raise AIAnalysisError(f"Chatbot could not respond: {exc}")
