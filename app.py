"""
app.py
------
CampusSense AI - An AI-Powered Campus Feedback and Problem Detection System.

CS 315: Application Development and Emerging Technologies - Activity 3.

This is the main Streamlit entry point. It wires together:
  - utils/data_cleaning.py  (loading + cleaning the feedback CSV with Pandas)
  - utils/ai_analysis.py    (GenAI-powered analysis, dataset summary, and the
                              CampusSense Assistant chatbot)
  - utils/datasets.py       (the 5 datasets: full, by location/category/sentiment/severity)
  - utils/visualization.py  (Plotly charts)

Run locally with:  streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.data_cleaning import load_data, clean_data, DataValidationError
from utils.ai_analysis import (
    analyze_dataframe,
    generate_campus_summary,
    chatbot_answer,
    get_hf_token,
    AIAnalysisError,
    VALID_CATEGORIES,
    VALID_SENTIMENTS,
    VALID_SEVERITIES,
)
from utils.datasets import build_datasets, is_analyzed, DATASET_NAMES, DATASET_DESCRIPTIONS
from utils import visualization as viz

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()  # loads HF_TOKEN from a local .env file (never hardcoded)

SAMPLE_DATA_PATH = os.path.join("data", "campus_feedback.csv")

st.set_page_config(
    page_title="CampusSense AI",
    page_icon="🎓",
    layout="wide",
)

# Session state defaults - keeps analysis results and chat history across reruns
if "analyzed_df" not in st.session_state:
    st.session_state.analyzed_df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

EXAMPLE_QUESTIONS = [
    "What is the most common problem?",
    "Which location has the most complaints?",
    "What are the major problems in the library?",
    "Summarize the negative feedback.",
    "What problems have high severity?",
    "What improvements are commonly suggested?",
]


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def safe_load_and_clean(file_path_or_buffer):
    """Load + clean a feedback CSV, surfacing friendly errors instead of crashing."""
    try:
        raw = load_data(file_path_or_buffer)
        cleaned = clean_data(raw)
        return cleaned, None
    except DataValidationError as exc:
        return None, str(exc)
    except Exception as exc:  # anything unexpected
        return None, f"Unexpected error while loading the dataset: {exc}"


# Filters are relaxed in this order when nothing matches: least important first.
RELAX_ORDER = [
    ("filter_severity", "Severity"),
    ("filter_sentiment", "Sentiment"),
    ("filter_category", "Category"),
    ("filter_date_range", "Date range"),
    ("filter_location", "Location"),
]


def relax_filters(df: pd.DataFrame):
    """
    When the exact filter combination has no records, find the smallest relaxation that does:
    drop one filter, then two, ... in RELAX_ORDER. Returns (DataFrame, [names of ignored filters]).
    """
    from itertools import combinations

    active = [(k, label) for k, label in RELAX_ORDER
              if st.session_state.get(k)]
    for size in range(1, len(active) + 1):
        for combo in combinations(active, size):
            result = apply_filters(df, skip={k for k, _ in combo})
            if not result.empty:
                return result, [label for _, label in combo]
    return df, [label for _, label in active]


def explain_empty_filters(df: pd.DataFrame) -> str:
    """Say how many records each selected filter matches ON ITS OWN, so it is clear which combination is empty."""
    lines = []
    for state_key, column, label in [
        ("filter_location", "location", "Location"),
        ("filter_category", "category", "Category"),
        ("filter_sentiment", "sentiment", "Sentiment"),
        ("filter_severity", "severity", "Severity"),
    ]:
        selected = st.session_state.get(state_key)
        if selected and column in df.columns:
            n = int(df[column].isin(selected).sum())
            lines.append(f"- {label} ({', '.join(selected)}): {n} record(s) on its own")
    hint = ""
    sent, sev = st.session_state.get("filter_sentiment"), st.session_state.get("filter_severity")
    if sent and sev and set(sent) <= {"Positive", "Neutral"} and set(sev) == {"High"}:
        hint = "\n\nPositive or neutral feedback is almost never High severity (High means a serious problem)."
    return "\n".join(lines) + hint


def apply_filters(df: pd.DataFrame, skip=()) -> pd.DataFrame:
    """
    Apply the sidebar filters. An empty multiselect means "All".
    Category / sentiment / severity filters only take effect once the data is analyzed.
    `skip` is a collection of filter state keys to ignore (used to relax an empty result).
    """
    if df is None or df.empty:
        return df

    filtered = df.copy()

    for state_key, column in [
        ("filter_location", "location"),
        ("filter_category", "category"),
        ("filter_sentiment", "sentiment"),
        ("filter_severity", "severity"),
    ]:
        selected = st.session_state.get(state_key)
        if state_key not in skip and selected and column in filtered.columns:
            filtered = filtered[filtered[column].isin(selected)]

    date_range = st.session_state.get("filter_date_range")
    if "filter_date_range" not in skip and date_range and len(date_range) == 2 and "date" in filtered.columns:
        start, end = date_range
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(start)) & (filtered["date"] <= pd.Timestamp(end))
        ]

    return filtered


def compute_priority_problems(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank problem categories by a priority score combining:
      - Frequency: how many reports fall into the category (40%)
      - Severity:  the average severity of those reports (40%)
      - Recency:   how recently the category was last reported (20%)

    This ranks PROBLEM TYPES (categories), never individual students.
    """
    if df.empty or "category" not in df.columns:
        return pd.DataFrame(columns=["Priority Problem", "Reports", "Severity", "Priority Score"])

    weight_map = {"Low": 1, "Medium": 2, "High": 3}
    work = df.copy()
    work["severity_weight"] = work["severity"].map(weight_map).fillna(1)

    grouped = work.groupby("category").agg(
        reports=("category", "count"),
        avg_severity_weight=("severity_weight", "mean"),
        most_recent=("date", "max"),
    ).reset_index()

    max_reports = grouped["reports"].max() or 1
    grouped["freq_score"] = grouped["reports"] / max_reports
    grouped["severity_score"] = grouped["avg_severity_weight"] / 3

    latest_overall = work["date"].max()
    grouped["days_since_last_report"] = (latest_overall - grouped["most_recent"]).dt.days
    max_days = grouped["days_since_last_report"].max()
    max_days = max_days if max_days and max_days > 0 else 1
    grouped["recency_score"] = 1 - (grouped["days_since_last_report"] / max_days)

    grouped["priority_score"] = (
        grouped["freq_score"] * 0.4
        + grouped["severity_score"] * 0.4
        + grouped["recency_score"] * 0.2
    ) * 100
    grouped["priority_score"] = grouped["priority_score"].round(1)

    grouped["severity_label"] = grouped["avg_severity_weight"].apply(
        lambda w: "High" if w >= 2.5 else ("Medium" if w >= 1.5 else "Low")
    )

    grouped = grouped.sort_values("priority_score", ascending=False)
    result = grouped.rename(columns={
        "category": "Priority Problem",
        "reports": "Reports",
        "severity_label": "Severity",
        "priority_score": "Priority Score",
    })
    return result[["Priority Problem", "Reports", "Severity", "Priority Score"]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Load data - the bundled CSV is loaded automatically (no upload needed).
# This happens BEFORE the sidebar is drawn so the filters can always be shown.
# ----------------------------------------------------------------------------
if st.session_state.raw_df is None:
    cleaned_df, error = safe_load_and_clean(SAMPLE_DATA_PATH)
    if error:
        st.error(f"⚠️ {error}")
        st.stop()
    st.session_state.raw_df = cleaned_df

analyzed = st.session_state.analyzed_df is not None
base_df = st.session_state.analyzed_df if analyzed else st.session_state.raw_df

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
header_cols = st.columns([1, 6])
with header_cols[0]:
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with header_cols[1]:
    st.title("CampusSense AI")
    st.caption("Campus Feedback and Problem Detection System")

st.write(
    "CampusSense AI analyzes student and campus feedback using a Generative AI model to "
    "automatically detect recurring problems, categorize them, gauge severity and sentiment, "
    "extract keywords, and suggest possible solutions - helping campus staff prioritize what "
    "to fix first."
)
st.divider()

# ----------------------------------------------------------------------------
# Sidebar - filters (always shown) + analyze button
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data & Filters")
    st.caption("Dataset: data/campus_feedback.csv (loaded automatically)")

    st.markdown("---")
    st.subheader("Filters")
    st.caption("Leave a filter empty to include everything.")

    st.multiselect(
        "Location",
        sorted(base_df["location"].dropna().unique().tolist()),
        key="filter_location",
        placeholder="All locations",
    )
    st.multiselect("Category", VALID_CATEGORIES, key="filter_category", placeholder="All categories")
    st.multiselect("Sentiment", VALID_SENTIMENTS, key="filter_sentiment", placeholder="All sentiments")
    st.multiselect("Severity", VALID_SEVERITIES, key="filter_severity", placeholder="All severities")

    min_date = base_df["date"].min()
    max_date = base_df["date"].max()
    if pd.notna(min_date) and pd.notna(max_date):
        st.date_input(
            "Date range",
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
            key="filter_date_range",
        )

    if not analyzed:
        st.caption("Category, sentiment and severity filters apply after you click Analyze Feedback.")

    st.markdown("---")
    analyze_clicked = st.button("🔍 Analyze Feedback", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Run AI analysis on demand
# ----------------------------------------------------------------------------
if analyze_clicked:
    if st.session_state.raw_df is None or st.session_state.raw_df.empty:
        st.error("⚠️ The dataset could not be loaded. Check that data/campus_feedback.csv exists.")
    elif not get_hf_token():
        st.error(
            "⚠️ HF_TOKEN is not set. Add your Hugging Face token to a .env file "
            "(or Streamlit secrets) - see README.md - before running analysis."
        )
    else:
        progress_bar = st.progress(0, text="Analyzing feedback with GenAI...")

        def _update_progress(current, total):
            progress_bar.progress(current / total, text=f"Analyzing feedback... ({current}/{total})")

        try:
            result_df = analyze_dataframe(st.session_state.raw_df, progress_callback=_update_progress)
            st.session_state.analyzed_df = result_df
            progress_bar.progress(1.0, text="Analysis complete!")
            st.success("✅ Feedback analysis complete.")
            if "analysis_error" in result_df.columns and result_df["analysis_error"].notna().any():
                failed_rows = result_df[result_df["analysis_error"].notna()]
                st.warning(
                    f"⚠️ The AI model could not analyze {len(failed_rows)} of {len(result_df)} rows, so simple "
                    "keyword rules were used for them (results are less accurate). "
                    f"First error: {str(failed_rows['analysis_error'].iloc[0])[:300]}"
                )
            analyzed = True
            base_df = result_df
        except AIAnalysisError as exc:
            st.error(f"⚠️ AI analysis failed: {exc}")
        except Exception as exc:
            st.error(f"⚠️ Unexpected error during analysis: {exc}")

# ----------------------------------------------------------------------------
# Filtered data + the 5 datasets
# ----------------------------------------------------------------------------
filtered_df = apply_filters(base_df)

if filtered_df.empty:
    detail = explain_empty_filters(base_df)
    relaxed_df, ignored = relax_filters(base_df)
    if relaxed_df.empty:
        st.warning("No feedback is available for this dataset.")
        st.stop()
    st.warning(
        "No feedback matches ALL of the selected filters, so the closest results are shown instead "
        f"(ignoring: {', '.join(ignored)})."
    )
    if detail:
        st.caption("Each filter alone would match:\n\n" + detail)
    filtered_df = relaxed_df

datasets = build_datasets(filtered_df)

# ----------------------------------------------------------------------------
# Dashboard metrics (after analysis)
# ----------------------------------------------------------------------------
if analyzed:
    total_feedback = len(filtered_df)
    positive_count = int((filtered_df["sentiment"] == "Positive").sum())
    negative_count = int((filtered_df["sentiment"] == "Negative").sum())
    high_priority_count = int((filtered_df["severity"] == "High").sum())
    most_reported_category = filtered_df["category"].value_counts().idxmax()
    most_affected_location = filtered_df["location"].value_counts().idxmax()

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total Feedback", total_feedback)
    metric_cols[1].metric("Positive", positive_count)
    metric_cols[2].metric("Negative", negative_count)
    metric_cols[3].metric("High Priority", high_priority_count)
    metric_cols[4].metric("Top Category", most_reported_category)
    metric_cols[5].metric("Top Location", most_affected_location)
    st.divider()

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_dashboard, tab_analysis, tab_priority, tab_assistant, tab_dataset = st.tabs(
    ["📊 Dashboard", "🔎 Feedback Analysis", "🚩 Priority Problems", "🤖 AI Assistant", "🗂️ Datasets"]
)

NEEDS_ANALYSIS_MSG = (
    "👋 The campus feedback dataset is already loaded. "
    "Click **Analyze Feedback** in the sidebar to run the AI analysis and unlock this section."
)

# --- Dashboard tab: AI summary + charts ---
with tab_dashboard:
    if not analyzed:
        st.info(NEEDS_ANALYSIS_MSG)
        st.subheader("Preview of loaded (cleaned) data")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    else:
        st.subheader("Campus Problem Summary")
        with st.expander("AI-generated summary of the current (filtered) data", expanded=True):
            if st.button("Generate / Refresh Summary"):
                try:
                    with st.spinner("Asking CampusSense AI for a summary..."):
                        summary_text = generate_campus_summary(filtered_df)
                    st.session_state["campus_summary"] = summary_text
                except AIAnalysisError as exc:
                    st.error(f"⚠️ {exc}")
            st.write(st.session_state.get("campus_summary", "Click the button above to generate a summary."))

        chart_row = st.columns(2)
        with chart_row[0]:
            fig = viz.sentiment_distribution_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        with chart_row[1]:
            fig = viz.severity_chart(filtered_df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        fig = viz.location_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

# --- Feedback Analysis tab: interactive detail table ---
with tab_analysis:
    if not analyzed:
        st.info(NEEDS_ANALYSIS_MSG)
    else:
        st.subheader("Feedback Details")
        st.caption("Full breakdown of every analyzed feedback record (respects sidebar filters).")

        search_text = st.text_input("Search within feedback text")
        display_df = filtered_df.copy()
        if search_text:
            display_df = display_df[display_df["feedback"].str.contains(search_text, case=False, na=False)]

        display_columns = [
            "feedback_id", "date", "location", "feedback", "category",
            "sentiment", "severity", "keywords", "summary", "suggested_action",
        ]
        display_columns = [c for c in display_columns if c in display_df.columns]
        st.dataframe(display_df[display_columns], use_container_width=True, height=450)

# --- Priority Problems tab ---
with tab_priority:
    if not analyzed:
        st.info(NEEDS_ANALYSIS_MSG)
    else:
        st.subheader("Priority Problems")
        st.write(
            "Problems are ranked using a **Priority Score** that combines three factors:\n\n"
            "- **Frequency (40%)** - how many reports fall into this problem category, "
            "relative to the most-reported category.\n"
            "- **Severity (40%)** - the average severity (Low=1, Medium=2, High=3) of reports "
            "in that category, relative to the maximum possible severity.\n"
            "- **Recency (20%)** - how recently the category was last reported, relative to the "
            "most recent report in the filtered dataset.\n\n"
            "`Priority Score = (Frequency Score × 0.4 + Severity Score × 0.4 + Recency Score × 0.2) × 100`\n\n"
            "This ranks **problem types**, not individual students."
        )
        priority_df = compute_priority_problems(filtered_df)
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

# --- AI Assistant tab: chatbot connected to the 5 datasets ---
with tab_assistant:
    st.subheader("CampusSense Assistant")
    st.caption(
        f"Connected to all 5 datasets ({len(filtered_df)} records, sidebar filters applied). "
        "It only answers from this data and says so when the data isn't enough."
    )
    if not analyzed:
        st.caption("Tip: click **Analyze Feedback** first so I can also use category, sentiment and severity.")

    # Example questions as one-click buttons
    st.write("Try one of these:")
    example_cols = st.columns(3)
    clicked_question = None
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if example_cols[i % 3].button(example, key=f"example_{i}", use_container_width=True):
            clicked_question = example

    if st.session_state.chat_history and st.button("🧹 Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("mode") == "local":
                st.caption("ℹ️ AI model unavailable - answered directly from the dataset.")

    typed_question = st.chat_input("Ask CampusSense Assistant about this dataset...")
    user_question = typed_question or clicked_question
    if user_question:
        history_for_model = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chatbot_answer(user_question, filtered_df, history=history_for_model, datasets=datasets)
            st.markdown(result["answer"])
            if result["mode"] == "local":
                st.caption("ℹ️ AI model unavailable - answered directly from the dataset.")
                if result["error"]:
                    with st.expander("Why?"):
                        st.code(result["error"])
        st.session_state.chat_history.append(
            {"role": "assistant", "content": result["answer"], "mode": result["mode"]}
        )

# --- Datasets tab: the 5 datasets ---
with tab_dataset:
    st.subheader("Datasets")
    st.caption("5 datasets built from the current (filtered) data. Each one can be downloaded as CSV.")

    choice = st.radio("Choose a dataset", DATASET_NAMES, horizontal=True, label_visibility="collapsed")
    st.caption(DATASET_DESCRIPTIONS[choice])

    table = datasets.get(choice)
    if table is None or table.empty:
        st.info("This dataset is available after you click **Analyze Feedback** in the sidebar.")
    else:
        st.dataframe(table, use_container_width=True, height=450, hide_index=True)
        file_slug = choice.lower().replace(" ", "_").replace("(", "").replace(")", "")
        st.download_button(
            label=f"⬇️ Download {choice} (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"campus_feedback_{file_slug}.csv",
            mime="text/csv",
        )

     