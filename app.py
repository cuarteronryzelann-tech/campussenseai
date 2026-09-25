"""
app.py
------
CampusSense AI - An AI-Powered Campus Feedback and Problem Detection System.

CS 315: Application Development and Emerging Technologies - Activity 3.

This is the main Streamlit entry point. It wires together:
  - utils/data_cleaning.py  (loading + cleaning the feedback CSV with Pandas)
  - utils/ai_analysis.py    (GenAI-powered sentiment/category/severity extraction,
                              dataset summary, and the CampusSense Assistant chatbot)
  - utils/visualization.py  (Plotly charts)

Run locally with:  streamlit run app.py
"""

import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from utils.data_cleaning import load_data, clean_data, DataValidationError
from utils.ai_analysis import (
    analyze_dataframe,
    generate_campus_summary,
    chatbot_answer,
    AIAnalysisError,
)
from utils import visualization as viz

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()  # loads GEMINI_API_KEY from a local .env file (never hardcoded)

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


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar filters to the analyzed DataFrame."""
    if df is None or df.empty:
        return df

    filtered = df.copy()

    if st.session_state.get("filter_location") and "All" not in st.session_state.filter_location:
        filtered = filtered[filtered["location"].isin(st.session_state.filter_location)]

    if "category" in filtered.columns and st.session_state.get("filter_category") \
            and "All" not in st.session_state.filter_category:
        filtered = filtered[filtered["category"].isin(st.session_state.filter_category)]

    if "sentiment" in filtered.columns and st.session_state.get("filter_sentiment") \
            and "All" not in st.session_state.filter_sentiment:
        filtered = filtered[filtered["sentiment"].isin(st.session_state.filter_sentiment)]

    if "severity" in filtered.columns and st.session_state.get("filter_severity") \
            and "All" not in st.session_state.filter_severity:
        filtered = filtered[filtered["severity"].isin(st.session_state.filter_severity)]

    date_range = st.session_state.get("filter_date_range")
    if date_range and len(date_range) == 2 and "date" in filtered.columns:
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
# Sidebar - data upload, filters, analyze button
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Data & Filters")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample dataset", value=(uploaded_file is None))

    st.markdown("---")
    st.subheader("Filters")

    base_df = st.session_state.analyzed_df if st.session_state.analyzed_df is not None else st.session_state.raw_df

    if base_df is not None and not base_df.empty:
        location_options = ["All"] + sorted(base_df["location"].dropna().unique().tolist())
        st.multiselect("Location", location_options, default=["All"], key="filter_location")

        if "category" in base_df.columns:
            category_options = ["All"] + sorted(base_df["category"].dropna().unique().tolist())
            st.multiselect("Category", category_options, default=["All"], key="filter_category")
        else:
            st.caption("Category filter available after analysis.")

        if "sentiment" in base_df.columns:
            sentiment_options = ["All"] + sorted(base_df["sentiment"].dropna().unique().tolist())
            st.multiselect("Sentiment", sentiment_options, default=["All"], key="filter_sentiment")
        else:
            st.caption("Sentiment filter available after analysis.")

        if "severity" in base_df.columns:
            severity_options = ["All"] + sorted(base_df["severity"].dropna().unique().tolist())
            st.multiselect("Severity", severity_options, default=["All"], key="filter_severity")
        else:
            st.caption("Severity filter available after analysis.")

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
    else:
        st.caption("Load a dataset to see filter options.")

    st.markdown("---")
    analyze_clicked = st.button("🔍 Analyze Feedback", use_container_width=True, type="primary")

# ----------------------------------------------------------------------------
# Load data (upload takes priority over sample)
# ----------------------------------------------------------------------------
data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif use_sample:
    data_source = SAMPLE_DATA_PATH

if data_source is not None and st.session_state.raw_df is None:
    cleaned_df, error = safe_load_and_clean(data_source)
    if error:
        st.error(f"⚠️ {error}")
    else:
        st.session_state.raw_df = cleaned_df

# Re-load if a new file was uploaded (different from what's cached)
if uploaded_file is not None:
    cleaned_df, error = safe_load_and_clean(uploaded_file)
    if error:
        st.error(f"⚠️ {error}")
        st.stop()
    else:
        st.session_state.raw_df = cleaned_df

# ----------------------------------------------------------------------------
# Run AI analysis on demand
# ----------------------------------------------------------------------------
if analyze_clicked:
    if st.session_state.raw_df is None or st.session_state.raw_df.empty:
        st.error("⚠️ No dataset loaded. Please upload a CSV or enable the sample dataset first.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error(
            "⚠️ GEMINI_API_KEY is not set. Add your API key to a .env file "
            "(see README.md) before running analysis."
        )
    else:
        progress_bar = st.progress(0, text="Analyzing feedback with GenAI...")

        def _update_progress(current, total):
            progress_bar.progress(current / total, text=f"Analyzing feedback... ({current}/{total})")

        try:
            analyzed = analyze_dataframe(st.session_state.raw_df, progress_callback=_update_progress)
            st.session_state.analyzed_df = analyzed
            progress_bar.progress(1.0, text="Analysis complete!")
            st.success("✅ Feedback analysis complete.")
        except AIAnalysisError as exc:
            st.error(f"⚠️ AI analysis failed: {exc}")
        except Exception as exc:
            st.error(f"⚠️ Unexpected error during analysis: {exc}")

# ----------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------
if st.session_state.analyzed_df is None:
    st.info(
        "👋 Load a dataset from the sidebar (the bundled sample is enabled by default) "
        "and click **Analyze Feedback** to run the AI analysis and unlock the dashboard."
    )
    if st.session_state.raw_df is not None:
        st.subheader("Preview of loaded (cleaned) data")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
    st.stop()

filtered_df = apply_filters(st.session_state.analyzed_df)

if filtered_df.empty:
    st.warning("No feedback matches the current filters. Try widening your filter selection.")
    st.stop()

# ----------------------------------------------------------------------------
# Dashboard metrics
# ----------------------------------------------------------------------------
total_feedback = len(filtered_df)
positive_count = int((filtered_df["sentiment"] == "Positive").sum())
negative_count = int((filtered_df["sentiment"] == "Negative").sum())
high_priority_count = int((filtered_df["severity"] == "High").sum())
most_reported_category = (
    filtered_df["category"].value_counts().idxmax() if not filtered_df["category"].empty else "N/A"
)
most_affected_location = (
    filtered_df["location"].value_counts().idxmax() if not filtered_df["location"].empty else "N/A"
)

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
    ["📊 Dashboard", "🔎 Feedback Analysis", "🚩 Priority Problems", "🤖 AI Assistant", "🗂️ Dataset"]
)

# --- Dashboard tab: charts + AI summary ---
with tab_dashboard:
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

    chart_row1 = st.columns(2)
    with chart_row1[0]:
        fig = viz.sentiment_distribution_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with chart_row1[1]:
        fig = viz.severity_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    chart_row2 = st.columns(2)
    with chart_row2[0]:
        fig = viz.category_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with chart_row2[1]:
        fig = viz.location_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    fig = viz.problems_over_time_chart(filtered_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    fig = viz.top_keywords_chart(filtered_df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No keywords available to chart yet.")

# --- Feedback Analysis tab: interactive detail table ---
with tab_analysis:
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

# --- AI Assistant tab: chatbot ---
with tab_assistant:
    st.subheader("CampusSense Assistant")
    st.caption(
        "Ask questions about the currently loaded and filtered dataset. "
        "The assistant only answers using this data - it will say so if the data isn't enough."
    )

    example_questions = [
        "What is the most common problem?",
        "Which location has the most complaints?",
        "What are the major problems in the library?",
        "Summarize the negative feedback.",
        "What problems have high severity?",
        "What improvements are commonly suggested?",
    ]
    st.caption("Try one of these: " + " | ".join(example_questions))

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask CampusSense Assistant about this dataset...")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    answer = chatbot_answer(user_question, filtered_df)
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except AIAnalysisError as exc:
                error_text = f"⚠️ {exc}"
                st.error(error_text)
                st.session_state.chat_history.append({"role": "assistant", "content": error_text})

# --- Dataset tab: raw preview + export ---
with tab_dataset:
    st.subheader("Analyzed Dataset")
    st.dataframe(filtered_df, use_container_width=True, height=450)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download analyzed dataset (campus_feedback_analyzed.csv)",
        data=csv_bytes,
        file_name="campus_feedback_analyzed.csv",
        mime="text/csv",
    )

    st.subheader("Raw (Cleaned) Dataset")
    st.caption("Data after Pandas cleaning, before AI analysis was applied.")
    st.dataframe(st.session_state.raw_df, use_container_width=True, height=300)
