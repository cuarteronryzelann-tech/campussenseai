"""
visualization.py
-----------------
Plotly chart builders for the CampusSense AI Streamlit dashboard.

Each function takes the analyzed (and already-filtered) DataFrame and
returns a Plotly Figure. Keeping chart-building code here means app.py
just calls these functions and st.plotly_chart()'s the result - all
charts automatically react to whatever filters are applied upstream.
"""

import pandas as pd
import plotly.express as px

# A consistent color palette used across charts for each sentiment value.
SENTIMENT_COLORS = {
    "Positive": "#2ecc71",
    "Neutral": "#95a5a6",
    "Negative": "#e74c3c",
    "Mixed": "#f39c12",
}

SEVERITY_COLORS = {
    "Low": "#2ecc71",
    "Medium": "#f39c12",
    "High": "#e74c3c",
}


def sentiment_distribution_chart(df: pd.DataFrame):
    """Pie chart of Positive / Neutral / Negative / Mixed feedback counts."""
    if df.empty or "sentiment" not in df.columns:
        return None
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts, names="sentiment", values="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution", hole=0.4,
    )
    fig.update_traces(textinfo="percent+label")
    return fig


def category_chart(df: pd.DataFrame):
    """Bar chart of problem categories, most frequent first."""
    if df.empty or "category" not in df.columns:
        return None
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(
        counts, x="count", y="category", orientation="h",
        title="Problem Categories", labels={"count": "Reports", "category": "Category"},
        color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig


def location_chart(df: pd.DataFrame):
    """Bar chart of reports per campus location."""
    if df.empty or "location" not in df.columns:
        return None
    counts = df["location"].value_counts().reset_index()
    counts.columns = ["location", "count"]
    fig = px.bar(
        counts, x="location", y="count",
        title="Problems by Location", labels={"count": "Reports", "location": "Location"},
        color="count", color_continuous_scale="Purples",
    )
    fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
    return fig


def severity_chart(df: pd.DataFrame):
    """Bar chart of Low / Medium / High severity counts."""
    if df.empty or "severity" not in df.columns:
        return None
    order = ["Low", "Medium", "High"]
    counts = df["severity"].value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["severity", "count"]
    fig = px.bar(
        counts, x="severity", y="count", color="severity",
        color_discrete_map=SEVERITY_COLORS, title="Severity Distribution",
        category_orders={"severity": order},
    )
    return fig


def problems_over_time_chart(df: pd.DataFrame):
    """Line chart showing number of reported problems over time."""
    if df.empty or "date" not in df.columns:
        return None
    daily_counts = (
        df.dropna(subset=["date"])
          .groupby(df["date"].dt.to_period("D"))
          .size()
          .reset_index(name="count")
    )
    daily_counts["date"] = daily_counts["date"].dt.to_timestamp()
    fig = px.line(
        daily_counts, x="date", y="count", markers=True,
        title="Problems Over Time", labels={"date": "Date", "count": "Reports"},
    )
    return fig


def top_keywords_chart(df: pd.DataFrame, top_n: int = 15):
    """Bar chart of the most frequently extracted keywords."""
    if df.empty or "keywords" not in df.columns:
        return None

    all_keywords = []
    for kw_string in df["keywords"].dropna():
        parts = [k.strip() for k in str(kw_string).split(",") if k.strip()]
        all_keywords.extend(parts)

    if not all_keywords:
        return None

    kw_series = pd.Series(all_keywords).str.lower().value_counts().head(top_n)
    kw_df = kw_series.reset_index()
    kw_df.columns = ["keyword", "count"]

    fig = px.bar(
        kw_df, x="count", y="keyword", orientation="h",
        title=f"Top {top_n} Keywords", labels={"count": "Mentions", "keyword": "Keyword"},
        color="count", color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    return fig
