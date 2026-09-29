"""
datasets.py
-----------
Builds the 5 datasets used by CampusSense AI (Datasets tab + AI Assistant):

  1. Feedback (Full)  - every feedback record (with AI columns once analyzed)
  2. By Location      - one row per campus location
  3. By Category      - one row per problem category
  4. By Sentiment     - one row per sentiment
  5. By Severity      - one row per severity level

Datasets 3-5 need the AI analysis (they group by AI-generated columns).
Dataset 2 works even before analysis (basic counts + dates).
"""

import pandas as pd

SEVERITY_WEIGHT = {"Low": 1, "Medium": 2, "High": 3}
SENTIMENTS = ["Positive", "Neutral", "Negative", "Mixed"]
SEVERITY_ORDER = ["High", "Medium", "Low"]

DATASET_NAMES = [
    "Feedback (Full)",
    "By Location",
    "By Category",
    "By Sentiment",
    "By Severity",
]

DATASET_DESCRIPTIONS = {
    "Feedback (Full)": "Every feedback record, with the AI analysis columns once analyzed.",
    "By Location": "Reports, sentiment/severity split and top issues for each campus location.",
    "By Category": "Reports, sentiment/severity split and top issues for each problem category.",
    "By Sentiment": "Reports, severity split and top issues for each sentiment.",
    "By Severity": "Reports, sentiment split and top issues for each severity level.",
}


def is_analyzed(df: pd.DataFrame) -> bool:
    """True when the DataFrame already contains the AI analysis columns."""
    return df is not None and {"category", "sentiment", "severity"}.issubset(df.columns)


def _mode(series: pd.Series) -> str:
    series = series.dropna()
    return str(series.value_counts().idxmax()) if not series.empty else "-"


def _top_issues(series: pd.Series, n: int = 3) -> str:
    s = series.dropna().astype(str).str.strip()
    s = s[~s.str.lower().isin(["none", "analysis unavailable", ""])]
    return "; ".join(s.value_counts().head(n).index) if not s.empty else "-"


def _fmt_date(value) -> str:
    return value.strftime("%Y-%m-%d") if pd.notna(value) else "-"


def _summary_table(df: pd.DataFrame, by: str, label: str, order=None) -> pd.DataFrame:
    """Group an analyzed DataFrame by one column and summarize it."""
    work = df.copy()
    work["_sev"] = work["severity"].map(SEVERITY_WEIGHT).fillna(1)
    total = len(work)
    rows = []
    for key, g in work.groupby(by):
        row = {label: key, "Reports": len(g), "% of Total": round(len(g) / total * 100, 1)}
        if by != "sentiment":
            for s in SENTIMENTS:
                row[s] = int((g["sentiment"] == s).sum())
        if by != "severity":
            for s in SEVERITY_ORDER:
                row[f"{s} Severity"] = int((g["severity"] == s).sum())
        row["Avg Severity (1-3)"] = round(float(g["_sev"].mean()), 2)
        if by != "category":
            row["Top Category"] = _mode(g["category"])
        if by != "location":
            row["Top Location"] = _mode(g["location"])
        row["Top Issues"] = _top_issues(g["main_issue"]) if "main_issue" in g.columns else "-"
        row["First Report"] = _fmt_date(g["date"].min())
        row["Latest Report"] = _fmt_date(g["date"].max())
        rows.append(row)

    out = pd.DataFrame(rows)
    if order:
        out["_o"] = out[label].map({v: i for i, v in enumerate(order)})
        out = out.sort_values("_o").drop(columns="_o")
    else:
        out = out.sort_values("Reports", ascending=False)
    return out.reset_index(drop=True)


def _basic_location_table(df: pd.DataFrame) -> pd.DataFrame:
    """Location summary that works before AI analysis (no AI columns needed)."""
    total = len(df)
    rows = []
    for key, g in df.groupby("location"):
        rows.append({
            "Location": key,
            "Reports": len(g),
            "% of Total": round(len(g) / total * 100, 1),
            "First Report": _fmt_date(g["date"].min()),
            "Latest Report": _fmt_date(g["date"].max()),
        })
    return pd.DataFrame(rows).sort_values("Reports", ascending=False).reset_index(drop=True)


def build_datasets(df: pd.DataFrame) -> dict:
    """
    Return {dataset name: DataFrame or None}. None means the dataset is not
    available yet (needs the AI analysis to be run first).
    """
    datasets = {name: None for name in DATASET_NAMES}
    if df is None or df.empty:
        return datasets

    full = df.copy()
    if "date" in full.columns:
        full["date"] = full["date"].dt.strftime("%Y-%m-%d")
    datasets["Feedback (Full)"] = full

    if is_analyzed(df):
        datasets["By Location"] = _summary_table(df, "location", "Location")
        datasets["By Category"] = _summary_table(df, "category", "Category")
        datasets["By Sentiment"] = _summary_table(df, "sentiment", "Sentiment", order=SENTIMENTS)
        datasets["By Severity"] = _summary_table(df, "severity", "Severity", order=SEVERITY_ORDER)
    else:
        datasets["By Location"] = _basic_location_table(df)

    return datasets


def datasets_to_text(datasets: dict) -> str:
    """Compact CSV-style text of the 4 summary datasets, for the AI Assistant prompt."""
    parts = []
    for name in DATASET_NAMES[1:]:
        table = datasets.get(name)
        if table is not None and not table.empty:
            parts.append(f"### Dataset: {name}\n{table.to_csv(index=False)}")
    return "\n".join(parts)