"""
check_coverage.py
-----------------
Runs the real AI analysis once on data/campus_feedback.csv, then reports:
  1. how many records each location / category / sentiment / severity ended up with
  2. every location + category combination that has fewer than MIN_ROWS records
  3. (if data/expected_labels.csv exists) how often the AI agreed with the labels the
     sample records were written to have

Usage:  python check_coverage.py
Needs HF_TOKEN in your .env (same as the app). ~570 rows = ~570 API calls, so this takes a while.
"""

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils.data_cleaning import load_data, clean_data
from utils.ai_analysis import analyze_dataframe, VALID_CATEGORIES, VALID_SENTIMENTS, VALID_SEVERITIES

MIN_ROWS = 5
CSV_PATH = "data/campus_feedback.csv"
EXPECTED_PATH = "data/expected_labels.csv"


def report(df):
    problems = 0
    for column, values in [("location", sorted(df["location"].unique())), ("category", VALID_CATEGORIES),
                           ("sentiment", VALID_SENTIMENTS), ("severity", VALID_SEVERITIES)]:
        print(f"\n== {column} ==")
        counts = df[column].value_counts()
        for v in values:
            n = int(counts.get(v, 0))
            problems += n < MIN_ROWS
            print(f"  {v:<22}{n:>4}{'' if n >= MIN_ROWS else f'   <-- fewer than {MIN_ROWS}'}")

    combos = df.groupby(["location", "category"]).size()
    thin = [(l, c, int(combos.get((l, c), 0))) for l in sorted(df["location"].unique()) for c in VALID_CATEGORIES
            if combos.get((l, c), 0) < MIN_ROWS and (l, c) in _intended(df)]
    print(f"\n== location + category combinations below {MIN_ROWS} (only ones the sample data intends) ==")
    for l, c, n in thin:
        print(f"  {l} + {c}: {n}")
    if not thin:
        print("  none")

    if "analysis_error" in df.columns and int(df["analysis_error"].notna().sum()):
        print(f"\n{int(df['analysis_error'].notna().sum())} row(s) fell back to placeholders (API errors) - re-run.")
    print("\nAll single values have enough rows." if not problems else f"\n{problems} value(s) below {MIN_ROWS}.")


def _intended(df):
    """Location + category pairs the sample data was designed to contain (from expected_labels.csv)."""
    if not os.path.exists(EXPECTED_PATH):
        return set(zip(df["location"], df["category"]))
    exp = pd.read_csv(EXPECTED_PATH)
    return set(zip(exp["location"], exp["category"]))


def agreement(df):
    if not os.path.exists(EXPECTED_PATH):
        return
    exp = pd.read_csv(EXPECTED_PATH).rename(columns={c: f"exp_{c}" for c in ("category", "sentiment", "severity")})
    m = df.merge(exp[["feedback_id", "exp_category", "exp_sentiment", "exp_severity"]], on="feedback_id")
    print("\n== AI vs. intended labels ==")
    for c in ("category", "sentiment", "severity"):
        print(f"  {c:<10}{(m[c] == m['exp_' + c]).mean() * 100:5.1f}% agree")


if __name__ == "__main__":
    data = clean_data(load_data(CSV_PATH))
    print(f"Analyzing {len(data)} rows ...")
    result = analyze_dataframe(data, progress_callback=lambda i, n: print(f"  {i}/{n}", end="\r"))
    report(result)
    agreement(result)
