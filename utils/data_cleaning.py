"""
data_cleaning.py
-----------------
Pandas-based data cleaning utilities for CampusSense AI.

This module is responsible for loading the raw campus feedback CSV and
turning it into a clean, validated DataFrame that is safe to pass on to
the AI analysis module. Keeping this logic separate from app.py keeps
the Streamlit code focused purely on the UI.
"""

import pandas as pd

# The columns every uploaded / loaded feedback CSV must contain.
REQUIRED_COLUMNS = ["feedback_id", "date", "location", "feedback"]


class DataValidationError(Exception):
    """Raised when the uploaded/loaded CSV does not match the expected schema."""
    pass


def load_data(file_path_or_buffer) -> pd.DataFrame:
    """
    Load a campus feedback CSV into a Pandas DataFrame.

    Accepts either a file path (str) or a file-like object, so it works
    both for the bundled sample dataset and for files uploaded through
    Streamlit's st.file_uploader.
    """
    try:
        df = pd.read_csv(file_path_or_buffer)
    except pd.errors.EmptyDataError:
        raise DataValidationError("The uploaded CSV file is empty.")
    except Exception as exc:
        raise DataValidationError(f"Could not read the CSV file: {exc}")

    if df.empty:
        raise DataValidationError("The dataset contains no rows.")

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Ensure the DataFrame has all required columns. Raises DataValidationError if not."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw campus feedback DataFrame:
      - validate required columns exist
      - drop exact duplicate rows
      - drop rows with missing/empty feedback text
      - normalize whitespace and text casing for location
      - parse the date column into real datetime objects
      - reset the index

    Returns a new, cleaned DataFrame. The original is not modified.
    """
    validate_columns(df)

    cleaned = df.copy()

    # --- Normalize whitespace in text columns ---
    text_columns = ["feedback_id", "location", "feedback"]
    for col in text_columns:
        if col in cleaned.columns:
            cleaned[col] = (
                cleaned[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

    # --- Handle missing values ---
    # Rows without any feedback text are useless for analysis, so drop them.
    cleaned["feedback"] = cleaned["feedback"].replace(
        {"nan": None, "None": None, "": None}
    )
    cleaned = cleaned.dropna(subset=["feedback"])

    # Fill missing locations with a clear placeholder rather than dropping data.
    cleaned["location"] = cleaned["location"].replace({"nan": None, "": None})
    cleaned["location"] = cleaned["location"].fillna("Unspecified")

    # --- Remove duplicate records ---
    cleaned = cleaned.drop_duplicates(subset=["feedback_id"], keep="first")
    cleaned = cleaned.drop_duplicates(subset=["location", "feedback"], keep="first")

    # --- Normalize location text (title case for consistent grouping/filtering) ---
    cleaned["location"] = cleaned["location"].str.title()

    # --- Convert dates to datetime, dropping rows with unparseable dates ---
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])

    cleaned = cleaned.reset_index(drop=True)

    if cleaned.empty:
        raise DataValidationError(
            "After cleaning, no valid feedback records remained. "
            "Please check the dataset for missing dates or feedback text."
        )

    return cleaned


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return quick summary stats about the cleaned dataset (used for sanity checks/debug)."""
    return {
        "total_records": len(df),
        "unique_locations": df["location"].nunique() if "location" in df else 0,
        "date_range": (
            (df["date"].min(), df["date"].max())
            if "date" in df and not df["date"].isna().all()
            else (None, None)
        ),
    }
