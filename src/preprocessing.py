"""
preprocessing.py
================
Retail Sales Prediction - Data Preprocessing & Feature Engineering Pipeline

This module handles all data ingestion, cleaning, feature engineering,
encoding, scaling, and export operations required before model training.

Author: Sales Neural Network Project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

RAW_DATA_PATH = Path("data/raw/sales.csv")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SCALER_PATH = MODELS_DIR / "scaler.joblib"
POWERBI_EXPORT_PATH = PROCESSED_DIR / "powerbi_clean_sales.csv"

CATEGORICAL_LOW_CARD = ["Region", "Segment", "Category"]
NUMERICAL_FEATURES = [
    "Quantity", "Discount", "Profit",
    "discount_percentage", "promo_active",
    "month", "year", "day_of_week", "is_weekend",
    "month_sin", "month_cos", "day_sin", "day_cos",
]
TARGET_COLUMN = "Sales"
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA LOADING & BASIC CLEANING
# ──────────────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV dataset from disk.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Raw DataFrame with all original columns.
    """
    print(f"[1/6] Loading raw data from: {filepath}")
    df = pd.read_csv(filepath, encoding="latin-1")
    print(f"      Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning:
    - Remove duplicates
    - Drop rows with null targets
    - Parse date columns
    - Strip whitespace from strings

    Args:
        df: Raw input DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    print("[2/6] Cleaning data...")
    original_len = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()
    print(f"      Removed {original_len - len(df)} duplicate rows.")

    # Drop rows where target is missing
    df = df.dropna(subset=[TARGET_COLUMN])

    # Parse date columns
    for col in ["Order Date", "Ship Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    # Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Fill remaining numeric nulls with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    print(f"      Clean dataset: {len(df):,} rows.")
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def engineer_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from 'Order Date', including cyclical
    sine/cosine transforms for seasonality awareness.

    Args:
        df: DataFrame with parsed 'Order Date' column.

    Returns:
        DataFrame enriched with temporal features.
    """
    df = df.copy()
    date_col = df["Order Date"]

    df["month"] = date_col.dt.month
    df["year"] = date_col.dt.year
    df["day_of_week"] = date_col.dt.dayofweek          # 0=Monday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding – helps NN understand circular nature of time
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def engineer_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create business-logic derived features:
    - discount_percentage: Discount as a percentage (0–100)
    - promo_active: Binary flag if any discount is applied

    Args:
        df: Cleaned DataFrame with 'Discount' column.

    Returns:
        DataFrame with additional business features.
    """
    df = df.copy()
    df["discount_percentage"] = df["Discount"] * 100.0
    df["promo_active"] = (df["Discount"] > 0).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame,
                         columns: List[str] = CATEGORICAL_LOW_CARD) -> pd.DataFrame:
    """
    Apply One-Hot Encoding to low-cardinality categorical columns.
    Uses drop_first=True to avoid the dummy variable trap.

    Args:
        df: DataFrame with categorical columns.
        columns: List of column names to encode.

    Returns:
        DataFrame with original categoricals replaced by OHE dummies.
    """
    cols_to_encode = [c for c in columns if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=float)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3: TRAIN / TEST SPLIT & SCALING
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Assemble the final feature matrix X and target vector y,
    dropping non-predictive identifier and date columns.

    Args:
        df: Fully engineered DataFrame.

    Returns:
        (X, y, feature_names) where X is the feature DataFrame,
        y is the Sales target Series, and feature_names is the list of columns.
    """
    drop_cols = [
        "Row ID", "Order ID", "Ship Date", "Order Date",
        "Customer ID", "Customer Name", "Product ID",
        "Product Name", "City", "State", "Postal Code", "Country",
        "Sub-Category", "Ship Mode",
        TARGET_COLUMN,
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[TARGET_COLUMN]

    # Ensure all columns are numeric
    X = X.select_dtypes(include=[np.number])

    return X, y, list(X.columns)


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split data into train/test sets and fit StandardScaler ONLY on train
    to prevent data leakage. Saves scaler artifact to disk.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to hold out for testing.
        random_state: Reproducibility seed.

    Returns:
        (X_train, X_test, y_train, y_test, fitted_scaler)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # Fit ONLY on train
    X_test_scaled  = scaler.transform(X_test)         # Transform test using train stats

    # Persist scaler for inference-time use
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Scaler saved → {SCALER_PATH}")

    return (
        X_train_scaled, X_test_scaled,
        y_train.values, y_test.values,
        scaler,
    )


# ──────────────────────────────────────────────────────────────────────────────
# STEP 4: POWER BI EXPORT
# ──────────────────────────────────────────────────────────────────────────────

def export_for_powerbi(df_clean: pd.DataFrame) -> None:
    """
    Export a denormalized, analysis-ready CSV for Power BI dashboarding.
    Adds derived columns useful for DAX measures and slicers.

    Args:
        df_clean: Cleaned (but not OHE-encoded) DataFrame from clean_data().
    """
    print("[5/6] Exporting Power BI dataset...")
    pbi = df_clean.copy()

    # Ensure date columns exist
    if "Order Date" in pbi.columns:
        pbi["Order Year"]       = pbi["Order Date"].dt.year
        pbi["Order Month"]      = pbi["Order Date"].dt.month
        pbi["Order Month Name"] = pbi["Order Date"].dt.strftime("%b")
        pbi["Order Quarter"]    = pbi["Order Date"].dt.quarter
        pbi["Order Week"]       = pbi["Order Date"].dt.isocalendar().week.astype(int)

    # Business flags
    pbi["Has Discount"]    = (pbi["Discount"] > 0).map({True: "Yes", False: "No"})
    pbi["Profit Margin %"] = np.where(
        pbi["Sales"] != 0, (pbi["Profit"] / pbi["Sales"]) * 100, 0
    ).round(2)
    pbi["Revenue Band"] = pd.cut(
        pbi["Sales"],
        bins=[0, 100, 500, 1000, 5000, np.inf],
        labels=["<$100", "$100–$500", "$500–$1K", "$1K–$5K", "$5K+"],
    )

    # Convert dates to string for PBI compatibility
    for col in ["Order Date", "Ship Date"]:
        if col in pbi.columns:
            pbi[col] = pbi[col].dt.strftime("%Y-%m-%d")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pbi.to_csv(POWERBI_EXPORT_PATH, index=False)
    print(f"      Power BI CSV saved → {POWERBI_EXPORT_PATH}  ({len(pbi):,} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    filepath: Path = RAW_DATA_PATH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Execute the full preprocessing pipeline end-to-end.

    Steps:
        1. Load raw CSV
        2. Clean data
        3. Export Power BI CSV (before encoding, for readability)
        4. Engineer datetime + business features
        5. One-Hot Encode categoricals
        6. Build feature matrix
        7. Split & scale

    Args:
        filepath: Path to raw CSV data.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("=" * 60)
    print("  RETAIL SALES — PREPROCESSING PIPELINE")
    print("=" * 60)

    # 1. Load
    df_raw = load_raw_data(filepath)

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Export Power BI CSV (pre-encoding, human-readable)
    export_for_powerbi(df_clean)

    # 4. Feature Engineering
    print("[3/6] Engineering features...")
    df_feat = engineer_datetime_features(df_clean)
    df_feat = engineer_business_features(df_feat)

    # 5. Encode categoricals
    print("[4/6] Encoding categorical features...")
    df_encoded = encode_categoricals(df_feat)

    # 6. Build feature matrix
    X, y, feature_names = build_feature_matrix(df_encoded)
    print(f"      Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")

    # 7. Split & scale
    print("[6/6] Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print()
    print("  ✓  Preprocessing complete.")
    print(f"     Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print("=" * 60)

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    run_preprocessing_pipeline()
