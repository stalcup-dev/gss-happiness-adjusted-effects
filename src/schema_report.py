from __future__ import annotations

from pathlib import Path

import pandas as pd
from .preprocess import preprocess_gss, validate_gss_extract


def load_gss_csv(path: Path) -> pd.DataFrame:
    """Load a GSS extract CSV and preprocess."""
    df = pd.read_csv(path)
    df = preprocess_gss(df, validate=True)
    return df


def run_schema_report(input_path: Path) -> None:
    """Print column structure, dtypes, missing counts, and unique values."""
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = load_gss_csv(input_path)
    
    print("=" * 80)
    print("GSS SCHEMA REPORT")
    print("=" * 80)
    print()
    
    # Validation summary
    is_valid, val_msg = validate_gss_extract(df)
    print(f"VALIDATION: {val_msg}")
    print()
    
    # Column list and dtypes
    print("COLUMN LIST & DTYPES")
    print("-" * 80)
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = 100 * missing_count / len(df)
        print(f"{col:30s} {str(df[col].dtype):15s} Missing: {missing_count:6d} ({missing_pct:5.1f}%)")
    print()
    
    # Happiness column (HAPPY)
    if "HAPPY" in df.columns:
        print("HAPPINESS VARIABLE (HAPPY)")
        print("-" * 80)
        print(f"Dtype: {df['HAPPY'].dtype}")
        print(f"Missing: {df['HAPPY'].isna().sum()}")
        print(f"Unique values: {df['HAPPY'].nunique()}")
        print(f"Value counts:")
        print(df['HAPPY'].value_counts(dropna=False))
        print()
    else:
        print("WARNING: 'HAPPY' column not found in data")
        print()
    
    # Year column
    if "YEAR" in df.columns:
        print("YEAR VARIABLE")
        print("-" * 80)
        print(f"Dtype: {df['YEAR'].dtype}")
        print(f"Range: {df['YEAR'].min()} to {df['YEAR'].max()}")
        print(f"Unique years: {sorted(df['YEAR'].dropna().unique())}")
        print()
    else:
        print("WARNING: 'YEAR' column not found in data")
        print()
    
    # Candidate predictors (common GSS variables)
    candidate_predictors = [
        "INCOME", "EDUC", "AGE", "SEX", "MARITAL", "REGION", 
        "RELIG", "DEGREE", "OCCUP", "WRKSTAT", "CLASS",
        "HEALTH", "KIDS", "PARTYID", "POLVIEWS"
    ]
    present_predictors = [col for col in candidate_predictors if col in df.columns]
    
    if present_predictors:
        print("CANDIDATE PREDICTORS (PRESENT)")
        print("-" * 80)
        for col in present_predictors:
            missing_count = df[col].isna().sum()
            unique_count = df[col].nunique()
            print(f"{col:30s} dtype={str(df[col].dtype):10s} unique={unique_count:6d} missing={missing_count:6d}")
        print()
    
    # Weight columns
    weight_cols = [col for col in df.columns if "WT" in col or "WEIGHT" in col]
    if weight_cols:
        print("WEIGHT COLUMNS")
        print("-" * 80)
        for col in weight_cols:
            print(f"{col:30s} dtype={str(df[col].dtype):10s} min={df[col].min():.4f} max={df[col].max():.4f}")
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    from .config import get_paths
    
    paths = get_paths()
    input_path = paths.data_raw / "gss_extract.csv"
    run_schema_report(input_path)
