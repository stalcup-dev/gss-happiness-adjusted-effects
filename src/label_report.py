from __future__ import annotations

from pathlib import Path

import pandas as pd
from .preprocess import preprocess_gss


def load_gss_csv(path: Path) -> pd.DataFrame:
    """Load a GSS extract CSV and preprocess."""
    df = pd.read_csv(path)
    df = preprocess_gss(df, validate=True)
    return df


def run_label_report(input_path: Path) -> None:
    """Print exact categories/labels for the HAPPY (happiness) variable."""
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    df = load_gss_csv(input_path)
    
    print("=" * 80)
    print("GSS HAPPINESS VARIABLE (HAPPY) - LABELS")
    print("=" * 80)
    print()
    
    if "HAPPY" not in df.columns:
        raise ValueError("'HAPPY' column not found in data")
    
    happy_col = df["HAPPY"]
    
    print(f"Column name: HAPPY")
    print(f"Dtype: {happy_col.dtype}")
    print(f"Total records: {len(df)}")
    print(f"Non-missing: {happy_col.notna().sum()}")
    print(f"Missing: {happy_col.isna().sum()}")
    print()
    
    print("VALUE COUNTS (including NaN):")
    print("-" * 80)
    vc = happy_col.value_counts(dropna=False)
    for val, count in vc.items():
        pct = 100 * count / len(df)
        if pd.isna(val):
            print(f"  NaN / Missing:              {count:6d} ({pct:5.1f}%)")
        else:
            print(f"  {str(val):30s} {count:6d} ({pct:5.1f}%)")
    print()
    
    # If numeric, show distribution
    if pd.api.types.is_numeric_dtype(happy_col):
        print("NUMERIC SUMMARY:")
        print("-" * 80)
        print(happy_col.describe())
        print()
    
    # Show first few raw values
    print("FIRST 10 RAW VALUES:")
    print("-" * 80)
    for i, val in enumerate(happy_col.head(10)):
        print(f"  Row {i}: {repr(val)}")
    print()
    
    print("=" * 80)


if __name__ == "__main__":
    from .config import get_paths
    
    paths = get_paths()
    input_path = paths.data_raw / "gss_extract.csv"
    run_label_report(input_path)
