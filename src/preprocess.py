from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_missing_codes_config(config_path: Path | None = None) -> dict:
    """Load variable-specific missing code configuration."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "missing_codes.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing codes config not found: {config_path}")
    
    with open(config_path) as f:
        return json.load(f)


def recode_gss_na(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """
    Recode GSS non-substantive codes to pandas NA using variable-specific mapping.
    
    Config structure:
    {
      "string_patterns": {
        "_all": ["IAP", "DK", "NA", "REFUSED"],
        "SOMECOLUMN": ["IAP", "DK"]
      },
      "numeric_codes": {
        "EDUC": [98, 99],
        "AGE": [98, 99]
      }
    }
    
    Args:
        df: Input DataFrame
        config: Missing codes config dict. If None, loads from missing_codes.json
    
    Returns:
        DataFrame with non-substantive codes replaced by NA
    """
    if config is None:
        config = load_missing_codes_config()
    
    df = df.copy()
    string_patterns = config.get("string_patterns", {})
    numeric_codes = config.get("numeric_codes", {})
    global_string_patterns = string_patterns.get("_all", [])
    
    for col in df.columns:
        # String column: apply patterns
        if df[col].dtype == "object":
            col_patterns = string_patterns.get(col, global_string_patterns)
            for pattern in col_patterns:
                mask = df[col].astype(str).str.upper() == pattern.upper()
                df.loc[mask, col] = pd.NA
        
        # Numeric column: recode listed codes only
        if pd.api.types.is_numeric_dtype(df[col]):
            codes_to_recode = numeric_codes.get(col, [])
            for code in codes_to_recode:
                df.loc[df[col] == code, col] = pd.NA
    
    return df


def validate_gss_extract(
    df: pd.DataFrame,
    min_rows: int = 5000,
    min_year_span: int = 5,
) -> tuple[bool, str]:
    """
    Validate GSS extract for minimum size, time coverage, and required columns.
    
    Args:
        df: DataFrame with GSS data
        min_rows: Minimum number of rows required (default 5000)
        min_year_span: Minimum year span required (default 5 years)
    
    Returns:
        (is_valid, message)
    """
    
    # Check required columns exist
    if "YEAR" not in df.columns:
        return False, "Missing required column: YEAR"
    if "HAPPY" not in df.columns:
        return False, "Missing required column: HAPPY"
    if "WTSSPS" not in df.columns:
        return False, "Missing required column: WTSSPS (weight column)"
    
    # Validate YEAR is integer
    try:
        year_vals = df["YEAR"].dropna()
        if len(year_vals) > 0:
            non_int = year_vals[year_vals != year_vals.astype(int)]
            if len(non_int) > 0:
                return False, f"YEAR column contains non-integer values: {non_int.unique()}"
    except (ValueError, TypeError):
        return False, "YEAR column cannot be cast to integer"
    
    # Validate WTSSPS is numeric and > 0
    if not pd.api.types.is_numeric_dtype(df["WTSSPS"]):
        return False, "WTSSPS must be numeric"
    wt_positive = df["WTSSPS"][df["WTSSPS"] > 0]
    if len(wt_positive) == 0:
        return False, "WTSSPS must have positive values"
    
    # Check row count after cleaning
    clean_df = df.dropna(subset=["YEAR", "HAPPY"])
    n_clean = len(clean_df)
    
    if n_clean < min_rows:
        return False, (
            f"Insufficient cases after removing missing YEAR/HAPPY: {n_clean} rows. "
            f"Minimum required: {min_rows}. "
            f"Expand extract in GSS Data Explorer."
        )
    
    # Check year span
    years = clean_df["YEAR"].dropna().unique()
    if len(years) == 0:
        return False, "No valid years found"
    
    year_span = int(years.max()) - int(years.min())
    if year_span < min_year_span:
        return False, (
            f"Year span too small: {year_span} years (min={min_year_span}). "
            f"Range: {int(years.min())} to {int(years.max())}. "
            f"Expand year range in GSS Data Explorer."
        )
    
    return True, f"Valid: {n_clean} cases, {len(years)} years ({int(years.min())} to {int(years.max())})"


def preprocess_gss(
    df: pd.DataFrame,
    validate: bool = True,
    cast_year_int: bool = True,
    config: dict | None = None,
    min_rows: int = 5000,
    min_year_span: int = 5,
) -> pd.DataFrame:
    """
    Full GSS preprocessing pipeline.
    
    1. Normalize column names to uppercase
    2. Cast YEAR to int (if cast_year_int=True)
    3. Recode non-substantive codes to NA
    4. Validate extract size and time coverage
    
    Args:
        df: Raw GSS extract
        validate: If True, raise on validation failure
        cast_year_int: If True, cast YEAR to int
        config: Missing codes config. If None, loads from missing_codes.json
        min_rows: Minimum rows threshold
        min_year_span: Minimum year span threshold
    
    Returns:
        Cleaned DataFrame
    
    Raises:
        ValueError: If validation fails and validate=True
    """
    
    # Normalize column names
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Cast YEAR to int
    if cast_year_int and "YEAR" in df.columns:
        try:
            df["YEAR"] = df["YEAR"].astype("Int64")  # nullable int
        except (ValueError, TypeError) as e:
            if validate:
                raise ValueError(f"Cannot cast YEAR to integer: {e}")
    
    # Recode non-substantive codes
    df = recode_gss_na(df, config=config)
    
    # Validate
    is_valid, message = validate_gss_extract(df, min_rows, min_year_span)
    if not is_valid and validate:
        raise ValueError(f"GSS extract validation failed: {message}")
    
    return df
