from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import choose_default_gss_path
from .preprocess import preprocess_gss


def load_happiness_csv(path: Path) -> pd.DataFrame:
    """
    Load and preprocess GSS extract CSV.
    
    Args:
        path: Path to CSV file
    
    Returns:
        Preprocessed DataFrame with normalized columns, recoded NA, and validated
    """
    df = pd.read_csv(path)
    df = preprocess_gss(df, validate=True)
    return df


def choose_default_input_path(repo_root: Path) -> Path:
    """Default input path for CLI utilities (Phase 1 EDA pipeline)."""
    return choose_default_gss_path(repo_root)
