"""
Missingness by year analysis.
Reports missing data patterns for predictors by year.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import get_paths
from .preprocess import preprocess_gss
from .io import load_happiness_csv


def missingness_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute missingness rate by year for all columns.
    
    Args:
        df: Preprocessed GSS DataFrame
    
    Returns:
        DataFrame with year x column missingness matrix
    """
    
    # Pivot: year x (column -> missing rate %)
    missingness = []
    
    for year in sorted(df["YEAR"].dropna().unique()):
        year_df = df[df["YEAR"] == year]
        n = len(year_df)
        
        row = {"YEAR": int(year), "N": n}
        for col in df.columns:
            if col not in ["YEAR"]:
                missing_count = year_df[col].isna().sum()
                missing_pct = 100 * missing_count / n if n > 0 else 0
                row[col] = missing_pct
        
        missingness.append(row)
    
    return pd.DataFrame(missingness)


def save_missingness_report(df: pd.DataFrame, out_dir: Path) -> Path:
    """Save missingness by year to CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "missingness_by_year.csv"
    df.to_csv(out_path, index=False)
    return out_path


def plot_missingness_heatmap(missingness_df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot missingness as heatmap (year x column).
    Highlights years/variables with high missing rates.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "missingness_heatmap.png"
    
    # Prepare data (exclude N column, set year as index)
    plot_df = missingness_df.set_index("YEAR").drop("N", axis=1)
    
    # Create heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        plot_df.T,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Missing %"},
        vmin=0,
        vmax=100
    )
    plt.title("Missingness by Year and Variable (%)")
    plt.xlabel("Year")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path


def main(input_path: Path | None = None, out_dir: Path | None = None) -> None:
    """
    Run missingness by year analysis.
    
    Args:
        input_path: Path to GSS extract CSV
        out_dir: Output directory for reports
    """
    from .io import choose_default_gss_path
    
    paths = get_paths()
    
    if input_path is None:
        input_path = choose_default_gss_path(paths.root)
    
    if out_dir is None:
        out_dir = paths.reports_tables
    
    # Load and preprocess
    df = load_happiness_csv(input_path)
    
    # Compute missingness
    miss_df = missingness_by_year(df)
    
    # Save CSV
    csv_path = save_missingness_report(miss_df, out_dir)
    print(f"Missingness report saved to: {csv_path}")
    
    # Plot heatmap
    fig_dir = paths.reports_figures
    plot_path = plot_missingness_heatmap(miss_df, fig_dir)
    print(f"Missingness heatmap saved to: {plot_path}")
    
    # Print summary
    print("\nMissingness by Year Summary:")
    print(miss_df.to_string(index=False))
    
    # Flag high missingness
    print("\nVariables with high missingness (>30% in any year):")
    for col in miss_df.columns:
        if col not in ["YEAR", "N"]:
            max_miss = miss_df[col].max()
            if max_miss > 30:
                max_year = miss_df.loc[miss_df[col].idxmax(), "YEAR"]
                print(f"  {col:20s}: {max_miss:5.1f}% (year {int(max_year)})")


if __name__ == "__main__":
    main()
