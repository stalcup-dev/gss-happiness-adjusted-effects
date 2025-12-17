"""
Generate Phase 2 figures:
- Weighted HAPPY-by-year trend
- Weighted crosstabs (HAPPY vs HEALTH, MARITAL, INCOME)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import get_paths
from .io import load_happiness_csv


def weighted_trend(df: pd.DataFrame, group_col: str = "YEAR", value_col: str = "HAPPY", weight_col: str = "WTSSPS") -> pd.DataFrame:
    """
    Compute weighted trend for value distribution across groups.
    Returns the proportion of each category by group, weighted.
    
    Args:
        df: Preprocessed DataFrame
        group_col: Column to group by (e.g., YEAR)
        value_col: Value column (e.g., HAPPY)
        weight_col: Weight column (e.g., WTSSPS)
    
    Returns:
        DataFrame with groups x categories, weighted proportions
    """
    
    result = []
    
    for group_val in sorted(df[group_col].dropna().unique()):
        subset = df[df[group_col] == group_val].copy()
        
        # Weighted proportions by category
        for cat in subset[value_col].dropna().unique():
            mask = subset[value_col] == cat
            weighted_n = subset.loc[mask, weight_col].sum()
            total_weight = subset[weight_col].sum()
            prop = weighted_n / total_weight if total_weight > 0 else 0
            
            result.append({
                group_col: group_val,
                value_col: cat,
                "weighted_prop": prop,
                "pct": 100 * prop
            })
    
    return pd.DataFrame(result)


def plot_weighted_happy_trend(df: pd.DataFrame, out_dir: Path) -> Path:
    """
    Plot weighted HAPPY trend over years.
    Shows proportion of each category by year.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "weighted_happy_trend.png"
    
    trend_df = weighted_trend(df, group_col="YEAR", value_col="HAPPY", weight_col="WTSSPS")
    
    plt.figure(figsize=(11, 6))
    
    for cat in trend_df["HAPPY"].unique():
        cat_df = trend_df[trend_df["HAPPY"] == cat].sort_values("YEAR")
        plt.plot(cat_df["YEAR"], cat_df["pct"], marker="o", label=cat, linewidth=2)
    
    plt.title("Weighted Happiness Trend Over Time (WTSSPS)")
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path


def weighted_crosstab(df: pd.DataFrame, row_var: str, col_var: str = "HAPPY", weight_col: str = "WTSSPS") -> pd.DataFrame:
    """
    Compute weighted crosstab.
    
    Returns proportions across row categories, weighted by weight_col.
    """
    
    result = []
    
    for row_val in df[row_var].dropna().unique():
        row_subset = df[df[row_var] == row_val].copy()
        row_total_weight = row_subset[weight_col].sum()
        
        if row_total_weight == 0:
            continue
        
        row_dict = {row_var: row_val}
        
        for col_val in df[col_var].dropna().unique():
            mask = row_subset[col_var] == col_val
            col_weight = row_subset.loc[mask, weight_col].sum()
            col_prop = col_weight / row_total_weight if row_total_weight > 0 else 0
            row_dict[f"{col_val}"] = col_prop
        
        result.append(row_dict)
    
    return pd.DataFrame(result)


def plot_weighted_crosstab(df: pd.DataFrame, row_var: str, out_dir: Path) -> Path:
    """
    Plot weighted crosstab as stacked bar chart.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"crosstab_{row_var.lower()}_x_happy.png"
    
    crosstab_df = weighted_crosstab(df, row_var=row_var, col_var="HAPPY", weight_col="WTSSPS")
    
    # Sort row_var by one category (e.g., VERY HAPPY) descending for better viz
    if "VERY HAPPY" in crosstab_df.columns:
        crosstab_df = crosstab_df.sort_values("VERY HAPPY", ascending=False)
    
    # Prepare for stacked bar
    plot_df = crosstab_df.set_index(row_var)
    plot_df = plot_df[[c for c in ["VERY HAPPY", "PRETTY HAPPY", "NOT TOO HAPPY"] if c in plot_df.columns]]
    
    ax = plot_df.plot(kind="barh", stacked=True, figsize=(10, 6), width=0.7)
    plt.title(f"Happiness Distribution by {row_var} (Weighted, WTSSPS)")
    plt.xlabel("Proportion")
    plt.ylabel(row_var)
    plt.legend(title="HAPPY", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path


def save_weighted_crosstab_csv(df: pd.DataFrame, row_var: str, out_dir: Path) -> Path:
    """Save weighted crosstab to CSV with percentages."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"crosstab_{row_var.lower()}_x_happy.csv"
    
    crosstab_df = weighted_crosstab(df, row_var=row_var, col_var="HAPPY", weight_col="WTSSPS")
    
    # Add percentages
    for col in ["VERY HAPPY", "PRETTY HAPPY", "NOT TOO HAPPY"]:
        if col in crosstab_df.columns:
            crosstab_df[f"{col}_pct"] = 100 * crosstab_df[col]
    
    crosstab_df.to_csv(out_path, index=False)
    return out_path


def main(input_path: Path | None = None, fig_dir: Path | None = None, table_dir: Path | None = None) -> None:
    """
    Generate Phase 2 figures and tables.
    
    Args:
        input_path: Path to GSS extract CSV
        fig_dir: Output directory for figures
        table_dir: Output directory for tables
    """
    from .io import choose_default_gss_path
    
    paths = get_paths()
    
    if input_path is None:
        input_path = choose_default_gss_path(paths.root)
    
    if fig_dir is None:
        fig_dir = paths.reports_figures
    
    if table_dir is None:
        table_dir = paths.reports_tables
    
    # Load and preprocess
    df = load_happiness_csv(input_path)
    
    print("Generating Phase 2 figures...")
    
    # Weighted HAPPY trend
    plot_path = plot_weighted_happy_trend(df, fig_dir)
    print(f"  ✓ Trend plot: {plot_path}")
    
    # Weighted crosstabs for key predictors
    # Add within-year INCOME_QUARTILE for a defensible, distribution-aware descriptive view
    if 'INCOME' in df.columns and 'YEAR' in df.columns:
        def rank_based_quantile_bin(series):
            if len(series) < 4:
                return pd.cut(series, bins=len(series.unique()), labels=False, duplicates='drop')
            ranks = series.rank(method='average')
            return pd.qcut(ranks, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

        df = df.copy()
        df['INCOME_QUARTILE'] = df.groupby('YEAR')['INCOME'].transform(rank_based_quantile_bin)

    for predictor in ["HEALTH", "MARITAL", "INCOME", "INCOME_QUARTILE"]:
        if predictor in df.columns:
            # Plot
            plot_path = plot_weighted_crosstab(df, predictor, fig_dir)
            print(f"  ✓ Crosstab plot ({predictor}): {plot_path}")
            
            # CSV
            csv_path = save_weighted_crosstab_csv(df, predictor, table_dir)
            print(f"  ✓ Crosstab table ({predictor}): {csv_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
