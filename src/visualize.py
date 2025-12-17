from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_score_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    _ensure_dir(out_dir)
    out_path = out_dir / "ladder_score_distribution.png"

    plt.figure(figsize=(9, 5))
    sns.histplot(df["ladder_score"].dropna(), bins=20, kde=True)
    plt.title("Happiness (Ladder Score) Distribution")
    plt.xlabel("Ladder score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_score_by_year(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "year" not in df.columns:
        return None

    _ensure_dir(out_dir)
    out_path = out_dir / "ladder_score_by_year.png"

    yearly = df.groupby("year")["ladder_score"].mean().reset_index().sort_values("year")

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=yearly, x="year", y="ladder_score", marker="o")
    plt.title("Average Happiness Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average ladder score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_score_by_region(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if "region" not in df.columns:
        return None

    _ensure_dir(out_dir)
    out_path = out_dir / "ladder_score_by_region.png"

    region_stats = (
        df.groupby("region")["ladder_score"].mean().reset_index().sort_values("ladder_score")
    )

    plt.figure(figsize=(9, 5))
    sns.barplot(data=region_stats, x="ladder_score", y="region")
    plt.title("Average Happiness by Region")
    plt.xlabel("Average ladder score")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_correlations(df: pd.DataFrame, out_dir: Path) -> Path | None:
    numeric_cols = [
        c
        for c in [
            "ladder_score",
            "gdp_per_capita",
            "social_support",
            "healthy_life_expectancy",
            "freedom",
            "generosity",
            "perceptions_of_corruption",
        ]
        if c in df.columns
    ]
    if len(numeric_cols) < 3:
        return None

    _ensure_dir(out_dir)
    out_path = out_dir / "correlation_heatmap.png"

    corr = df[numeric_cols].corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
