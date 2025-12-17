from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EdaOutputs:
    cleaned: pd.DataFrame
    summary_overall: pd.DataFrame
    summary_by_year: pd.DataFrame
    summary_by_region: pd.DataFrame | None


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop rows missing required fields
    df = df.dropna(subset=["country", "year", "ladder_score"])

    # keep reasonable score range if present (0..10 typical)
    df = df[(df["ladder_score"] >= 0) & (df["ladder_score"] <= 10)]

    return df


def summarize(df: pd.DataFrame) -> EdaOutputs:
    df = basic_clean(df)

    summary_overall = df[["ladder_score"]].describe().T
    summary_overall.index.name = "metric"

    summary_by_year = (
        df.groupby("year", dropna=True)["ladder_score"]
        .agg(n_countries="count", mean="mean", median="median", std="std")
        .reset_index()
        .sort_values("year")
    )

    summary_by_region = None
    if "region" in df.columns:
        summary_by_region = (
            df.groupby("region", dropna=True)["ladder_score"]
            .agg(n_countries="count", mean="mean", median="median")
            .reset_index()
            .sort_values("mean", ascending=False)
        )

    return EdaOutputs(
        cleaned=df,
        summary_overall=summary_overall,
        summary_by_year=summary_by_year,
        summary_by_region=summary_by_region,
    )
