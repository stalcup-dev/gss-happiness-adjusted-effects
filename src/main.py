from __future__ import annotations

import argparse
from pathlib import Path

from .config import get_paths
from .eda import summarize
from .io import choose_default_input_path, load_happiness_csv
from .visualize import (
    plot_correlations,
    plot_score_by_region,
    plot_score_by_year,
    plot_score_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run happiness EDA pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input CSV (defaults to data/raw/gss_extract.csv)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = get_paths()

    input_path = Path(args.input) if args.input else choose_default_input_path(paths.root)
    df = load_happiness_csv(input_path)

    outputs = summarize(df)

    # write tables
    paths.reports_tables.mkdir(parents=True, exist_ok=True)
    outputs.summary_overall.to_csv(paths.reports_tables / "summary_overall.csv")
    outputs.summary_by_year.to_csv(paths.reports_tables / "summary_by_year.csv", index=False)
    if outputs.summary_by_region is not None:
        outputs.summary_by_region.to_csv(paths.reports_tables / "summary_by_region.csv", index=False)

    # write figures
    plot_score_distribution(outputs.cleaned, paths.reports_figures)
    plot_score_by_year(outputs.cleaned, paths.reports_figures)
    plot_score_by_region(outputs.cleaned, paths.reports_figures)
    plot_correlations(outputs.cleaned, paths.reports_figures)

    print(f"Input: {input_path}")
    print(f"Wrote figures to: {paths.reports_figures}")
    print(f"Wrote tables to: {paths.reports_tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
