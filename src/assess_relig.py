"""
Missingness mechanism analysis for RELIG (religion) variable.
Assesses whether RELIG is MCAR (Missing Completely At Random) or MAR.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def assess_relig_missingness(df: pd.DataFrame) -> dict:
    """
    Analyze RELIG missingness mechanism.
    
    Checks if RELIG missing is:
    - MCAR: independent of observed variables
    - MAR: depends on other variables (e.g., YEAR, AGE)
    - MNAR: depends on RELIG itself (unobservable)
    
    Args:
        df: Preprocessed DataFrame with RELIG column
    
    Returns:
        Dict with findings
    """
    
    # Create RELIG missingness indicator
    df_copy = df.copy()
    df_copy['RELIG_MISSING'] = df_copy['RELIG'].isna().astype(int)
    
    findings = {
        'n_missing': df_copy['RELIG_MISSING'].sum(),
        'pct_missing': 100 * df_copy['RELIG_MISSING'].mean(),
        'by_year': {},
        'by_age_group': {},
        'by_health': {},
        'conclusion': ''
    }
    
    # Missingness by YEAR
    if 'YEAR' in df_copy.columns:
        by_year = df_copy.groupby('YEAR')['RELIG_MISSING'].agg(['sum', 'count', 'mean'])
        by_year['pct'] = 100 * by_year['mean']
        findings['by_year'] = by_year[['sum', 'count', 'pct']].to_dict('index')
    
    # Missingness by age group
    if 'AGE' in df_copy.columns:
        df_copy['AGE_GROUP'] = pd.cut(df_copy['AGE'], bins=[0, 30, 50, 70, 100], 
                                       labels=['18-30', '31-50', '51-70', '70+'])
        by_age = df_copy.groupby('AGE_GROUP', dropna=False)['RELIG_MISSING'].agg(['sum', 'count', 'mean'])
        by_age['pct'] = 100 * by_age['mean']
        findings['by_age_group'] = by_age[['sum', 'count', 'pct']].to_dict('index')
    
    # Missingness by HEALTH
    if 'HEALTH' in df_copy.columns:
        by_health = df_copy.groupby('HEALTH')['RELIG_MISSING'].agg(['sum', 'count', 'mean'])
        by_health['pct'] = 100 * by_health['mean']
        findings['by_health'] = by_health[['sum', 'count', 'pct']].to_dict('index')
    
    # Assess mechanism
    year_var = [v['pct'] for v in findings['by_year'].values()] if findings['by_year'] else []
    age_var = [v['pct'] for v in findings['by_age_group'].values()] if findings['by_age_group'] else []
    health_var = [v['pct'] for v in findings['by_health'].values()] if findings['by_health'] else []
    
    max_year_diff = max(year_var) - min(year_var) if year_var else 0
    max_age_diff = max(age_var) - min(age_var) if age_var else 0
    max_health_diff = max(health_var) - min(health_var) if health_var else 0
    
    findings['max_year_variation'] = max_year_diff
    findings['max_age_variation'] = max_age_diff
    findings['max_health_variation'] = max_health_diff
    
    # Conclusion
    if max_year_diff < 5 and max_age_diff < 5 and max_health_diff < 5:
        findings['conclusion'] = (
            "[OK] RELIG appears MCAR (Missing Completely At Random). "
            "Missingness does not vary significantly by YEAR, AGE, or HEALTH. "
            "Safe to include in modeling."
        )
        findings['recommendation'] = 'INCLUDE'
    elif max_year_diff < 10 or max_age_diff < 10:
        findings['conclusion'] = (
            "[WARNING] RELIG appears MAR (depends on observed variables). "
            "Missingness varies slightly by YEAR/AGE/HEALTH. "
            "Listwise deletion acceptable with note; consider multiple imputation for sensitivity."
        )
        findings['recommendation'] = 'INCLUDE_WITH_CAUTION'
    else:
        findings['conclusion'] = (
            "[ERROR] RELIG missingness depends strongly on observed variables (MAR/MNAR). "
            "Cannot safely use without imputation. Defer to Phase 3b (MI strategy)."
        )
        findings['recommendation'] = 'EXCLUDE'
    
    return findings


def print_relig_assessment(findings: dict) -> None:
    """Print missingness assessment in readable format."""
    print("\n" + "=" * 80)
    print("RELIG (RELIGION) MISSINGNESS MECHANISM ASSESSMENT")
    print("=" * 80)
    print()
    
    print(f"Missing cases: {findings['n_missing']:.0f} ({findings['pct_missing']:.1f}%)")
    print()
    
    print("VARIATION BY YEAR:")
    print(f"  Range: {findings['max_year_variation']:.1f} percentage points")
    if findings['by_year']:
        for year, vals in sorted(findings['by_year'].items())[:3]:
            print(f"    {int(year)}: {vals['pct']:.1f}%")
        print(f"    ... ({len(findings['by_year'])} years total)")
    print()
    
    print("VARIATION BY AGE GROUP:")
    print(f"  Range: {findings['max_age_variation']:.1f} percentage points")
    if findings['by_age_group']:
        for age, vals in findings['by_age_group'].items():
            print(f"    {age}: {vals['pct']:.1f}%")
    print()
    
    print("VARIATION BY HEALTH:")
    print(f"  Range: {findings['max_health_variation']:.1f} percentage points")
    if findings['by_health']:
        for health, vals in findings['by_health'].items():
            print(f"    {health}: {vals['pct']:.1f}%")
    print()
    
    print("CONCLUSION:")
    print(f"  {findings['conclusion']}")
    print()
    print(f"RECOMMENDATION: {findings['recommendation']}")
    print("=" * 80)


if __name__ == "__main__":
    from .io import load_happiness_csv
    from .config import get_paths
    
    paths = get_paths()
    input_path = paths.data_raw / "gss_extract.csv"
    
    df = load_happiness_csv(input_path)
    findings = assess_relig_missingness(df)
    print_relig_assessment(findings)
