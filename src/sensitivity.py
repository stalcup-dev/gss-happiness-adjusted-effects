"""
Sensitivity analysis: compare models with/without 2021-2022 data.

Checks if recent mode change in GSS affects predicted probability patterns.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .modeling import HappinessModel


def sensitivity_analysis_by_year(df: pd.DataFrame) -> dict:
    """
    Fit models on two samples: pre-2021 and full sample.
    
    Compare predicted probability patterns to assess 2021-2022 mode change impact.
    
    Args:
        df: Full dataset with YEAR column
        
    Returns:
        Dict with results from both samples
    """
    
    # Split data
    df_pre_2021 = df[df['YEAR'] < 2021].reset_index(drop=True)
    df_full = df.reset_index(drop=True)
    
    print(f"Pre-2021 sample: n={len(df_pre_2021)}")
    print(f"Full sample: n={len(df_full)}")
    print(f"2021-2022 records: n={len(df_full) - len(df_pre_2021)}")
    print()
    
    results = {}
    
    for label, data in [("Pre-2021", df_pre_2021), ("Full (incl. 2021-2022)", df_full)]:
        print(f"Fitting models on {label} sample...")
        
        model = HappinessModel(data)
        
        # Baseline
        baseline = model.build_baseline(use_ordinal=True)
        if 'error' in baseline:
            baseline = model.build_baseline(use_ordinal=False)
        
        # Core
        core = model.build_core(use_ordinal=True)
        if 'error' in core:
            core = model.build_core(use_ordinal=False)
        
        results[label] = {
            'n': len(data),
            'baseline_aic': baseline.get('aic', np.nan),
            'core_aic': core.get('aic', np.nan),
            'delta_aic': baseline.get('aic', np.nan) - core.get('aic', np.nan) if 'aic' in baseline and 'aic' in core else np.nan,
            'model': model
        }
        
        print(f"  Baseline AIC: {results[label]['baseline_aic']:.1f}")
        print(f"  Core AIC: {results[label]['core_aic']:.1f}")
        print(f"  ΔAIC (baseline - core): {results[label]['delta_aic']:.1f}")
        print()
    
    return results


def compare_predicted_probs(results: dict) -> pd.DataFrame:
    """
    Compare predicted probabilities across years for key predictors.
    
    Args:
        results: Output from sensitivity_analysis_by_year
        
    Returns:
        DataFrame with comparison of predicted probs by predictor value
    """
    
    comparisons = []
    
    # Compare happy distribution at key covariate values
    # For example: Health (Excellent), Marital (Married), Income (median)
    
    for label in ["Pre-2021", "Full (incl. 2021-2022)"]:
        model = results[label]['model']
        
        # Extract summary from core model
        core_summary = model.summaries.get('core_ordinal', {})
        
        comparisons.append({
            'Sample': label,
            'N': results[label]['n'],
            'Baseline AIC': f"{results[label]['baseline_aic']:.1f}",
            'Core AIC': f"{results[label]['core_aic']:.1f}",
            'ΔAIC': f"{results[label]['delta_aic']:.1f}"
        })
    
    return pd.DataFrame(comparisons)
