"""
Generate predicted probability plots for key predictors with categorical YEAR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .modeling import HappinessModel


def plot_predicted_probs(model_obj: HappinessModel, out_dir: Path, predictor_list: list[str] | None = None) -> list[Path]:
    """
    Plot predicted probabilities of HAPPY levels for each predictor.
    Shows sample distributions binned by focal predictor (for now; can extend to model predictions).
    
    Args:
        model_obj: Fitted HappinessModel with models
        out_dir: Output directory for plots
        predictor_list: Predictors to plot (default: HEALTH, MARITAL, INCOME, YEAR)
    
    Returns:
        List of output paths
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = []
    
    if predictor_list is None:
        predictor_list = ['HEALTH', 'MARITAL', 'INCOME', 'YEAR']
    
    # Get core model (ordinal preferred)
    core_model = model_obj.get_model('core_ordinal') or model_obj.get_model('core_multinomial')
    if core_model is None:
        print("WARNING Core model not fitted yet; skipping prediction plots")
        return []
    
    happy_levels = ['NOT TOO HAPPY', 'PRETTY HAPPY', 'VERY HAPPY']
    
    for predictor in predictor_list:
        if predictor not in model_obj.df.columns:
            continue
        
        try:
            fig, ax = plt.subplots(figsize=(11, 6))
            
            # Categorical predictors: show by category
            if model_obj.df[predictor].dtype == 'object' or predictor in ['YEAR', 'MARITAL', 'HEALTH', 'SEX', 'PARTYID', 'RACE', 'WRKSTAT']:
                happy_by_pred = model_obj.df.groupby(predictor)['HAPPY'].value_counts(normalize=True).unstack(fill_value=0)
                # Reorder columns
                happy_by_pred = happy_by_pred[[c for c in happy_levels if c in happy_by_pred.columns]]
                happy_by_pred.plot(kind='bar', stacked=True, ax=ax, width=0.7, color=['#e74c3c', '#f39c12', '#27ae60'])
                ax.set_xlabel(predictor, fontsize=12)
                ax.set_title(f'Happiness Distribution by {predictor} (n={len(model_obj.df)})', fontsize=14, fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Numeric predictors: bin and plot
            else:
                pred_binned = pd.cut(model_obj.df[predictor].dropna(), bins=5)
                plot_df = model_obj.df[model_obj.df[predictor].notna()].copy()
                plot_df['_bin'] = pred_binned
                happy_by_pred = plot_df.groupby('_bin')['HAPPY'].value_counts(normalize=True).unstack(fill_value=0)
                happy_by_pred = happy_by_pred[[c for c in happy_levels if c in happy_by_pred.columns]]
                happy_by_pred.plot(kind='bar', stacked=True, ax=ax, width=0.7, color=['#e74c3c', '#f39c12', '#27ae60'])
                ax.set_xlabel(f'{predictor} (quintiles)', fontsize=12)
                ax.set_title(f'Happiness Distribution by {predictor} (n={len(plot_df)})', fontsize=14, fontweight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            ax.set_ylabel('Proportion', fontsize=12)
            ax.legend(title='HAPPY', labels=happy_levels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            out_path = out_dir / f"predicted_probs_{predictor.lower()}.png"
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            out_paths.append(out_path)
        
        except Exception as e:
            print(f"WARNING Failed to plot {predictor}: {e}")
            continue
    
    return out_paths


if __name__ == "__main__":
    from .io import load_happiness_csv
    from .config import get_paths
    
    paths = get_paths()
    df = load_happiness_csv(paths.data_raw / "gss_extract.csv")
    
    model_obj = HappinessModel(df)
    model_obj.build_core()
    
    plot_paths = plot_predicted_probs(model_obj, paths.reports_figures)
    for p in plot_paths:
        print(f"OK {p}")
