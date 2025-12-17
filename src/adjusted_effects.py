"""
Compute adjusted predicted probabilities via marginal standardization (g-computation).

For each focal variable (HEALTH, MARITAL, INCOME_Q), set that variable to each of its 
levels for all rows, predict with the fitted core model, then average using WTSSPS.

Output: adjusted probabilities, pp gaps, bootstrap CIs.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .config import get_paths
from .io import load_happiness_csv
from .modeling import HappinessModel


def compute_adjusted_effects(
    model_obj: HappinessModel,
    df: pd.DataFrame,
    focal_var: str,
    core_vars: list[str] | None = None,
    weight_col: str = 'WTSSPS',
    ci_contrast: tuple[str, str] | None = None,
    n_boot: int = 500,
    random_state: int = 42,
    verbose: bool = False,
    debug_bootstrap: bool = False,
) -> dict:
    """
    Compute adjusted predicted probabilities for a focal variable.
    
    Uses marginal standardization (g-computation): for each level of focal_var,
    set only that variable to a given level for every row, predict P(HAPPY),
    and average using weight_col.
    
    Bootstrap CIs are computed via nonparametric respondent-level resampling:
    - For each of B resamples, draw n rows with replacement (n = full sample size)
    - Recompute adjusted effects end-to-end on resampled data
    - Weights (WTSSPS) are applied within each resample when averaging predictions
    - Does NOT use survey strata/PSU design; treats respondents as independent
    
    Args:
        model_obj: Fitted HappinessModel with core model
        df: Data (should be the training data or a representative sample)
        focal_var: Variable to compute effects for (e.g., 'HEALTH')
        core_vars: Core variables used in the model
        weight_col: Weight column for averaging
        ci_contrast: Explicit (bottom, top) contrast tuple for CI computation
        n_boot: Number of bootstrap resamples for CIs (default 500)
        random_state: Random seed for bootstrap
        verbose: Print progress messages
        debug_bootstrap: If True, print resampling diagnostics
        
    Returns:
        dict with adjusted probabilities, gaps, bootstrap CIs, and bootstrap samples
    """
    
    if core_vars is None:
        core_vars = model_obj.core_vars

    # Always prepare predictors via the trained pipeline (core_vars).
    # For INCOME_QUARTILE, we will override INCOME_Q_* dummy columns directly.
    vars_for_prep = core_vars.copy() if isinstance(core_vars, list) else list(core_vars)
    
    # Get fitted model
    core_model = model_obj.models.get('core_ordinal')
    if core_model is None:
        raise ValueError("Core model not fitted. Run model.build_core() first.")
    
    # Get levels of focal variable
    focal_levels = sorted(df[focal_var].dropna().astype(str).unique())
    
    if verbose:
        print(f"Computing adjusted effects for {focal_var} ({len(focal_levels)} levels)", flush=True)
    
    # Main analysis: compute adjusted probs for each level
    results = []
    
    exog_names = list(core_model.model.exog_names)

    def _build_exog(df_model: pd.DataFrame, level: str | None) -> pd.DataFrame:
        df_model_local = df_model.copy()

        # If INCOME_QUARTILE is the focal var, override the trained income dummy columns.
        if focal_var == 'INCOME_QUARTILE' and level is not None:
            income_dummy_cols = [c for c in ['INCOME_Q_Q2', 'INCOME_Q_Q3', 'INCOME_Q_Q4'] if c in df_model_local.columns]
            if income_dummy_cols:
                df_model_local[income_dummy_cols] = 0.0
                if level == 'Q2' and 'INCOME_Q_Q2' in df_model_local.columns:
                    df_model_local['INCOME_Q_Q2'] = 1.0
                elif level == 'Q3' and 'INCOME_Q_Q3' in df_model_local.columns:
                    df_model_local['INCOME_Q_Q3'] = 1.0
                elif level == 'Q4' and 'INCOME_Q_Q4' in df_model_local.columns:
                    df_model_local['INCOME_Q_Q4'] = 1.0
                # Q1 is the reference: all zeros

        X = pd.DataFrame(index=df_model_local.index)
        for name in exog_names:
            if name == 'const':
                X[name] = 1.0
            elif name in df_model_local.columns:
                X[name] = df_model_local[name].astype(float)
            else:
                X[name] = 0.0
        return X

    for level in focal_levels:
        # Create prediction dataset: set ONLY the focal variable level across all rows
        df_pred = df.copy()
        df_pred[focal_var] = level

        # Add dummy HAPPY_CODE column for _prepare_data to work
        if 'HAPPY_CODE' not in df_pred.columns:
            df_pred['HAPPY_CODE'] = 1  # Dummy value, won't be used

        df_pred_model, predictors = model_obj._prepare_data(df_pred, vars_for_prep)
        X_pred = _build_exog(df_pred_model[predictors], level)

        # Assertion: verify income quartile indicators are set correctly (if focal_var is INCOME_QUARTILE)
        if focal_var == 'INCOME_QUARTILE':
            income_q_cols = [c for c in ['INCOME_Q_Q2', 'INCOME_Q_Q3', 'INCOME_Q_Q4'] if c in X_pred.columns]
            if income_q_cols:
                row_sums = X_pred[income_q_cols].sum(axis=1).unique()
                # For Q1 (reference), all indicators should be 0; for Q2/Q3/Q4, exactly one should be 1
                expected_sum = 0 if level == 'Q1' else 1
                assert all(s == expected_sum for s in row_sums), (
                    f"Income quartile indicators not set correctly for {focal_var}={level}: "
                    f"expected sum={expected_sum}, got sums={list(row_sums)}"
                )
                # Further: verify that the RIGHT indicator is set for non-reference levels
                if level in ['Q2', 'Q3', 'Q4']:
                    expected_col = f'INCOME_Q_{level}'
                    if expected_col in income_q_cols:
                        assert (X_pred[expected_col] == 1.0).all(), (
                            f"Expected {expected_col}=1 for all rows when {focal_var}={level}"
                        )
                        for col in income_q_cols:
                            if col != expected_col:
                                assert (X_pred[col] == 0.0).all(), (
                                    f"Expected {col}=0 for all rows when {focal_var}={level}"
                                )

        preds = core_model.predict(X_pred)  # (n, 3) for NOT TOO / PRETTY / VERY
        
        # Convert to numpy if it's a DataFrame
        if hasattr(preds, 'values'):
            preds = preds.values
        
        # Average across rows using weight_col
        # Handle NaNs in predictions by replacing with 0
        preds_clean = np.nan_to_num(preds, nan=0.0)
        
        weights = df[weight_col].values / df[weight_col].sum()
        p_very = (preds_clean[:, 2] * weights).sum()  # VERY HAPPY is category 2
        p_pretty = (preds_clean[:, 1] * weights).sum()  # PRETTY HAPPY is category 1
        p_not_too = (preds_clean[:, 0] * weights).sum()  # NOT TOO HAPPY is category 0
        
        results.append({
            'level': level,
            'p_very': p_very,
            'p_pretty': p_pretty,
            'p_not_too': p_not_too,
            'n': len(df)
        })
    
    results_df = pd.DataFrame(results)
    
    # Define the contrast for the pp gap + CI
    if ci_contrast is not None:
        bottom_level, top_level = ci_contrast
    elif len(results_df) >= 2:
        # Default: top minus bottom by P(VERY)
        tmp = results_df.sort_values('p_very')
        bottom_level = str(tmp['level'].iloc[0])
        top_level = str(tmp['level'].iloc[-1])
    else:
        bottom_level = top_level = None

    if bottom_level is not None and top_level is not None:
        p_very_gap_point = (
            float(results_df.loc[results_df['level'].astype(str) == str(top_level), 'p_very'].iloc[0])
            - float(results_df.loc[results_df['level'].astype(str) == str(bottom_level), 'p_very'].iloc[0])
        )
    else:
        p_very_gap_point = 0.0
    
    # ========================================
    # Bootstrap CIs for pp gap (nonparametric respondent-level bootstrap)
    # ========================================
    np.random.seed(random_state)
    boot_gaps = []
    
    if debug_bootstrap:
        sampled_indices_per_rep = []
    
    for boot_rep in range(n_boot):
        # CRITICAL: Resample rows with replacement at the individual level
        # This creates a bootstrap sample of n rows (where n = len(df))
        boot_idx = np.random.choice(len(df), size=len(df), replace=True)
        df_boot = df.iloc[boot_idx].reset_index(drop=True)
        
        if debug_bootstrap and boot_rep < 2:
            sampled_indices_per_rep.append(boot_idx[:15])
        
        # Compute adjusted probs for bottom and top levels
        # CRITICAL: recompute end-to-end from scratch using the RESAMPLED data
        bottom_results = []
        top_results = []
        
        for level in [bottom_level, top_level]:
            if level is None:
                continue
            
            # Set focal variable to the specified level for all resampled rows
            df_pred = df_boot.copy()
            df_pred[focal_var] = level
            
            # Add dummy HAPPY_CODE column for _prepare_data to work
            if 'HAPPY_CODE' not in df_pred.columns:
                df_pred['HAPPY_CODE'] = 1  # Dummy value

            # CRITICAL: Prepare data on the RESAMPLED dataset
            # This ensures any standardization or feature engineering uses resampled statistics
            df_pred_model, predictors = model_obj._prepare_data(df_pred, vars_for_prep)
            X_pred = _build_exog(df_pred_model[predictors], str(level))
            
            # Predict using the trained model (fitted on full data)
            preds = core_model.predict(X_pred)
            
            # Convert to numpy if it's a DataFrame
            if hasattr(preds, 'values'):
                preds = preds.values
            
            # Handle NaNs in predictions
            preds = np.nan_to_num(preds, nan=0.0)
            
            # CRITICAL: Use resampled weights (renormalized within each resample)
            weights_boot = df_boot[weight_col].values / df_boot[weight_col].sum()
            p_very_boot = (preds[:, 2] * weights_boot).sum()
            
            if level == bottom_level:
                bottom_results.append(p_very_boot)
            else:
                top_results.append(p_very_boot)
        
        if bottom_results and top_results:
            gap_boot = top_results[0] - bottom_results[0]
            boot_gaps.append(gap_boot)
    
    boot_gaps = np.array(boot_gaps)
    
    # Debug: print bootstrap diagnostics
    if debug_bootstrap:
        print(f"\n[BOOTSTRAP DEBUG] {focal_var} {bottom_level} vs {top_level}")
        if len(sampled_indices_per_rep) >= 1:
            print(f"  Replicate 0, first 15 sampled indices: {sampled_indices_per_rep[0]}")
        if len(sampled_indices_per_rep) >= 2:
            print(f"  Replicate 1, first 15 sampled indices: {sampled_indices_per_rep[1]}")
            differ = not np.array_equal(sampled_indices_per_rep[0], sampled_indices_per_rep[1])
            print(f"    -> Indices differ? {differ}")
        if boot_gaps.size > 0:
            print(f"  Bootstrap gaps (B={len(boot_gaps)}):")
            print(f"    Mean:   {np.mean(boot_gaps) * 100:.4f} pp")
            print(f"    Std:    {np.std(boot_gaps) * 100:.4f} pp")
            print(f"    Min:    {np.min(boot_gaps) * 100:.4f} pp")
            print(f"    Max:    {np.max(boot_gaps) * 100:.4f} pp")
            print(f"    P2.5:   {np.percentile(boot_gaps, 2.5) * 100:.4f} pp")
            print(f"    P97.5:  {np.percentile(boot_gaps, 97.5) * 100:.4f} pp")
    
    if boot_gaps.size:
        gap_ci_lower = np.percentile(boot_gaps, 2.5)
        gap_ci_upper = np.percentile(boot_gaps, 97.5)
    else:
        gap_ci_lower = np.nan
        gap_ci_upper = np.nan
    
    return {
        'results_df': results_df,
        'focal_var': focal_var,
        'focal_levels': focal_levels,
        'bottom_level': bottom_level,
        'top_level': top_level,
        'p_very_gap_point': p_very_gap_point * 100,  # Convert to percentage points
        'gap_ci_lower': (gap_ci_lower * 100) if pd.notna(gap_ci_lower) else np.nan,
        'gap_ci_upper': (gap_ci_upper * 100) if pd.notna(gap_ci_upper) else np.nan,
        'n_total': len(df),
        'boot_gaps': boot_gaps * 100,  # Return raw bootstrap gaps (in pp) for saving
        'n_boot': n_boot,
    }


def plot_adjusted_effects(results: dict, out_dir: Path) -> Path:
    """Plot adjusted predicted probabilities for a focal variable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    focal_var = results['focal_var']
    results_df = results['results_df'].sort_values('p_very', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot stacked bars
    bottom = np.zeros(len(results_df))
    colors = {'p_very': '#27ae60', 'p_pretty': '#f39c12', 'p_not_too': '#e74c3c'}
    labels_map = {'p_very': 'Very Happy', 'p_pretty': 'Pretty Happy', 'p_not_too': 'Not Too Happy'}
    
    for col, (color, label) in zip(['p_not_too', 'p_pretty', 'p_very'], 
                                     [(colors['p_not_too'], labels_map['p_not_too']),
                                      (colors['p_pretty'], labels_map['p_pretty']),
                                      (colors['p_very'], labels_map['p_very'])]):
        ax.bar(results_df['level'].astype(str), results_df[col], bottom=bottom, 
               label=label, color=color, width=0.6)
        bottom += results_df[col].values
    
    ax.set_xlabel(focal_var, fontsize=12, fontweight='bold')
    ax.set_ylabel('Adjusted Probability', fontsize=12, fontweight='bold')
    ax.set_title(f'Adjusted Happiness Distribution by {focal_var}\n(Marginal Standardization)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    out_path = out_dir / f"adjusted_effects_{focal_var.lower()}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return out_path


def main(input_path: Path | None = None, table_dir: Path | None = None, fig_dir: Path | None = None, 
         n_boot: int = 500, debug_bootstrap: bool = False) -> None:
    """
    Main: compute adjusted effects for HEALTH, MARITAL, INCOME_QUARTILE.
    
    Args:
        input_path: Path to GSS data file
        table_dir: Output directory for CSVs
        fig_dir: Output directory for plots
        n_boot: Number of bootstrap resamples (default 500)
        debug_bootstrap: If True, print resampling diagnostics
    """
    paths = get_paths()
    
    if input_path is None:
        from .io import choose_default_gss_path
        input_path = choose_default_gss_path(paths.root)
    
    if table_dir is None:
        table_dir = paths.reports_tables
    
    if fig_dir is None:
        fig_dir = paths.reports_figures
    
    table_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data and fitting core model...")
    df = load_happiness_csv(input_path)
    
    model = HappinessModel(df)
    core_fit = model.build_core(use_ordinal=True)
    
    if 'error' in core_fit:
        print(f"ERROR: Core model fit failed: {core_fit['error']}")
        return
    
    print("Core model fitted successfully.\n")

    # Use the identical analytic subset used for model fitting
    df_core = model._get_analytic_subset(model.core_vars)
    
    # Create INCOME_QUARTILE for analysis using SAME binning as model
    def rank_based_quantile_bin(series):
        """Bin a series into quartiles using rank, matching model's method."""
        if len(series) < 4:
            return pd.cut(series, bins=len(series.unique()), labels=False, duplicates='drop')
        ranks = series.rank(method='average')  # Average rank for ties
        return pd.qcut(ranks, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    
    # Use analytic subset (already has HAPPY_CODE)
    df_with_iq = df_core.copy()
    df_with_iq['INCOME_QUARTILE'] = df_with_iq.groupby('YEAR')['INCOME'].transform(
        rank_based_quantile_bin
    )

    # Compute adjusted effects for each variable (explicit headline contrasts)
    focal_vars = [
        ('HEALTH', df_core, ('Poor', 'Excellent')),
        ('MARITAL', df_core, ('Divorced', 'Married')),
        ('INCOME_QUARTILE', df_with_iq, ('Q1', 'Q4')),
    ]

    for focal_var, df_for_var, ci_contrast in focal_vars:
        print(f"\n{'='*70}")
        print(f"Computing adjusted effects for {focal_var}")
        print(f"{'='*70}")
        
        try:
            results = compute_adjusted_effects(
                model, 
                df_for_var, 
                focal_var, 
                core_vars=model.core_vars,
                ci_contrast=ci_contrast,
                n_boot=n_boot,
                random_state=42,
                verbose=False,
                debug_bootstrap=debug_bootstrap,
            )
            
            # Save CSV
            results_df = results['results_df'].copy()
            results_df['p_very_pct'] = results_df['p_very'] * 100
            results_df['p_pretty_pct'] = results_df['p_pretty'] * 100
            results_df['p_not_too_pct'] = results_df['p_not_too'] * 100
            
            csv_path = table_dir / f"adjusted_effects_{focal_var.lower()}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"[OK] Saved CSV: {csv_path}")
            
            # Save bootstrap gaps
            if 'boot_gaps' in results:
                boot_gaps_path = table_dir / f"bootstrap_gaps_{focal_var.lower()}_{results['bottom_level']}_{results['top_level']}.csv"
                pd.DataFrame({'gap_pp': results['boot_gaps']}).to_csv(boot_gaps_path, index=False)
                print(f"[OK] Saved bootstrap gaps: {boot_gaps_path}")
            
            print(f"\nResults:")
            print(results_df[['level', 'p_very_pct', 'p_pretty_pct', 'p_not_too_pct']].to_string(index=False))
            
            # Print gap
            print(f"\nAdjusted P(Very Happy) gap ({results['top_level']} - {results['bottom_level']}):")
            print(f"  Point estimate: {results['p_very_gap_point']:.1f} pp")
            if pd.notna(results['gap_ci_lower']) and pd.notna(results['gap_ci_upper']):
                print(f"  Nonparametric respondent-level bootstrap 95% CI (B={results['n_boot']}): [{results['gap_ci_lower']:.1f}, {results['gap_ci_upper']:.1f}] pp")
            else:
                print("  95% CI: (not available)")
            
            # Plot
            plot_path = plot_adjusted_effects(results, fig_dir)
            print(f"[OK] Saved plot: {plot_path}")
        
        except Exception as e:
            print(f"ERROR computing adjusted effects for {focal_var}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    # Parse command-line arguments
    n_boot = 500
    debug_bootstrap = False
    
    if '--debug_bootstrap' in sys.argv:
        debug_bootstrap = True
        print("Bootstrap debugging enabled.")
    
    # Check for n_boot argument
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--n_boot='):
            try:
                n_boot = int(arg.split('=')[1])
                print(f"Using {n_boot} bootstrap resamples.")
            except ValueError:
                print(f"Invalid n_boot value: {arg}")
    
    main(n_boot=n_boot, debug_bootstrap=debug_bootstrap)
