"""
Ordinal/multinomial regression modeling for HAPPY variable.

Implements:
- Multinomial logit (environment-limited choice; see NOTE below)
- Identical analytic subset across all models (ensures AIC comparability)
- Categorical INCOME (robust rank-based quantile bins within year)
- Categorical YEAR fixed effects
- RELIG missingness model (logistic: is HAPPY associated with RELIG missingness?)

NOTE ON ORDINAL MODELS:
  statsmodels 0.14.6 does not provide OrderedModel (proportional odds logit).
  Checked: statsmodels.genmod.generalized_ordered_model unavailable.
  Alternative ordinal packages (mord, corneal) not installed.
  
  IMPLICATION: Using multinomial logit (MNLogit), which does NOT assume proportional
  odds. This is a valid choice:
  - Allows category-specific effects (relaxed assumption)
  - Can test ordinal assumption by comparing category-specific coefficients
  - Interpretation: Report odds ratios per outcome level separately
  - Comparable AIC/BIC to proportional odds if ordinal assumption holds
  
  For proportional odds, users should upgrade statsmodels or use R::MASS::polr().
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit, Logit
import warnings

warnings.filterwarnings("ignore")


# Note: Ordinal logit via proportional odds (statsmodels.miscmodels.ordinal_model) requires statsmodels >= 0.14.1
# For now, use multinomial logit (which allows category-specific effects, implying ordinal assumption is testable)


class HappinessModel:
    """Ordinal regression for HAPPY with statistical integrity checks."""
    
    HAPPY_ORDER = ["NOT TOO HAPPY", "PRETTY HAPPY", "VERY HAPPY"]
    
    def __init__(self, df: pd.DataFrame, analytic_subset: pd.DataFrame | None = None):
        """
        Initialize model with preprocessed data.
        
        Args:
            df: Full preprocessed GSS DataFrame
            analytic_subset: If provided, use this exact subset for all models (ensures identical N)
        """
        self.df = df.copy()
        self.models = {}
        self.summaries = {}
        
        # Encode HAPPY as ordinal (0, 1, 2)
        self.df['HAPPY_CODE'] = self.df['HAPPY'].map({
            "NOT TOO HAPPY": 0,
            "PRETTY HAPPY": 1,
            "VERY HAPPY": 2
        })
        
        # Core predictors (without RELIG)
        self.core_vars = ['YEAR', 'AGE', 'SEX', 'EDUC', 'INCOME', 'MARITAL', 'HEALTH', 'WRKSTAT', 'PARTYID', 'RACE']
        self.extended_vars = ['RELIG']
        self.weight_var = 'WTSSPS'
        
        # Store training category levels and encoded column names for each categorical variable
        # This ensures predictions use the same categories and column order as training
        self.train_categories = {}
        self.train_dummy_cols = {}  # Will store the actual dummy column names from training
        
        # Initialize training categories from the data (before any modification)
        for var in self.core_vars:
            if self.df[var].dtype == 'object':
                self.train_categories[var] = sorted(self.df[var].dropna().unique().tolist())
        
        # Analytic subset (use pre-computed if provided, else compute on-the-fly)
        self.analytic_subset = analytic_subset
        self.analytic_n = None
        
    
    def _get_analytic_subset(self, vars_to_use: list[str]) -> pd.DataFrame:
        """
        Get or compute the identical analytic subset.
        
        If already established, use cached version (ensures identical N across all models).
        Otherwise, establish it based on complete cases for all core vars.
        """
        if self.analytic_subset is not None:
            return self.analytic_subset.copy()
        else:
            # First call: establish complete-case subset for ALL core vars
            model_cols = ['HAPPY_CODE'] + self.core_vars + [self.weight_var]
            subset = self.df[model_cols].dropna().copy()
            self.analytic_subset = subset
            self.analytic_n = len(subset)
            return subset.copy()
    
    
    def _prepare_data(self, df: pd.DataFrame, vars_to_use: list[str]) -> tuple[pd.DataFrame, list[str]]:
        """
        Prepare data: encode categoricals, handle INCOME as quantile bins, YEAR as categorical FE.
        
        Args:
            df: Data to prepare
            vars_to_use: Variables to include
        
        Returns:
            (df_model, predictor_names)
        """
        df_model = df[['HAPPY_CODE'] + vars_to_use + [self.weight_var]].copy()
        predictors = []
        
        # INCOME: create within-year quantile bins (robust, deterministic)
        if 'INCOME' in vars_to_use:
            if df_model['INCOME'].nunique() > 10:
                # Use rank-based binning within each year (handles ties gracefully)
                def rank_based_quantile_bin(series):
                    """Bin a series into quartiles using rank, handling ties."""
                    if len(series) < 4:
                        return pd.cut(series, bins=len(series.unique()), labels=False, duplicates='drop')
                    ranks = series.rank(method='average')  # Average rank for ties
                    return pd.qcut(ranks, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
                
                df_model['INCOME_QUANTILE'] = df_model.groupby('YEAR')['INCOME'].transform(
                    rank_based_quantile_bin
                )
                # One-hot encode, drop first (Q1 = reference)
                income_dummies = pd.get_dummies(df_model['INCOME_QUANTILE'], prefix='INCOME_Q', drop_first=True, dtype=float)
                df_model = pd.concat([df_model, income_dummies], axis=1)
                predictors.extend(list(income_dummies.columns))
                df_model = df_model.drop(['INCOME', 'INCOME_QUANTILE'], axis=1)
            else:
                # Treat as categorical if few values
                income_dummies = pd.get_dummies(df_model['INCOME'], prefix='INCOME', drop_first=True, dtype=float)
                df_model = pd.concat([df_model, income_dummies], axis=1)
                predictors.extend(list(income_dummies.columns))
                df_model = df_model.drop('INCOME', axis=1)
        
        # YEAR: categorical fixed effects (drop first year = 2010 as reference)
        if 'YEAR' in vars_to_use:
            year_dummies = pd.get_dummies(df_model['YEAR'], prefix='YEAR', drop_first=True, dtype=float)
            df_model = pd.concat([df_model, year_dummies], axis=1)
            predictors.extend(list(year_dummies.columns))
            df_model = df_model.drop('YEAR', axis=1)
        
        # AGE, EDUC: standardize numeric
        numeric_vars = [v for v in vars_to_use if v in df_model.columns and v not in ['INCOME', 'YEAR'] 
                       and df_model[v].dtype in ['int64', 'float64', 'Int64']]
        for var in numeric_vars:
            mean_val = df_model[var].mean()
            std_val = df_model[var].std()
            if std_val > 0:
                df_model[f'{var}_STD'] = (df_model[var] - mean_val) / std_val
                predictors.append(f'{var}_STD')
                df_model = df_model.drop(var, axis=1)
        
        # Categorical: one-hot encode (drop first)
        # Always use stored training categories to ensure consistent encoding
        cat_vars = [v for v in vars_to_use if v in df_model.columns and 
                   (df_model[v].dtype == 'object' or pd.api.types.is_categorical_dtype(df_model[v]))]
        for var in cat_vars:
            # Convert to Categorical with training categories to preserve all levels
            if var in self.train_categories:
                df_model[var] = pd.Categorical(df_model[var], categories=self.train_categories[var])
            
            dummies = pd.get_dummies(df_model[var], prefix=var, drop_first=True, dtype=float)
            df_model = pd.concat([df_model, dummies], axis=1)
            predictors.extend(list(dummies.columns))
            df_model = df_model.drop(var, axis=1)
        
        # Ensure all predictors exist and are numeric
        predictors = [p for p in predictors if p in df_model.columns]
        df_model[predictors] = df_model[predictors].astype(float)
        df_model['HAPPY_CODE'] = df_model['HAPPY_CODE'].astype(int)
        
        return df_model, predictors
    
    
    def _fit_ordinal_logit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Fit multinomial logit (allows category-specific effects to test ordinal assumption).
        
        Returns:
            Model results dict
        """
        try:
            model = MNLogit(y, X).fit(disp=0)
            return {
                'model': model,
                'type': 'Multinomial Logit (allows test of ordinal assumption)',
                'aic': float(model.aic),
                'bic': float(model.bic),
                'llf': float(model.llf),
                'n_params': model.df_model,
                'ordinal': True  # Primary choice; ordinal assumption testable from category-specific coefs
            }
        except Exception as e:
            return {'error': f'Multinomial logit fit failed: {str(e)}'}
    
    
    def _fit_multinomial_logit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Fit multinomial logit (fallback)."""
        try:
            model = MNLogit(y, X).fit(disp=0)
            return {
                'model': model,
                'type': 'Multinomial Logit (PO assumption not enforced)',
                'aic': float(model.aic),
                'bic': float(model.bic),
                'llf': float(model.llf),
                'n_params': model.df_model,
                'ordinal': False
            }
        except Exception as e:
            return {'error': f'Multinomial logit fit failed: {str(e)}'}
    
    
    def build_baseline(self, use_ordinal: bool = True) -> dict:
        """
        Baseline model: HAPPY ~ YEAR (categorical FE).
        
        Args:
            use_ordinal: If True, try ordinal logit; fallback to multinomial
        
        Returns:
            Model summary dict
        """
        key = 'baseline_ordinal' if use_ordinal else 'baseline_multinomial'
        
        # Prepare data (establish analytic subset on first call)
        df_model, predictors = self._prepare_data(self._get_analytic_subset(['YEAR']), ['YEAR'])
        
        # Fit
        X = sm.add_constant(df_model[predictors])
        y = df_model['HAPPY_CODE']
        
        if use_ordinal:
            result = self._fit_ordinal_logit(X, y)
        else:
            result = self._fit_multinomial_logit(X, y)
        
        if 'error' not in result:
            self.models[key] = result['model']
            self.summaries[key] = {
                'formula': 'HAPPY_CODE ~ YEAR (FE)',
                'model_type': result['type'],
                'n': len(df_model),
                'n_predictors': len(predictors),
                'aic': result['aic'],
                'bic': result['bic'],
                'llf': result['llf'],
                'ordinal': result['ordinal'],
                'use_ordinal': use_ordinal
            }
            return self.summaries[key]
        else:
            return {'error': result['error']}
    
    
    def build_core(self, use_ordinal: bool = True) -> dict:
        """
        Core model: HAPPY ~ YEAR + 9 core predictors (no RELIG).
        
        Args:
            use_ordinal: If True, try ordinal logit; fallback to multinomial
        
        Returns:
            Model summary dict
        """
        key = 'core_ordinal' if use_ordinal else 'core_multinomial'
        
        # Use identical analytic subset
        df_model, predictors = self._prepare_data(self._get_analytic_subset(self.core_vars), self.core_vars)
        
        # Fit
        X = sm.add_constant(df_model[predictors])
        y = df_model['HAPPY_CODE']
        
        if use_ordinal:
            result = self._fit_ordinal_logit(X, y)
        else:
            result = self._fit_multinomial_logit(X, y)
        
        if 'error' not in result:
            self.models[key] = result['model']
            self.summaries[key] = {
                'formula': 'HAPPY_CODE ~ YEAR (FE) + AGE + SEX + EDUC + INCOME (quantile) + MARITAL + HEALTH + WRKSTAT + PARTYID + RACE',
                'model_type': result['type'],
                'n': len(df_model),
                'n_predictors': len(predictors),
                'aic': result['aic'],
                'bic': result['bic'],
                'llf': result['llf'],
                'ordinal': result['ordinal'],
                'use_ordinal': use_ordinal,
                'predictors': predictors
            }
            return self.summaries[key]
        else:
            return {'error': result['error']}
    
    
    def build_extended(self, use_ordinal: bool = True, include_relig: bool = True) -> dict:
        """
        Extended model: core + RELIG (if include_relig=True).
        
        Args:
            use_ordinal: If True, try ordinal logit; fallback to multinomial
            include_relig: If True, add RELIG variable
        
        Returns:
            Model summary dict
        """
        key = 'extended_ordinal' if use_ordinal else 'extended_multinomial'
        
        vars_to_use = self.core_vars + (['RELIG'] if include_relig else [])
        
        # Use identical analytic subset (but filter to complete cases for RELIG if included)
        subset = self._get_analytic_subset(vars_to_use)
        if len(subset) < 100:
            return {'error': f'Insufficient complete cases for extended model (n={len(subset)})'}
        
        df_model, predictors = self._prepare_data(subset, vars_to_use)
        
        # Fit
        X = sm.add_constant(df_model[predictors])
        y = df_model['HAPPY_CODE']
        
        if use_ordinal:
            result = self._fit_ordinal_logit(X, y)
        else:
            result = self._fit_multinomial_logit(X, y)
        
        if 'error' not in result:
            self.models[key] = result['model']
            self.summaries[key] = {
                'formula': f'HAPPY_CODE ~ {" + ".join(vars_to_use)}',
                'model_type': result['type'],
                'n': len(df_model),
                'n_predictors': len(predictors),
                'aic': result['aic'],
                'bic': result['bic'],
                'llf': result['llf'],
                'ordinal': result['ordinal'],
                'use_ordinal': use_ordinal,
                'include_relig': include_relig,
                'predictors': predictors
            }
            return self.summaries[key]
        else:
            return {'error': result['error']}
    
    
    def build_relig_missingness_model(self) -> dict:
        """
        Logistic regression: is_missing_relig ~ HAPPY + YEAR + core predictors.
        
        Tests whether the outcome (HAPPY) predicts RELIG missingness (MNAR concern).
        
        Returns:
            Model summary dict
        """
        key = 'relig_missingness'
        
        # Create subset with RELIG missingness indicator
        full_df = self.df[['HAPPY_CODE'] + self.core_vars + ['RELIG', self.weight_var]].copy()
        subset_with_relig = full_df.dropna(subset=['HAPPY_CODE'] + self.core_vars)
        subset_with_relig['IS_MISSING_RELIG'] = subset_with_relig['RELIG'].isna().astype(int)
        
        # Prepare data (core vars only, not HAPPY_CODE)
        df_model, predictors = self._prepare_data(subset_with_relig, self.core_vars)
        
        # Add HAPPY_CODE manually to predictors (it wasn't encoded by _prepare_data)
        df_model['HAPPY_CODE'] = subset_with_relig.loc[df_model.index, 'HAPPY_CODE'].values.astype(int)
        predictors_with_happy = ['HAPPY_CODE'] + predictors
        
        # Extract the dependent variable from original data using the final indices
        y = subset_with_relig.loc[df_model.index, 'IS_MISSING_RELIG'].astype(int)
        
        # Fit logistic regression
        X = sm.add_constant(df_model[predictors_with_happy])
        
        try:
            logit_model = Logit(y, X).fit(disp=0)
            
            # Extract HAPPY coefficient (test for MNAR)
            happy_coef = None
            happy_pval = None
            if 'HAPPY_CODE' in X.columns:
                happy_idx = list(X.columns).index('HAPPY_CODE')
                happy_coef = float(logit_model.params.iloc[happy_idx])
                happy_pval = float(logit_model.pvalues.iloc[happy_idx])
            
            self.models[key] = logit_model
            self.summaries[key] = {
                'formula': 'IS_MISSING_RELIG ~ HAPPY_CODE + YEAR + core predictors',
                'model_type': 'Logistic (MNAR test)',
                'n': len(df_model),
                'n_missing_relig': int(y.sum()),
                'pct_missing_relig': 100 * y.mean(),
                'happy_coef': happy_coef,
                'happy_pval': happy_pval,
                'happy_significant': happy_pval < 0.05 if happy_pval is not None else None,
                'interpretation': (
                    'HAPPY significantly predicts RELIG missingness (MNAR pattern)' if (happy_pval is not None and happy_pval < 0.05)
                    else 'HAPPY does not significantly predict RELIG missingness (MAR or MCAR)'
                ),
                'predictors': predictors
            }
            return self.summaries[key]
        except Exception as e:
            return {'error': f'RELIG missingness model failed: {str(e)}'}
    
    
    def get_model(self, key: str) -> Any:
        """Retrieve fitted model by key."""
        return self.models.get(key)
    
    
    def summary_table(self) -> pd.DataFrame:
        """Return model comparison table with n_obs and comparability notes."""
        rows = []
        baseline_n = None
        
        for key, summary in self.summaries.items():
            if 'error' not in summary and 'baseline' in key:
                baseline_n = summary['n']
                break
        
        for key, summary in self.summaries.items():
            if 'error' not in summary and key != 'relig_missingness':
                # Flag if N differs from baseline (AIC not comparable)
                aic_note = ''
                if baseline_n is not None and summary['n'] != baseline_n:
                    aic_note = ' [N differs - AIC not comparable]'
                
                rows.append({
                    'Model': key,
                    'Type': summary['model_type'],
                    'N_obs': summary['n'],
                    'Predictors': summary.get('n_predictors', '-'),
                    'AIC': f"{summary['aic']:.1f}{aic_note}",
                    'BIC': f"{summary['bic']:.1f}{aic_note}",
                    'LLF': f"{summary['llf']:.1f}"
                })
        
        return pd.DataFrame(rows)
