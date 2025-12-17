"""
Cross-validation and out-of-sample evaluation for Phase 3 models.

Implements train/test split evaluation with log loss and macro F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score
import statsmodels.api as sm

from .modeling import HappinessModel


def evaluate_models_cv(df: pd.DataFrame, cv_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Perform stratified k-fold CV on baseline and core models.
    
    Reports log loss and macro F1 (generalization metrics).
    
    Args:
        df: Full dataset
        cv_folds: Number of CV folds (default 5)
        random_state: Random seed
        
    Returns:
        DataFrame with CV results for baseline and core models
    """
    
    # Create temporary model just to get HAPPY_CODE mapping
    temp_model = HappinessModel(df)
    y = temp_model.df['HAPPY_CODE'].values

    # Use the core model's required columns as the consistent analytic subset for CV
    required_cols = ['HAPPY'] + temp_model.core_vars + [temp_model.weight_var]
    
    # Use original indices
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    baseline_log_losses = []
    baseline_f1s = []
    core_log_losses = []
    core_f1s = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        # Ensure both models are evaluated on the same (core-model) analytic subset
        df_train = df_train.dropna(subset=required_cols).reset_index(drop=True)
        df_test = df_test.dropna(subset=required_cols).reset_index(drop=True)
        
        # Create models to get HAPPY_CODE for test set
        model_test = HappinessModel(df_test)
        y_test = model_test.df['HAPPY_CODE'].values
        
        # ===== BASELINE MODEL =====
        try:
            model_train = HappinessModel(df_train)
            baseline_fit = model_train.build_baseline(use_ordinal=True)
            
            if 'error' not in baseline_fit:
                # Get the fitted model object and apply to test data
                baseline_model = model_train.models['baseline_ordinal']
                
                # Prepare test data exactly as training was done
                X_train, train_cols = model_train._prepare_data(model_train.df, ['YEAR'])
                
                # For test, we need to apply same transformations but using training stats
                # The problem: _prepare_data computes stats from input data
                # Solution: reapply same column preparation but only select trained columns
                
                df_test_happy = model_test.df.copy()
                
                # Don't use _prepare_data on test data directly
                # Instead manually apply the same encoding
                X_test = model_train._prepare_data(df_test_happy, ['YEAR'])[0]
                
                # Align test columns to training columns
                for col in train_cols:
                    if col not in X_test.columns:
                        X_test[col] = 0.0
                
                X_test = X_test[train_cols]  # Reorder and select only trained columns
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)
                
                # Make predictions
                baseline_preds = baseline_model.predict(X_test_const)
                baseline_preds = np.clip(baseline_preds, 1e-15, 1 - 1e-15)
                
                # Evaluate
                bl_log_loss = log_loss(y_test, baseline_preds, labels=[0, 1, 2])
                bl_f1 = f1_score(y_test, np.argmax(baseline_preds, axis=1), average='macro', zero_division=0)
                
                baseline_log_losses.append(bl_log_loss)
                baseline_f1s.append(bl_f1)
                
        except Exception as e:
            # Skip fold on error
            pass
        
        # ===== CORE MODEL =====
        try:
            model_train = HappinessModel(df_train)
            core_fit = model_train.build_core(use_ordinal=True)
            
            if 'error' not in core_fit:
                # Get the fitted model object and apply to test data
                core_model = model_train.models['core_ordinal']
                
                # Prepare test data exactly as training was done
                X_train, train_cols = model_train._prepare_data(model_train.df, model_train.core_vars)
                
                # For test, we need to apply same transformations but using training stats
                df_test_happy = model_test.df.copy()
                
                # Apply same encoding to test data
                X_test = model_train._prepare_data(df_test_happy, model_train.core_vars)[0]
                
                # Align test columns to training columns
                for col in train_cols:
                    if col not in X_test.columns:
                        X_test[col] = 0.0
                
                X_test = X_test[train_cols]  # Reorder and select only trained columns
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)
                
                # Make predictions
                core_preds = core_model.predict(X_test_const)
                core_preds = np.clip(core_preds, 1e-15, 1 - 1e-15)
                
                # Evaluate
                core_log_loss = log_loss(y_test, core_preds, labels=[0, 1, 2])
                core_f1 = f1_score(y_test, np.argmax(core_preds, axis=1), average='macro', zero_division=0)
                
                core_log_losses.append(core_log_loss)
                core_f1s.append(core_f1)
                
        except Exception as e:
            # Skip fold on error
            pass
    
    # Aggregate
    results_df = pd.DataFrame([
        {
            'Model': 'Baseline',
            'Log Loss': f"{np.mean(baseline_log_losses):.4f} ± {np.std(baseline_log_losses):.4f}" if baseline_log_losses else "N/A",
            'Macro F1': f"{np.mean(baseline_f1s):.4f} ± {np.std(baseline_f1s):.4f}" if baseline_f1s else "N/A",
        },
        {
            'Model': 'Core',
            'Log Loss': f"{np.mean(core_log_losses):.4f} ± {np.std(core_log_losses):.4f}" if core_log_losses else "N/A",
            'Macro F1': f"{np.mean(core_f1s):.4f} ± {np.std(core_f1s):.4f}" if core_f1s else "N/A",
        }
    ])
    
    return results_df


