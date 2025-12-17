#!/usr/bin/env python
"""Debug CV evaluation - detailed error reporting."""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.config import get_paths
from src.io import load_happiness_csv
from src.modeling import HappinessModel
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score

def debug_cv():
    paths = get_paths()
    print("Loading data...")
    df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')
    print(f"Loaded {len(df)} rows\n")
    
    # Get HAPPY_CODE
    temp_model = HappinessModel(df)
    y = temp_model.df['HAPPY_CODE'].values
    print(f"HAPPY_CODE value counts:\n{pd.Series(y).value_counts().sort_index()}\n")
    
    # Setup CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        print(f"=" * 60)
        print(f"FOLD {fold_idx + 1}/5")
        print(f"=" * 60)
        
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
        
        # Test baseline
        print("\nTesting BASELINE model...")
        try:
            model_train = HappinessModel(df_train)
            print(f"  - Created HappinessModel for train set")
            
            baseline_fit = model_train.build_baseline(use_ordinal=True)
            print(f"  - build_baseline() returned: {baseline_fit.get('formula', 'N/A')}")
            
            if 'error' in baseline_fit:
                print(f"  - ERROR in fit: {baseline_fit['error']}")
            else:
                baseline_model = model_train.models['baseline_ordinal']
                print(f"  - Model fit successful, AIC={baseline_fit.get('aic', 'N/A')}")
                
                # Try predictions
                try:
                    # Get test predictions
                    model_test = HappinessModel(df_test)
                    y_test = model_test.df['HAPPY_CODE'].values
                    
                    df_test_prep, predictors = model_train._prepare_data(df_test, ['YEAR'])
                    print(f"  - Test data prepared: {df_test_prep.shape}")
                    
                    import statsmodels.api as sm
                    X_test = sm.add_constant(df_test_prep)
                    preds = baseline_model.predict(X_test)
                    preds = np.clip(preds, 1e-15, 1 - 1e-15)
                    
                    bl_log_loss = log_loss(y_test, preds, labels=[0, 1, 2])
                    bl_f1 = f1_score(y_test, np.argmax(preds, axis=1), average='macro', zero_division=0)
                    
                    print(f"  - Log Loss: {bl_log_loss:.4f}, F1: {bl_f1:.4f} ✓")
                    
                except Exception as e:
                    print(f"  - ERROR during prediction: {type(e).__name__}: {e}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"  - ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        # Test core
        print("\nTesting CORE model...")
        try:
            model_train = HappinessModel(df_train)
            print(f"  - Created HappinessModel for train set")
            
            core_fit = model_train.build_core(use_ordinal=True)
            print(f"  - build_core() returned: {core_fit.get('formula', 'N/A')}")
            
            if 'error' in core_fit:
                print(f"  - ERROR in fit: {core_fit['error']}")
            else:
                core_model = model_train.models['core_ordinal']
                print(f"  - Model fit successful, AIC={core_fit.get('aic', 'N/A')}")
                
                # Try predictions
                try:
                    model_test = HappinessModel(df_test)
                    y_test = model_test.df['HAPPY_CODE'].values
                    
                    df_test_prep, predictors = model_train._prepare_data(df_test, model_train.core_vars)
                    print(f"  - Test data prepared: {df_test_prep.shape}")
                    
                    import statsmodels.api as sm
                    X_test = sm.add_constant(df_test_prep)
                    preds = core_model.predict(X_test)
                    preds = np.clip(preds, 1e-15, 1 - 1e-15)
                    
                    core_log_loss = log_loss(y_test, preds, labels=[0, 1, 2])
                    core_f1 = f1_score(y_test, np.argmax(preds, axis=1), average='macro', zero_division=0)
                    
                    print(f"  - Log Loss: {core_log_loss:.4f}, F1: {core_f1:.4f} ✓")
                    
                except Exception as e:
                    print(f"  - ERROR during prediction: {type(e).__name__}: {e}")
                    traceback.print_exc()
        
        except Exception as e:
            print(f"  - ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
        
        # Stop after first fold for debugging
        break

if __name__ == '__main__':
    debug_cv()
