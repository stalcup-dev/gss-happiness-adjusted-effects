#!/usr/bin/env python
"""Test CV with single fold."""

from src.config import get_paths
from src.io import load_happiness_csv
from src.modeling import HappinessModel
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, f1_score
import statsmodels.api as sm

paths = get_paths()
df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')

temp_model = HappinessModel(df)
y = temp_model.df['HAPPY_CODE'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
    print(f"Processing fold {fold_idx}...", flush=True)
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    model_test = HappinessModel(df_test)
    y_test = model_test.df['HAPPY_CODE'].values
    
    # Test BASELINE
    try:
        model_train = HappinessModel(df_train)
        print(f"  Train model created", flush=True)
        
        baseline_fit = model_train.build_baseline(use_ordinal=True)
        print(f"  Baseline fit done", flush=True)
        
        if 'error' not in baseline_fit:
            baseline_model = model_train.models['baseline_ordinal']
            print(f"  Got baseline model", flush=True)
            
            X_train, train_cols = model_train._prepare_data(model_train.df, ['YEAR'])
            print(f"  Prepared train: {X_train.shape}, cols={len(train_cols)}", flush=True)
            
            df_test_happy = model_test.df.copy()
            X_test = model_train._prepare_data(df_test_happy, ['YEAR'])[0]
            print(f"  Prepared test: {X_test.shape}", flush=True)
            
            for col in train_cols:
                if col not in X_test.columns:
                    X_test[col] = 0.0
            
            X_test = X_test[train_cols]
            print(f"  Aligned test: {X_test.shape}", flush=True)
            
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test)
            print(f"  Added constants", flush=True)
            
            baseline_preds = baseline_model.predict(X_test_const)
            print(f"  Got predictions: {baseline_preds.shape}", flush=True)
            
            baseline_preds = np.clip(baseline_preds, 1e-15, 1 - 1e-15)
            bl_log_loss = log_loss(y_test, baseline_preds, labels=[0, 1, 2])
            bl_f1 = f1_score(y_test, np.argmax(baseline_preds, axis=1), average='macro', zero_division=0)
            
            print(f"  Baseline - Log Loss: {bl_log_loss:.4f}, F1: {bl_f1:.4f}", flush=True)
    
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    break
