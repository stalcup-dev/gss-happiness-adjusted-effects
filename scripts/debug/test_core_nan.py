#!/usr/bin/env python
"""Debug core model NaN issue."""

from src.config import get_paths
from src.io import load_happiness_csv
from src.modeling import HappinessModel
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
    
    # Test CORE
    try:
        model_train = HappinessModel(df_train)
        print(f"  Train model created", flush=True)
        
        core_fit = model_train.build_core(use_ordinal=True)
        print(f"  Core fit done", flush=True)
        
        if 'error' not in core_fit:
            core_model = model_train.models['core_ordinal']
            print(f"  Got core model", flush=True)
            
            X_train, train_cols = model_train._prepare_data(model_train.df, model_train.core_vars)
            print(f"  Prepared train: {X_train.shape}, cols={len(train_cols)}", flush=True)
            print(f"  Train columns: {list(train_cols)}", flush=True)
            print(f"  Train has NaN: {X_train.isna().sum().sum()}", flush=True)
            
            df_test_happy = model_test.df.copy()
            X_test = model_train._prepare_data(df_test_happy, model_train.core_vars)[0]
            print(f"  Prepared test: {X_test.shape}", flush=True)
            print(f"  Test columns before align: {list(X_test.columns)}", flush=True)
            print(f"  Test has NaN before align: {X_test.isna().sum().sum()}", flush=True)
            
            for col in train_cols:
                if col not in X_test.columns:
                    X_test[col] = 0.0
                    print(f"    Added missing column: {col}", flush=True)
            
            X_test = X_test[train_cols]
            print(f"  Aligned test: {X_test.shape}", flush=True)
            print(f"  Test has NaN after align: {X_test.isna().sum().sum()}", flush=True)
            
            if X_test.isna().sum().sum() > 0:
                nan_cols = X_test.columns[X_test.isna().any()].tolist()
                print(f"  Columns with NaN: {nan_cols}", flush=True)
    
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    break
