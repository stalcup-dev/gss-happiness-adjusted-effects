#!/usr/bin/env python
"""Debug CV with detailed error tracking."""

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
import statsmodels.api as sm

paths = get_paths()
print("Loading data...")
df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')
print(f"Loaded {len(df)} rows\n")

# Get HAPPY_CODE
temp_model = HappinessModel(df)
y = temp_model.df['HAPPY_CODE'].values

# Setup CV - use just 2 folds for faster debugging
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

fold_num = 0
for train_idx, test_idx in skf.split(df, y):
    fold_num += 1
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}/2")
    print(f"{'='*70}")
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    # Create models to get HAPPY_CODE for test set
    model_test = HappinessModel(df_test)
    y_test = model_test.df['HAPPY_CODE'].values
    
    print(f"\nTest y_test shape: {y_test.shape}, values: {np.unique(y_test)}")
    
    # Test BASELINE
    print("\n" + "-"*70)
    print("BASELINE MODEL")
    print("-"*70)
    
    try:
        model_train = HappinessModel(df_train)
        print("✓ Created HappinessModel for train set")
        
        baseline_fit = model_train.build_baseline(use_ordinal=True)
        print(f"✓ build_baseline() returned: {baseline_fit.get('formula')}")
        
        if 'error' in baseline_fit:
            print(f"✗ ERROR in build: {baseline_fit['error']}")
        else:
            baseline_model = model_train.models['baseline_ordinal']
            print(f"✓ Retrieved model from models dict")
            print(f"  AIC: {baseline_fit.get('aic'):.2f}")
            
            # Get predictions on test set
            df_train_happy = model_train.df.copy()
            df_test_happy = model_test.df.copy()
            
            print(f"  Train happy shape: {df_train_happy.shape}")
            print(f"  Test happy shape: {df_test_happy.shape}")
            
            # Prepare training data (for encoding reference)
            X_train, train_vars = model_train._prepare_data(df_train_happy, ['YEAR'])
            print(f"✓ Prepared train data: {X_train.shape}")
            print(f"  Columns: {list(X_train.columns[:5])}...")
            
            # Prepare test data using training stats for numeric vars
            X_test_raw, _ = model_train._prepare_data(df_test_happy, ['YEAR'])
            print(f"✓ Prepared test data: {X_test_raw.shape}")
            print(f"  Columns: {list(X_test_raw.columns[:5])}...")
            
            X_train_const = sm.add_constant(X_train)
            X_test_const = sm.add_constant(X_test_raw)
            
            print(f"  Train with const: {X_train_const.shape}")
            print(f"  Test with const: {X_test_const.shape}")
            
            # Make predictions
            baseline_preds = baseline_model.predict(X_test_const)
            print(f"✓ Got predictions: shape={baseline_preds.shape}")
            print(f"  Sample probs: {baseline_preds[0]}")
            
            baseline_preds = np.clip(baseline_preds, 1e-15, 1 - 1e-15)
            
            # Evaluate
            bl_log_loss = log_loss(y_test, baseline_preds, labels=[0, 1, 2])
            bl_f1 = f1_score(y_test, np.argmax(baseline_preds, axis=1), average='macro', zero_division=0)
            
            print(f"✓ Log Loss: {bl_log_loss:.4f}")
            print(f"✓ Macro F1: {bl_f1:.4f}")
    
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Stop after first fold
    break
