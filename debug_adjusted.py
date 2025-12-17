#!/usr/bin/env python
"""Debug script for adjusted effects computation."""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from src.io import load_happiness_csv
from src.modeling import HappinessModel
import statsmodels.api as sm

# Load data
print("Loading data...")
paths = Path(__file__).parent / 'data' / 'happiness.csv'
df = load_happiness_csv(str(paths))
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Check target
print(f"\nTarget variable HAPPY_CODE unique values: {sorted(df['HAPPY_CODE'].unique())}")
print(f"Target variable HAPPY_CODE value counts:\n{df['HAPPY_CODE'].value_counts()}")

# Fit model
print("\nFitting core model...")
core_vars = ['HEALTH', 'MARITAL', 'INCOME_QUARTILE', 'AGE_GROUP', 'YEAR']
model = HappinessModel(df, predictors=core_vars, weight_var='WTSSPS')
model.fit()
core_model = model.model
print(f"Model fitted successfully")
try:
    print(f"Model classes: {core_model.multinomial_model.classes_}")
except:
    print(f"Model type: {type(core_model)}")

# Test prediction on original data
print("\n=== Testing prediction on original data ===")
X_train, train_cols = model._prepare_data(model.df, core_vars)
X_const = sm.add_constant(X_train)
print(f"X_const shape: {X_const.shape}")
print(f"X_const dtypes:\n{X_const.dtypes}")
print(f"X_const has NaNs: {X_const.isna().sum().sum()}")

preds_train = core_model.predict(X_const)
print(f"Predictions type: {type(preds_train)}")
print(f"Predictions shape: {preds_train.shape}")
if hasattr(preds_train, 'values'):
    preds_train = preds_train.values
print(f"Predictions array shape: {preds_train.shape}")
print(f"Predictions sample:\n{preds_train[:5]}")
print(f"Predictions has NaNs: {np.isnan(preds_train).sum()}")
print(f"Predictions sum per row (should be ~1.0):\n{preds_train.sum(axis=1)[:5]}")

# Test with a modified dataset
print("\n=== Testing prediction with modified data (HEALTH=Excellent) ===")
df_pred = df.copy()
df_pred['HEALTH'] = 'Excellent'
if 'HAPPY_CODE' not in df_pred.columns:
    df_pred['HAPPY_CODE'] = 1
    
df_pred_prep, _ = model._prepare_data(df_pred, core_vars)
print(f"df_pred_prep shape: {df_pred_prep.shape}")
print(f"df_pred_prep dtypes:\n{df_pred_prep.dtypes}")
print(f"df_pred_prep has NaNs: {df_pred_prep.isna().sum().sum()}")

# Ensure alignment
for col in train_cols:
    if col not in df_pred_prep.columns:
        df_pred_prep[col] = 0.0
df_pred_prep = df_pred_prep[train_cols]
print(f"After alignment - df_pred_prep shape: {df_pred_prep.shape}")
print(f"After alignment - df_pred_prep has NaNs: {df_pred_prep.isna().sum().sum()}")

X_pred = sm.add_constant(df_pred_prep)
print(f"X_pred shape: {X_pred.shape}")
print(f"X_pred has NaNs: {X_pred.isna().sum().sum()}")

preds = core_model.predict(X_pred)
print(f"Predictions shape: {preds.shape}")
if hasattr(preds, 'values'):
    preds = preds.values
print(f"Predictions array shape: {preds.shape}")
print(f"Predictions sample:\n{preds[:5]}")
print(f"Predictions has NaNs: {np.isnan(preds).sum()}")

# Calculate weighted average
weights = df['WTSSPS'].values / df['WTSSPS'].sum()
print(f"\nWeights shape: {weights.shape}")
print(f"Weights sum: {weights.sum()}")
print(f"Weights have NaNs: {np.isnan(weights).sum()}")

p_very = (preds[:, 2] * weights).sum()
print(f"P(Very) = {p_very}")
