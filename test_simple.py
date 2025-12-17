#!/usr/bin/env python
"""Minimal CV debug."""

import sys
sys.path.insert(0, '.')

from src.config import get_paths
from src.io import load_happiness_csv
from src.modeling import HappinessModel

paths = get_paths()
print("Loading...")
sys.stdout.flush()

df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')
print(f"Loaded {len(df)} rows")
sys.stdout.flush()

temp_model = HappinessModel(df)
print(f"Created model, df shape = {temp_model.df.shape}")
sys.stdout.flush()

# Take first 100 rows as train
df_train = df.iloc[:100].copy()
print(f"Train subset: {len(df_train)}")
sys.stdout.flush()

model_train = HappinessModel(df_train)
print(f"Created train model")
sys.stdout.flush()

baseline_fit = model_train.build_baseline(use_ordinal=True)
print(f"Built baseline: formula={baseline_fit.get('formula')}, error={baseline_fit.get('error')}")
sys.stdout.flush()
