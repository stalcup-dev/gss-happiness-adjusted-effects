#!/usr/bin/env python
"""Check what predictors are in the core model."""

from src.io import load_happiness_csv
from src.modeling import HappinessModel

df = load_happiness_csv('data/raw/gss_extract.csv')
model = HappinessModel(df)
result = model.build_core(use_ordinal=True)

print('Predictors in core model:')
income_preds = [p for p in result['predictors'] if 'INCOME' in p]
print(f'INCOME-related predictors: {income_preds}')
print(f'\nTotal predictors: {len(result["predictors"])}')
print(f'\nAll predictors:')
for i, p in enumerate(result['predictors']):
    print(f'{i}: {p}')
