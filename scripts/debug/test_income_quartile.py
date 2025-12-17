from src.io import load_happiness_csv
from src.modeling import HappinessModel
import pandas as pd
import json

with open('src/paths.json') as f:
    paths = json.load(f)
    
df = load_happiness_csv(paths['raw_data_path'])
print(f'Original df has INCOME: {"INCOME" in df.columns}')

model = HappinessModel(df)

# Create INCOME_QUARTILE like in adjusted_effects
def rank_based_quantile_bin(series):
    if len(series) < 4:
        return pd.cut(series, bins=len(series.unique()), labels=False, duplicates='drop')
    ranks = series.rank(method='average')
    return pd.qcut(ranks, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

df_with_iq = model.df.copy()
df_with_iq['INCOME_QUARTILE'] = df_with_iq.groupby('YEAR')['INCOME'].transform(rank_based_quantile_bin)

print(f'df_with_iq has INCOME: {"INCOME" in df_with_iq.columns}')
print(f'df_with_iq has INCOME_QUARTILE: {"INCOME_QUARTILE" in df_with_iq.columns}')
print(f'df_with_iq columns count: {len(df_with_iq.columns)}')
