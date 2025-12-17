"""Test sensitivity analysis"""
from src.config import get_paths
from src.io import load_happiness_csv
from src.sensitivity import sensitivity_analysis_by_year, compare_predicted_probs

paths = get_paths()
df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')

print('Running sensitivity analysis...')
print()
results = sensitivity_analysis_by_year(df)
comparison = compare_predicted_probs(results)
print('Sensitivity Analysis Complete')
print()
print(comparison.to_string(index=False))
