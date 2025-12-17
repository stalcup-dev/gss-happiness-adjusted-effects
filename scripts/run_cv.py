"""Run cross-validation evaluation"""
from src.config import get_paths
from src.io import load_happiness_csv
from src.evaluate import evaluate_models_cv

paths = get_paths()
df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')

print('Running 5-fold stratified cross-validation evaluation...')
print('(This may take 2-3 minutes)')
print()

cv_results = evaluate_models_cv(df, cv_folds=5, random_state=42)

print('Cross-Validation Results:')
print()
print(cv_results.to_string(index=False))
print()

# Save to CSV
cv_results.to_csv(paths.reports_tables / 'cv_evaluation.csv', index=False)
print(f'Results saved to: {paths.reports_tables}/cv_evaluation.csv')
