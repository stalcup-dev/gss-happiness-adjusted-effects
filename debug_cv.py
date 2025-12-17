"""Debug cross-validation evaluation"""
from src.config import get_paths
from src.io import load_happiness_csv
from src.modeling import HappinessModel
from src.evaluate import evaluate_models_cv
import traceback

paths = get_paths()
df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')

print('Testing baseline model on full dataset...')
try:
    m = HappinessModel(df)
    baseline = m.build_baseline(use_ordinal=True)
    print(f"Baseline result: {baseline}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print('\nTesting core model on full dataset...')
try:
    m = HappinessModel(df)
    core = m.build_core(use_ordinal=True)
    print(f"Core result: {core}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
