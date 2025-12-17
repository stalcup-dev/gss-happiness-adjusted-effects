#!/usr/bin/env python
"""Test CV evaluation with full error reporting."""

import sys
import traceback
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path.cwd()))

from src.config import get_paths
from src.io import load_happiness_csv
from src.evaluate import evaluate_models_cv

def main():
    try:
        paths = get_paths()
        print(f"Loading data from {paths.data_raw / 'gss_extract.csv'}...")
        df = load_happiness_csv(paths.data_raw / 'gss_extract.csv')
        print(f"Loaded {len(df)} rows\n")
        
        print("Running 5-fold stratified cross-validation...")
        print("This may take 2-3 minutes...\n")
        
        result = evaluate_models_cv(df, cv_folds=5, random_state=42)
        
        print("Cross-Validation Results:")
        print(result)
        print()
        
        # Save results
        output_path = paths.reports_tables / 'cv_evaluation.csv'
        result.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
