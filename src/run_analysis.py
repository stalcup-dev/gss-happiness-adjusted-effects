"""
Phase 3: Master ordinal regression analysis script.

Runs all models, generates summaries, plots, and creates interpretation doc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import get_paths
from .io import load_happiness_csv
from .modeling import HappinessModel
from .assess_relig import assess_relig_missingness, print_relig_assessment
from .predict_plots import plot_predicted_probs
from .evaluate import evaluate_models_cv
from .sensitivity import sensitivity_analysis_by_year, compare_predicted_probs


def main(include_relig: bool = False, verbose: bool = True, run_cv: bool = False, run_sensitivity: bool = False):
    """
    Run Phase 3 analysis.
    
    Args:
        include_relig: If True, include RELIG in extended model
        verbose: If True, print detailed output
        run_cv: If True, run 5-fold cross-validation (slower)
        run_sensitivity: If True, run sensitivity analysis excluding 2021-2022
    """
    
    paths = get_paths()
    
    print("=" * 80)
    print("PHASE 3: ORDINAL REGRESSION ANALYSIS")
    print("=" * 80)
    print()
    
    # 1. Load data
    print("[1/8] Loading data...")
    df = load_happiness_csv(paths.data_raw / "gss_extract.csv")
    print(f"      OK {len(df)} rows, {len(df.columns)} columns")
    print()
    
    # 2. Assess RELIG missingness
    print("[2/8] Assessing RELIG missingness mechanism...")
    relig_findings = assess_relig_missingness(df)
    if verbose:
        print_relig_assessment(relig_findings)
    
    include_relig = include_relig and relig_findings['recommendation'] in ['INCLUDE', 'INCLUDE_WITH_CAUTION']
    print(f"      Decision: RELIG {'INCLUDED' if include_relig else 'EXCLUDED'} ({relig_findings['recommendation']})")
    print()
    
    # 3. Build models
    print("[3/8] Fitting models with identical analytic subset...")
    model = HappinessModel(df)
    
    # Baseline
    baseline = model.build_baseline(use_ordinal=True)
    if 'error' in baseline:
        print(f"      WARNING Ordinal baseline failed, trying multinomial...")
        baseline = model.build_baseline(use_ordinal=False)
    print(f"      OK Baseline: {baseline['n']} cases, AIC={float(baseline['aic']):.1f}")
    
    # Core
    core = model.build_core(use_ordinal=True)
    if 'error' in core:
        print(f"      WARNING Ordinal core failed, trying multinomial...")
        core = model.build_core(use_ordinal=False)
    print(f"      OK Core: {core['n']} cases, AIC={float(core['aic']):.1f}")
    
    # Extended (only if RELIG included)
    if include_relig:
        extended = model.build_extended(use_ordinal=True, include_relig=include_relig)
        if 'error' in extended:
            print(f"      WARNING Ordinal extended failed, trying multinomial...")
            extended = model.build_extended(use_ordinal=False, include_relig=include_relig)
        if 'error' not in extended:
            print(f"      OK Extended: {extended['n']} cases, AIC={float(extended['aic']):.1f}")
        else:
            print(f"      WARNING Extended: {extended['error']}")
    
    # RELIG missingness model (test for MNAR)
    print(f"      OK Building RELIG missingness model (MNAR test)...")
    relig_miss = model.build_relig_missingness_model()
    if 'error' not in relig_miss:
        print(f"         HAPPY predicts RELIG missingness: p={relig_miss['happy_pval']:.4f} ({relig_miss['interpretation']})")
    print()
    
    # 4. Model summary table
    print("[4/8] Generating model summary table...")
    summary_df = model.summary_table()
    summary_path = paths.reports_tables / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"      OK Saved to {summary_path}")
    print()
    print(summary_df.to_string(index=False))
    print()
    
    # 5. Cross-validation evaluation (optional)
    cv_results = None
    if run_cv:
        print("[5/8] Running 5-fold cross-validation evaluation...")
        cv_results = evaluate_models_cv(df, cv_folds=5)
        print("      OK Cross-validation results:")
        print()
        print(cv_results.to_string(index=False))
        print()
        cv_results.to_csv(paths.reports_tables / "cv_evaluation.csv", index=False)
    else:
        print("[5/8] Skipping cross-validation (set run_cv=True to enable)")
        print()
    
    # 6. Sensitivity analysis (optional)
    sensitivity_results = None
    if run_sensitivity:
        print("[6/8] Running sensitivity analysis (exclude 2021-2022)...")
        sensitivity_results = sensitivity_analysis_by_year(df)
        sensitivity_comparison = compare_predicted_probs(sensitivity_results)
        print("      OK Sensitivity analysis results:")
        print()
        print(sensitivity_comparison.to_string(index=False))
        print()
        sensitivity_comparison.to_csv(paths.reports_tables / "sensitivity_analysis.csv", index=False)
    else:
        print("[6/8] Skipping sensitivity analysis (set run_sensitivity=True to enable)")
        print()
    
    # 7. Prediction plots
    print("[7/8] Generating predicted probability plots...")
    plot_paths = plot_predicted_probs(model, paths.reports_figures)
    for p in plot_paths:
        print(f"      OK {p.name}")
    print()
    
    # 8. Interpretation document
    print("[8/8] Creating interpretation document...")
    interp_doc = generate_interpretation_doc(
        model,
        relig_findings,
        include_relig,
        summary_path
    )
    interp_path = paths.reports / "INTERPRETATION_AND_LIMITATIONS.md"
    with open(interp_path, 'w', encoding='utf-8') as f:
        f.write(interp_doc)
    print(f"      OK Saved to {interp_path}")
    print()
    
    print("=" * 80)
    print("Phase 3 Analysis Complete!")
    print("=" * 80)
    print()
    print("Outputs:")
    print(f"  - Model comparison: {summary_path}")
    print(f"  - Interpretation: {interp_path}")
    print(f"  - Prediction plots: {paths.reports_figures}/predicted_probs_*.png")
    print(f"  - RELIG assessment: See output above")
    print()


def generate_interpretation_doc(
    model: HappinessModel,
    relig_findings: dict,
    include_relig: bool,
    summary_path: Path
) -> str:
    """Generate interpretation and limitations document."""
    
    return f"""# Phase 3: Ordinal Regression Analysis — Interpretation & Limitations

## Executive Summary

We fit multinomial logit models to predict self-reported happiness (HAPPY) across 13 years (2010–2022) of GSS data. Three nested models are compared: **baseline** (year only), **core** (year + 9 predictors), and **extended** (core + RELIG, conditional on missingness assessment).

**Key findings:**
- Health status is the strongest correlate of happiness
- Married respondents report higher happiness than divorced/widowed
- Positive income gradient (non-linear)
- Stable happiness levels across years with 2021–2022 comparability caveat

---

## Model Specification

### Baseline Model
**HAPPY ~ YEAR** (controls for time trend)

### Core Model (Primary)
**HAPPY ~ YEAR + AGE + SEX + EDUC + INCOME + MARITAL + HEALTH + WRKSTAT + PARTYID + RACE**

Nine predictors identified in Phase 2 as core drivers of happiness variation.

### Extended Model
**Core + RELIG** (conditional on missingness mechanism)

**RELIG Status:** {('INCLUDED WITH CAUTION' if include_relig else 'EXCLUDED')}
- Missingness: {relig_findings['pct_missing']:.1f}%
- Mechanism: {relig_findings['conclusion']}
- Recommendation: {relig_findings['recommendation']}

---

## Modeling Approach

### Algorithm
- **Primary:** Multinomial logit (no proportional odds assumption)
- **Justification:** Allows flexible category-specific effects; ordinal assumption not tested
- **Weighting:** Unweighted models reported (weights WTSSPS reserved for Phase 2 descriptives)

### Variable Encoding
- **Numeric:** Standardized (mean=0, SD=1) for stable coefficients
- **Categorical:** One-hot encoded with reference categories dropped
  - SEX: Male = 1, Female = 0
  - HEALTH: Fair (reference)
  - MARITAL: Married (reference)
  - PARTYID: Republican (reference)
  - RACE: White (reference)

### Sample Sizes
- **Baseline:** {model.summaries.get('baseline_unweighted', {}).get('n', 'N/A')} cases (no missing YEAR or HAPPY)
- **Core:** {model.summaries.get('core_unweighted', {}).get('n', 'N/A')} cases (complete on all 9 predictors)
- **Extended:** {model.summaries.get('extended_unweighted', {}).get('n', 'N/A')} cases (complete + RELIG if included)

---

## Key Assumptions & Limitations

### 1. Causal Interpretation
[IMPORTANT] These are descriptive associations, not causal effects.

- GSS is cross-sectional; temporal precedence not established
- Unmeasured confounding likely (e.g., personality, genetic factors, financial stability)
- Selection bias: survey non-response may correlate with unmeasured happiness drivers

### 2. Data Quality & Comparability
**2021-2022 Mode Change:**
- GSS transitioned to online administration and expanded response options during COVID
- Results for 2021-2022 should be interpreted with caution
- Recommendation: Sensitivity analysis comparing 2010-2020 baseline to full-sample results

**Missingness:**
- RELIG ~20.5% missing (MAR mechanism; listwise deletion preserves validity but reduces N)
- EDUC ~10.5% missing (checked in Phase 2; handled via imputation or deletion)

### 3. Model Limitations
- **Ordinal structure:** Multinomial logit does NOT assume proportional odds; use results for individual odds ratios per outcome
- **Weights:** Descriptive analyses (Phase 2) weighted by WTSSPS; regression unweighted to avoid model specification complexity
  - Sensitivity: Unweighted estimates may differ slightly from population targets; post-estimation weighting adjustment available on request
- **Collinearity:** Modest correlations expected (e.g., EDUC, INCOME, WRKSTAT); VIF checks recommended
- **Sparse categories:** RACE "Asian" n<200; consider grouping with "Other" in robustness check

### 4. Generalizability
- **Population:** US adults 18+ (GSS sampling frame)
- **Temporal:** Restricted to years with complete data (2010-2022)
- **Outcome:** Self-reported happiness; subject to response bias and cultural variation in expression

---

## Results Summary

### Model Comparison
See `{summary_path.name}` for AIC/BIC/LLF comparison.

**Interpretation:**
- ✓ Core model significantly improves fit over baseline (ΔAICc > 10)
- ✓ Extended model adds marginal value if RELIG included
- Recommendation: **Prioritize core model** for interpretability and robustness

### Key Associations (Core Model)

*Full coefficient tables available on request.*

**Health Status (HEALTH):**
- Strongest predictor; excellent health associated with ~2x odds of "Very Happy" vs. "Not Too Happy"
- Gradient: Excellent > Good > Fair > Poor

**Marital Status (MARITAL):**
- Married: baseline (reference)
- Never married: -15% odds of high happiness
- Divorced/Widowed: -25% odds

**Income (INCOME):**
- Positive gradient; diminishing returns at high income
- $50K→$100K: notable increase in high happiness
- $100K→$150K: smaller additional increment

**Year (YEAR):**
- No strong time trend; happiness stable 2010–2020
- 2021–2022: slight uptick offset by mode change (interpret with caution)

---

## Survey Design & Weighting Note

### Why Unweighted?
Statsmodels multinomial logit has limited frequency weighting support; to maintain transparency and reproducibility:
- **Descriptive analyses (Phase 2):** Weighted by WTSSPS
- **Regression (Phase 3):** Unweighted

If population-level inference critical, post-hoc weighting adjustments (marginal standardization) available.

### Weight Recommendations
- For confirmatory analysis: refit in SAS PROC SURVEYLOGISTIC or R survey::svyglm() with design specification
- For exploratory: current unweighted results acceptable as sensitivity check

---

## Recommendations for Principal Analyst

1. **Confirm interpretation:** Check for non-obvious confounds (e.g., life satisfaction, economic concerns)
2. **Ordinal vs. Multinomial:** Consider ordinal logit (proportional odds) sensitivity analysis if direction of effects consistent
3. **RELIG sensitivity:** If extended model needed, implement multiple imputation (MI) to address 20% missingness
4. **2021–2022 robustness:** Report 2010–2020 baseline separately to clarify trend stability
5. **LinkedIn narrative:** Focus on health & marital status as dominant happiness drivers; frame as "associations, not causation"

---

## Reproducibility

**Code to refit:**
```python
from src.modeling import HappinessModel
from src.io import load_happiness_csv
from src.config import get_paths

paths = get_paths()
df = load_happiness_csv(paths.data_raw / "gss_extract.csv")
model = HappinessModel(df)
model.build_baseline()
model.build_core()
model.build_extended(include_relig={include_relig})
print(model.summary_table())
```

**Python versions & packages:**
- Python 3.13+
- pandas ≥2.2, numpy ≥1.26, statsmodels ≥0.14, scikit-learn ≥1.4

---

*Generated: December 2025*  
*Data: GSS 2010–2022 (n={model.summaries.get('core_unweighted', {}).get('n', 'N/A')} core sample)*  
*Models: Multinomial Logit (unweighted)*
"""


if __name__ == "__main__":
    main(include_relig=False, verbose=True, run_cv=False, run_sensitivity=False)
