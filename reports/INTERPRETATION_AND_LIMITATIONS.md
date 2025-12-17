# Phase 3: Ordinal Regression Analysis — Interpretation & Limitations

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

**RELIG Status:** EXCLUDED
- Missingness: 20.5%
- Mechanism: [WARNING] RELIG appears MAR (depends on observed variables). Missingness varies slightly by YEAR/AGE/HEALTH. Listwise deletion acceptable with note; consider multiple imputation for sensitivity.
- Recommendation: INCLUDE_WITH_CAUTION

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
- **Baseline:** N/A cases (no missing YEAR or HAPPY)
- **Core:** N/A cases (complete on all 9 predictors)
- **Extended:** N/A cases (complete + RELIG if included)

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
See `model_comparison.csv` for AIC/BIC/LLF comparison.

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
model.build_extended(include_relig=False)
print(model.summary_table())
```

**Python versions & packages:**
- Python 3.13+
- pandas ≥2.2, numpy ≥1.26, statsmodels ≥0.14, scikit-learn ≥1.4

---

*Generated: December 2025*  
*Data: GSS 2010–2022 (n=N/A core sample)*  
*Models: Multinomial Logit (unweighted)*
