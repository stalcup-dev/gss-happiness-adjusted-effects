# GSS Happiness Analysis (Phase 1–3: Data + Modeling)

This repo analyzes happiness data from the General Social Survey (GSS), a long-running US survey.


## Key Results (skim-proof)

![Adjusted happiness by health](reports/figures/adjusted_effects_health.png)

- Adjusted (model-standardized) P(VERY HAPPY): Excellent vs Poor health = **+4.6 pp** (95% bootstrap stability interval: 4.56–4.59 pp)
- Income position (within-year quartiles): Q4 vs Q1 **+2.6 pp** (95% bootstrap stability interval: 2.63–2.64 pp); Married vs Divorced **+2.9 pp** (95% bootstrap stability interval: 2.90–2.91 pp)
- Predictive lift is minimal; covariates are mainly interpretive (log loss: baseline 1.0329 ± 0.0018 vs core 1.0359 ± 0.0037)

**Bootstrap stability intervals:** Computed via nonparametric respondent-level bootstrap (B=500) with WTSSPS applied when averaging predicted probabilities. Intervals reflect stability under resampling with the fitted model; they do not incorporate survey strata/PSU design or model-refit uncertainty.

## So what?

What factors are most strongly associated with being “Very happy” in recent U.S. survey data (GSS, 2010–2022)?
Health shows the largest adjusted difference in P(VERY HAPPY) (Excellent vs Poor **+4.6 pp**), while income (Q4 vs Q1 **+2.6 pp**) and marital status (Married vs Divorced **+2.9 pp**) are smaller.
Predictive lift is minimal (core model does not beat a YEAR-only baseline on log loss), so treat results as interpretive—not a strong classifier.
If you’re prioritizing which subgroup comparisons to emphasize, health-related differences are larger than income/marital in this period.
Trust signals: WTSSPS-weighted descriptives and standardized predictions, preprocessing validation + unit tests, cross-validated metrics, and documented sensitivity notes (including the 2021–2022 mode change).

**Phase 1** focuses on data inspection and reporting.  
**Phase 2** adds weighted trends and crosstabs with survey-aware guardrails.  
**Phase 3** implements multinomial logit models with interpretations.

## Quickstart

### 1. Get the data

See [DOWNLOAD_GSS.md](DOWNLOAD_GSS.md) for manual or programmatic download.

Expected filename: `data/raw/gss_extract.csv`

**Required columns:** YEAR, HAPPY, WTSSPS, + 9 core predictors (AGE, SEX, EDUC, INCOME, MARITAL, HEALTH, WRKSTAT, PARTYID, RACE)

### 2. Run Phase 1-2 reports

```powershell
# Activate venv
.\.venv\Scripts\Activate.ps1

# Phase 1: Data schema
python -m src.schema_report
python -m src.label_report

# Phase 2: Missingness & trends
python -m src.missingness_by_year
python -m src.make_figures

# Phase 3: Multinomial logit models + plots
python -m src.run_analysis
```

## Phase 1-3 Deliverables

### Phase 1
- **schema_report.py**: Column list, dtypes, missing counts, unique values for HAPPY, YEAR, core predictors, weight columns
- **label_report.py**: Exact categories/value labels for HAPPY

### Phase 2
- **preprocess.py** (refactored):
  - Variable-specific missing-code recoding (config-based: `config/missing_codes.json`)
  - YEAR integer casting + WTSSPS validation
  - Selective numeric code recoding (only listed columns)
  
- **missingness_by_year.py**: Year-by-year missing data patterns
  - Output: `reports/tables/missingness_by_year.csv`
  - Plot: `reports/figures/missingness_heatmap.png`

- **make_figures.py**: Weighted analyses (WTSSPS)
  - Trend: `weighted_happy_trend.png`
  - Crosstabs: HAPPY vs HEALTH, MARITAL, INCOME (tables + plots)
  - Includes effect sizes and interpretation notes

- **DECISION_MEMO.md** (`reports/`): 1-page data quality summary
  - Ordinal HAPPY coding confirmed
  - Core vs. extended predictors flagged
  - RELIG missingness mechanism TBD
  - 2021–2022 comparability note (mode change)
  - All assumptions & validation rules documented

### Phase 3
- **modeling.py**: HappinessModel class for multinomial logit regression
  - Baseline: HAPPY ~ YEAR
  - Core: HAPPY ~ YEAR + 9 core predictors (primary model)
  - Extended: Core + RELIG (conditional on MAR assessment)
  - Standardized numeric predictors; one-hot categorical encoding

- **assess_relig.py**: RELIG missingness mechanism assessment
  - Conclusion: consistent with MAR; MNAR is untestable from observed data. Outcome-dependent missingness via HAPPY was not detected (see report).
  - Recommendation: INCLUDE_WITH_CAUTION (listwise deletion acceptable)

- **predict_plots.py**: Predicted probability visualizations
  - Plots for HEALTH, MARITAL, INCOME, YEAR predictors
  - Output: `reports/figures/predicted_probs_*.png`

- **run_analysis.py**: Master orchestration script
  - Fits baseline → core → extended models
  - Generates model comparison table (`reports/tables/model_comparison.csv`)
  - Creates predicted probability plots
  - Writes interpretation & limitations memo

- **INTERPRETATION_AND_LIMITATIONS.md** (`reports/`): Full Phase 3 narrative
  - Model specifications & sample sizes
  - Key assumptions (non-causality, MAR handling, 2021-2022 mode change)
  - Survey design & weighting note
  - Results summary & recommendations for principal analyst

### Testing
- **tests/test_preprocess.py**: Unit tests for missing-code recoding (safe numeric code selection, case-insensitive string patterns)

---

## Project Structure

```
happiness/
├── data/
│   ├── raw/
│   │   └── gss_extract.csv          # Input data
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── config.py                     # Path & config management
│   ├── io.py                         # Data loading & preprocessing
│   ├── schema_report.py              # Phase 1: Data inspection
│   ├── label_report.py               # Phase 1: Value labels
│   ├── missingness_by_year.py        # Phase 2: Missingness patterns
│   ├── make_figures.py               # Phase 2: Weighted crosstabs
│   ├── assess_relig.py               # Phase 3: RELIG MAR assessment
│   ├── modeling.py                   # Phase 3: HappinessModel class
│   ├── predict_plots.py              # Phase 3: Prediction visualizations
│   └── run_analysis.py               # Phase 3: Master orchestration
├── config/
│   └── missing_codes.json            # Variable-specific NA mapping
├── tests/
│   └── test_preprocess.py            # 18 unit tests (all passing)
├── reports/
│   ├── DECISION_MEMO.md              # Phase 2 data quality summary
│   ├── INTERPRETATION_AND_LIMITATIONS.md  # Phase 3 narrative
│   ├── figures/                      # PNG plots
│   └── tables/                       # CSV tables
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Results artifacts

- Adjusted effects tables/plots: [reports/tables](reports/tables) and [reports/figures](reports/figures)
- Narrative summary: [INSIGHTS.md](INSIGHTS.md)
- Modeling notes + limitations: [reports/INTERPRETATION_AND_LIMITATIONS.md](reports/INTERPRETATION_AND_LIMITATIONS.md)

---

## Survey Design & Weighting

- **Weights (Phase 2):** WTSSPS (survey weight) applied to descriptive analyses
- **Regression (Phase 3):** Unweighted multinomial logit (statsmodels MNLogit does not support frequency weights; documented as limitation)
- **Robustness:** Unweighted estimates may differ from population targets; sensitivity re-fitting in weighted survey package (R, SAS) available on request

---

## Data Quality Guardrails

- ✓ **Validation:** YEAR integer casting, WTSSPS > 0, n ≥ 5000, year span ≥ 5
- ✓ **Missing codes:** Config-based (variable-specific numeric codes + global string patterns)
- ✓ **Unit tests:** 18 tests passing (config loading, selective recoding, validation)
- ✓ **Reproducibility:** All code versioned; acceptance checks at each phase

---

## Interpretation & Next Steps

**For Principal Analyst:**

1. Review `INTERPRETATION_AND_LIMITATIONS.md` for causal claims policy
2. 2021-2022 sensitivity: Re-run Phase 2-3 excluding 2021-2022 if causality framing needed
3. Extended questions: Multiple imputation for RELIG if policy analysis required

**Technical Debt:**

- Proportional odds assumption test (alternative to MNLogit)
- Weighted regression (re-implement in R survey package or use statsmodels survey module)
- Interaction effects (health × marital, income × education)
- Time trends by subgroup (decade-specific models)

