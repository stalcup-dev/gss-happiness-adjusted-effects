# Happiness in the GSS (2010–2022): adjusted and unadjusted insights

So what: In adjusted (model-standardized) probabilities, P(VERY HAPPY) is higher for Excellent vs Poor health (**+4.6 pp**), higher for income quartile Q4 vs Q1 (**+2.6 pp**), and higher for Married vs Divorced (**+2.9 pp**).
These are modest but stable associations over 2010–2022—useful for interpretation and subgroup framing rather than as a strong individual-level predictor (predictive lift in CV is minimal).
Key limitation: this is observational survey data, and the multinomial regression itself is unweighted (statsmodels MNLogit limitation), so estimates may differ from fully survey-weighted population targets.
See the limitations memo: [reports/INTERPRETATION_AND_LIMITATIONS.md](reports/INTERPRETATION_AND_LIMITATIONS.md).

This repo analyzes self-reported happiness in the U.S. General Social Survey (GSS), 2010–2022. Numbers below use the complete-case analytic sample from the Phase 3 core model (unweighted N=5,481) and WTSSPS weights for descriptive summaries.

## Adjusted insights (marginal standardization)

Primary results come from model-based standardization (also called g-computation / marginal standardization):

- Fit the Phase 3 core multinomial model (YEAR fixed effects + AGE, SEX, EDUC, income position, MARITAL, HEALTH, WRKSTAT, PARTYID, RACE).
- For a focal variable, set only that variable to a given level for every row, predict P(HAPPY), then average predicted probabilities using WTSSPS.

These are associations (not causal effects), but they are less confounded than raw crosstabs because they hold the other covariates at their observed distribution.

Source tables: [reports/tables/adjusted_effects_health.csv](reports/tables/adjusted_effects_health.csv), [reports/tables/adjusted_effects_income_quartile.csv](reports/tables/adjusted_effects_income_quartile.csv), [reports/tables/adjusted_effects_marital.csv](reports/tables/adjusted_effects_marital.csv).

### HEALTH (headline: Excellent vs Poor)

- Adjusted P(VERY HAPPY): Excellent = 32.6%, Poor = 28.0%
- Gap (Excellent − Poor): **+4.6 pp** (95% bootstrap stability interval: 4.56–4.59 pp)
- Unweighted N (analytic sample): Excellent n=1,654; Poor n=284

Interpretation: self-reported health status has the largest adjusted association with reporting “VERY HAPPY” in this analysis.

### INCOME_QUARTILE (headline: Q4 vs Q1)

Income is represented as within-year quartiles (rank-based within each YEAR). For adjusted effects, the counterfactual toggles only the income-quartile indicators used by the trained model.

- Adjusted P(VERY HAPPY): Q4 = 33.0%, Q1 = 30.4%
- Gap (Q4 − Q1): **+2.6 pp** (95% bootstrap stability interval: 2.63–2.64 pp)
- Unweighted N (analytic sample): Q4 n=1,274; Q1 n=1,446

Note: the pattern is not strictly monotonic across quartiles (Q1–Q3 are tightly clustered), but Q4 is clearly higher in this sample.

### MARITAL (headline: Married vs Divorced)

- Adjusted P(VERY HAPPY): Married = 31.1%, Divorced = 28.2%
- Gap (Married − Divorced): **+2.9 pp** (95% bootstrap stability interval: 2.90–2.91 pp)
- Unweighted N (analytic sample): Married n=2,727; Divorced n=825

Footnote on interpretability: Widowed shows a higher adjusted P(VERY HAPPY) than Divorced in this sample, but comparisons involving widowed respondents are especially susceptible to residual confounding (e.g., age, cohort, selection). Treat “widowed vs divorced” as descriptive, not explanatory.

## Unadjusted (weighted) descriptive patterns

These are weighted crosstabs of HAPPY by group (WTSSPS), reported conservatively as associations. Source tables: [reports/tables/crosstab_health_x_happy.csv](reports/tables/crosstab_health_x_happy.csv), [reports/tables/crosstab_marital_x_happy.csv](reports/tables/crosstab_marital_x_happy.csv), [reports/tables/crosstab_income_quartile_x_happy.csv](reports/tables/crosstab_income_quartile_x_happy.csv).

- HEALTH: Excellent = 32.42% VERY HAPPY vs Poor = 29.58% (difference: +2.84 pp)
- MARITAL: Married = 31.24% VERY HAPPY vs Divorced = 28.09% (difference: +3.15 pp)
- INCOME_QUARTILE: Q4 = 32.97% VERY HAPPY vs Q1 = 30.67% (difference: +2.30 pp)

## Predictive performance (why this is mostly interpretive)

In 5-fold CV ([reports/tables/cv_evaluation.csv](reports/tables/cv_evaluation.csv)), adding covariates provides minimal predictive lift over a YEAR-only baseline:

- Log loss: Baseline 1.0329 ± 0.0018 vs Core 1.0359 ± 0.0037
- Macro F1: Baseline 0.2208 ± 0.0003 vs Core 0.2248 ± 0.0027

Takeaway: use the covariates to describe adjusted associations and subgroup patterns, not as a strong individual-level predictor of happiness.
## Bootstrap stability intervals

All reported 95% bootstrap stability intervals for adjusted effects gaps use **nonparametric respondent-level bootstrap (B=500)**:

1. For each of 500 resamples: draw n rows with replacement (n = 5,481)
2. For each focal variable level (e.g., Excellent vs Poor), compute adjusted P(VERY HAPPY) using the fitted model
3. Calculate the gap (top level − bottom level)
4. Report the [2.5th, 97.5th] percentiles of the 500 resampled gaps

Resampling is **unweighted** (each row has equal probability of selection), then WTSSPS weights are applied within each resample when averaging predicted probabilities. These intervals reflect resampling stability under the fitted model and do **NOT** incorporate survey strata/PSU design (respondents are treated as independent units).
## Interpretation guardrails

- Observational survey data: treat differences as “associated with,” not causal.
- Weights: WTSSPS is used for descriptive crosstabs and for averaging predicted probabilities; the multinomial model fit itself is unweighted (statsmodels MNLogit limitation).
- Income quartiles: within-year position is used to reduce comparability issues across survey years.
- 2021–2022 mode change: interpret time-related comparisons with caution; see the sensitivity analysis in [src/sensitivity.py](src/sensitivity.py).

---

**Bootstrap methodology**: Bootstrap stability intervals are computed via nonparametric respondent-level bootstrap (B=500) with WTSSPS applied when averaging predicted probabilities. This approach does not incorporate survey strata/PSU design or model-refit uncertainty.