# GSS Happiness (2010–2022): portfolio one-pager

This project analyzes self-reported happiness in the U.S. General Social Survey (GSS) and translates survey crosstabs + a multinomial model into defensible, non-causal “adjusted associations.” The goal is a hiring-manager-ready artifact: clear findings, clear guardrails.

## Dataset

- Source: General Social Survey (GSS) extract
- Period: 2010–2022
- Analytic sample: N=5,481 complete cases (core model subset)
- Survey weight used for descriptive summaries: WTSSPS

## 3 key findings (adjusted)

(All adjusted results are marginal standardized predictions from the Phase 3 core model; see INSIGHTS.md.)

1) Health is the dominant adjusted association
- Adjusted P(VERY HAPPY): Excellent = 32.6% vs Poor = 28.0% (gap **+4.6 pp**, 95% CI: 4.56–4.59 pp)

2) Income position matters, but mostly at the top quartile
- Within-year income quartiles: Q4 = 33.0% vs Q1 = 30.4% (gap **+2.6 pp**, 95% CI: 2.63–2.64 pp)
- Q1–Q3 are tightly clustered; the pattern is not strictly monotonic

3) Marital status shows a meaningful (but observational) difference
- Adjusted P(VERY HAPPY): Married = 31.1% vs Divorced = 28.2% (gap **+2.9 pp**, 95% CI: 2.90–2.91 pp)

## Methods (high-level)

- Weights: WTSSPS applied to weighted descriptives and to average predicted probabilities
- Adjusted effects: marginal standardization (set only one focal variable level across rows; hold other covariates as observed)
- Confidence intervals: nonparametric respondent-level bootstrap (B=500) with WTSSPS applied in averaging; does not incorporate survey strata/PSU or model-refit uncertainty
- Cross-validation: baseline (YEAR-only) vs core model shows minimal predictive lift
- Sensitivity: reruns excluding 2021–2022 to gauge potential mode-change impact

## Limitations (what I did and did not claim)

- Observational comparisons: interpret as “associated with,” not causal effects
- Regression weighting: the multinomial regression fit is unweighted (statsmodels MNLogit limitation); weights are used for descriptive summaries and for standardizing predictions
- Modeling choice: multinomial logit used instead of proportional-odds ordinal logit due to environment constraints
- Derived income quartiles: “income position within year” is a useful normalization, but counterfactual quartile shifts should be read as descriptive scenarios, not literal interventions
