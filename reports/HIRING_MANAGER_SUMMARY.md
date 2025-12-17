# Hiring Manager Summary — GSS Happiness Analysis (2010–2022)

## Problem / question
This project asks: which respondent characteristics are most strongly associated with reporting “VERY HAPPY” in recent U.S. survey data?
It focuses on magnitude and direction of adjusted subgroup differences, and checks whether the model is meaningfully predictive or mainly interpretive.

## Data
- **Source:** General Social Survey (GSS)
- **Years / sample:** 2010–2022; complete-case analytic sample **N = 5,481**
- **Weights:** WTSSPS used for descriptive summaries and for averaging model-standardized predicted probabilities

## Methods
- Preprocessing guardrails (type checks, missing-code recoding via config, weight validation) plus unit tests
- Weighted descriptives for trends and crosstabs (WTSSPS)
- Multinomial logistic regression (statsmodels MNLogit) for “HAPPY” categories; core spec includes YEAR fixed effects + 9 core predictors
- Adjusted effects via marginal standardization (toggle one variable level at a time; predict; WTSSPS-weighted averaging)
- Trust checks: 5-fold cross-validation for predictive lift, plus documented sensitivity notes (including the 2021–2022 survey mode change)

## Findings (adjusted gaps in P(VERY HAPPY))
- **Health:** Excellent vs Poor = **+4.6 pp** (95% bootstrap stability interval: 4.56–4.59 pp)
- **Income position (within-year quartiles):** Q4 vs Q1 = **+2.6 pp** (95% bootstrap stability interval: 2.63–2.64 pp)
- **Marital status:** Married vs Divorced = **+2.9 pp** (95% bootstrap stability interval: 2.90–2.91 pp)

## So what
- If you need a single “where are differences largest?” headline, health-related group differences are larger than income/marital in this period.
- Because predictive lift is minimal versus a YEAR-only baseline, use the model to interpret associations and subgroup patterns—not as a strong classifier.
- For decision-making, prioritize measurement/segmentation narratives around health first, with income and marital status as secondary context.

## Limitations
- Observational survey analysis: results are associations and should not be read as causal effects.
- The multinomial regression fit is unweighted (statsmodels MNLogit limitation), so estimates may differ from fully survey-weighted population targets.
- The reported “bootstrap stability intervals” summarize resampling stability under the fitted model; they are not survey-design-based population intervals.

## How to run
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.run_analysis
python -m src.make_figures
```
