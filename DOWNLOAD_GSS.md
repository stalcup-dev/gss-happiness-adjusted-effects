# Download GSS Data

The General Social Survey (GSS) is maintained by NORC at the University of Chicago.

## Manual Download (Recommended for Phase 1-2)

1. Visit: https://gssdataexplorer.norc.org/
2. Create an extract with:
   - **Required variables:**
     - `YEAR` (sample year)
     - `HAPPY` (general happiness level)
     - **Weight columns** (choose one or both):
       - `WTSSPS` (postratification weight, all years) **← USE THIS FOR ANALYSIS**
       - `WTSSNRPS` (non-response adjusted weight, 2004–2022)
   - **Core predictors (Phase 2):**
     - `AGE`, `SEX`, `EDUC`, `INCOME`, `MARITAL`, `HEALTH`, `WRKSTAT`, `PARTYID`, `RACE`
   - **Extended predictors (Phase 3):**
     - `RELIG` (religion/spirituality; flagged for missingness review)
     - `REGION`, `DEGREE`, `OCCUP`, `CLASS` (optional)
   - **Output format:** CSV
3. Download the file and save it as:
   ```
   data/raw/gss_extract.csv
   ```

## Common Gotcha: Case Limit / Sample Size

GSS Data Explorer has a **10,000 case limit** per extract. If your selected year range and variables exceed this:
- Reduce year span (e.g., 2010–2022 instead of 2000–2022), OR
- Remove less-critical variables, OR
- Download multiple extracts and concatenate them

**Phase 1-2 validation requires:**
- Minimum **5,000 cases** after cleaning
- Year span of at least **5 years**
- WTSSPS weight column present, numeric, >0
- YEAR must be integer (no decimals)

If either threshold is not met, preprocessing will fail and prompt you to expand the extract.

## Alternative: GSS-Rdata Package (R users)

If you have R installed, the `gssr` package provides programmatic access:
```R
install.packages("gssr")
library(gssr)
data(gss_all)
write.csv(gss_all, "data/raw/gss_extract.csv", row.names=FALSE)
```

## Once Downloaded

After placing `gss_extract.csv` in `data/raw/`, run:

```powershell
# Activate venv (if not already active)
.\.venv\Scripts\Activate.ps1

# Phase 1: Print column structure, missing values, unique values
python -m src.schema_report

# Phase 1: Print exact labels for the HAPPY variable
python -m src.label_report

# Phase 2: Missingness by year + heatmap
python -m src.missingness_by_year

# Phase 2: Weighted trend + crosstabs
python -m src.make_figures
```

**Outputs:**
- `reports/tables/missingness_by_year.csv`
- `reports/figures/missingness_heatmap.png`
- `reports/figures/weighted_happy_trend.png`
- `reports/figures/crosstab_*.png` and `reports/tables/crosstab_*.csv`
- Review [DECISION_MEMO.md](../reports/DECISION_MEMO.md) for data quality summary
