"""Unit tests for src/preprocess.py"""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import load_missing_codes_config, recode_gss_na, validate_gss_extract, preprocess_gss


class TestMissingCodesConfig:
    def test_load_config(self):
        """Test loading missing codes config from file."""
        config = load_missing_codes_config()
        assert "string_patterns" in config
        assert "numeric_codes" in config
        assert "_all" in config["string_patterns"]

    def test_load_config_not_found(self):
        """Test error on missing config file."""
        fake_path = Path("/nonexistent/path/config.json")
        with pytest.raises(FileNotFoundError):
            load_missing_codes_config(fake_path)


class TestRecodeGssNa:
    def test_recode_string_patterns(self):
        """Test recoding string missing patterns (IAP, DK, NA, REFUSED)."""
        df = pd.DataFrame({
            "COL1": ["VALUE", "DK", "NA", "REFUSED", "IAP"],
            "COL2": [1, 2, 3, 4, 5]
        })
        
        config = {
            "string_patterns": {"_all": ["DK", "NA", "REFUSED", "IAP"]},
            "numeric_codes": {}
        }
        
        result = recode_gss_na(df, config)
        
        # Check that string patterns were recoded
        assert pd.isna(result.loc[1, "COL1"])  # DK
        assert pd.isna(result.loc[2, "COL1"])  # NA
        assert pd.isna(result.loc[3, "COL1"])  # REFUSED
        assert pd.isna(result.loc[4, "COL1"])  # IAP
        
        # Check that valid value was preserved
        assert result.loc[0, "COL1"] == "VALUE"
        
        # Check numeric column untouched
        assert list(result["COL2"]) == [1, 2, 3, 4, 5]

    def test_recode_numeric_codes_selective(self):
        """Test that numeric codes are only recoded for listed columns."""
        df = pd.DataFrame({
            "EDUC": [10, 98, 99, 15],
            "AGE": [30, 98, 99, 25],
            "INCOME": [0, 5000, 10000, 15000]  # 0 should NOT be recoded (not listed)
        })
        
        config = {
            "string_patterns": {},
            "numeric_codes": {
                "EDUC": [98, 99],
                "AGE": [98, 99],
                "INCOME": []
            }
        }
        
        result = recode_gss_na(df, config)
        
        # EDUC: 98, 99 should be NA
        assert pd.isna(result.loc[1, "EDUC"])
        assert pd.isna(result.loc[2, "EDUC"])
        assert result.loc[0, "EDUC"] == 10
        
        # AGE: 98, 99 should be NA
        assert pd.isna(result.loc[1, "AGE"])
        assert pd.isna(result.loc[2, "AGE"])
        
        # INCOME: 0 is VALID (not in numeric_codes list)
        assert result.loc[0, "INCOME"] == 0  # NOT recoded
        assert result.loc[1, "INCOME"] == 5000

    def test_case_insensitive_string_patterns(self):
        """Test that string patterns are matched case-insensitively."""
        df = pd.DataFrame({"COL": ["dk", "DK", "Dk", "valid"]})
        
        config = {
            "string_patterns": {"_all": ["DK"]},
            "numeric_codes": {}
        }
        
        result = recode_gss_na(df, config)
        
        # All variations of "dk" should be NA
        assert pd.isna(result.loc[0, "COL"])
        assert pd.isna(result.loc[1, "COL"])
        assert pd.isna(result.loc[2, "COL"])
        assert result.loc[3, "COL"] == "valid"


class TestValidateGssExtract:
    def test_validate_missing_year(self):
        """Test validation fails if YEAR column missing."""
        df = pd.DataFrame({"HAPPY": ["VERY HAPPY"], "WTSSPS": [1.0]})
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "YEAR" in msg

    def test_validate_missing_happy(self):
        """Test validation fails if HAPPY column missing."""
        df = pd.DataFrame({"YEAR": [2020], "WTSSPS": [1.0]})
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "HAPPY" in msg

    def test_validate_missing_wtssps(self):
        """Test validation fails if WTSSPS weight column missing."""
        df = pd.DataFrame({"YEAR": [2020], "HAPPY": ["VERY HAPPY"]})
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "WTSSPS" in msg

    def test_validate_insufficient_rows(self):
        """Test validation fails if fewer than min_rows cases."""
        df = pd.DataFrame({
            "YEAR": [2020] * 100,
            "HAPPY": ["VERY HAPPY"] * 100,
            "WTSSPS": [1.0] * 100
        })
        is_valid, msg = validate_gss_extract(df, min_rows=5000)
        assert not is_valid
        assert "Insufficient cases" in msg

    def test_validate_insufficient_year_span(self):
        """Test validation fails if year span < min_year_span."""
        df = pd.DataFrame({
            "YEAR": [2020] * 5500,
            "HAPPY": ["VERY HAPPY"] * 5500,
            "WTSSPS": [1.0] * 5500
        })
        is_valid, msg = validate_gss_extract(df, min_rows=5000, min_year_span=5)
        assert not is_valid
        assert "Year span too small" in msg

    def test_validate_year_non_integer(self):
        """Test validation fails if YEAR contains non-integers."""
        df = pd.DataFrame({
            "YEAR": [2020.5] * 5500,
            "HAPPY": ["VERY HAPPY"] * 5500,
            "WTSSPS": [1.0] * 5500
        })
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "non-integer" in msg

    def test_validate_wtssps_not_numeric(self):
        """Test validation fails if WTSSPS is not numeric."""
        df = pd.DataFrame({
            "YEAR": [2020] * 5500,
            "HAPPY": ["VERY HAPPY"] * 5500,
            "WTSSPS": ["weight"] * 5500
        })
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "must be numeric" in msg

    def test_validate_wtssps_not_positive(self):
        """Test validation fails if WTSSPS has no positive values."""
        df = pd.DataFrame({
            "YEAR": [2020] * 5500,
            "HAPPY": ["VERY HAPPY"] * 5500,
            "WTSSPS": [0.0] * 5500
        })
        is_valid, msg = validate_gss_extract(df)
        assert not is_valid
        assert "positive values" in msg

    def test_validate_valid_extract(self):
        """Test validation passes for valid extract."""
        df = pd.DataFrame({
            "YEAR": list(range(2010, 2023)) * 500,  # 13 years, 6500 rows
            "HAPPY": ["VERY HAPPY"] * 6500,
            "WTSSPS": [1.0] * 6500
        })
        is_valid, msg = validate_gss_extract(df, min_rows=5000, min_year_span=5)
        assert is_valid
        assert "Valid" in msg


class TestPreprocessGss:
    def test_preprocess_normalize_columns(self):
        """Test column names normalized to uppercase."""
        df = pd.DataFrame({"year": [2020], "happy": ["VERY HAPPY"], "wtssps": [1.0]})
        result = preprocess_gss(df, validate=False, cast_year_int=False)
        assert "YEAR" in result.columns
        assert "HAPPY" in result.columns
        assert "WTSSPS" in result.columns

    def test_preprocess_cast_year_int(self):
        """Test YEAR is cast to int."""
        df = pd.DataFrame({
            "YEAR": list(range(2010, 2023)) * 500,  # 13 years, 6500 rows
            "HAPPY": ["VERY HAPPY"] * 6500,
            "WTSSPS": [1.0] * 6500
        })
        result = preprocess_gss(df, validate=True, cast_year_int=True)
        # After casting to Int64 (nullable), should still work
        assert result["YEAR"].dtype in ["Int64", "int64"]

    def test_preprocess_validation_fails(self):
        """Test validation error raised when validation=True."""
        df = pd.DataFrame({
            "YEAR": [2020] * 100,  # Too few rows
            "HAPPY": ["VERY HAPPY"] * 100,
            "WTSSPS": [1.0] * 100
        })
        with pytest.raises(ValueError, match="GSS extract validation failed"):
            preprocess_gss(df, validate=True, min_rows=5000)

    def test_preprocess_full_pipeline(self):
        """Test full preprocessing pipeline."""
        df = pd.DataFrame({
            "year": list(range(2010, 2023)) * 500,
            "happy": ["VERY HAPPY"] * 6500,
            "wtssps": [1.0] * 6500,
            "educ": [12, 98, 15, 99, 16] * 1300,
        })
        
        config = {
            "string_patterns": {"_all": []},
            "numeric_codes": {"EDUC": [98, 99]}
        }
        
        result = preprocess_gss(df, validate=True, cast_year_int=True, config=config)
        
        # Check column names normalized
        assert "YEAR" in result.columns
        
        # Check YEAR cast to int
        assert result["YEAR"].dtype in ["Int64", "int64"]
        
        # Check numeric codes recoded
        na_count = result["EDUC"].isna().sum()
        assert na_count > 0  # Some 98, 99 should be NA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
