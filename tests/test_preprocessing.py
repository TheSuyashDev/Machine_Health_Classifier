"""
test_preprocessing.py
=====================
Unit tests for src/data_preprocessing.py.

Run:
    pytest tests/test_preprocessing.py -v

Tests cover:
    - load_csv: valid file, missing file, empty file
    - validate_schema: correct schema, missing column, extra column
    - validate_categories: valid values, invalid nominal, invalid ordinal, invalid target
    - coerce_types: string to float, string to int, date parsing
    - validate_binary: valid 0/1, invalid value
    - validate_numeric_ranges: in-range, out-of-range detection
    - clamp_numeric: clamping below min, above max
    - encode_ordinal: correct normalized ranks, NaN passthrough
    - analyze_missing: counts missing values correctly
    - drop_duplicates: removes same machine_id + record_date
    - chronological_split: correct sizes and ordering
    - build_feature_config: correct structure and keys
    - load_and_preprocess: integration test on real CSV

Each test uses minimal self-contained DataFrames rather than loading the full
CSV wherever possible, keeping tests fast and independent.
"""

import os
import sys
import math
import tempfile
import pytest
import pandas as pd
import numpy as np

# Add src/ to path so we can import data_preprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import (
    load_csv,
    validate_schema,
    validate_categories,
    coerce_types,
    validate_binary,
    validate_numeric_ranges,
    clamp_numeric,
    encode_ordinal,
    analyze_missing,
    drop_duplicates,
    chronological_split,
    build_feature_config,
    load_and_preprocess,
    FEATURE_COLS,
    TARGET_COL,
    ORDINAL_ORDERS,
    NUMERIC_RANGES,
    ALL_EXPECTED_COLS,
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS: Minimal valid record and minimal DataFrame builders
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_record(overrides: dict = None) -> dict:
    """Return a valid, complete record dict (string-typed, as loaded from CSV)."""
    rec = {
        "machine_id":             "CNC-001",
        "record_date":            "2023-06-15",
        "machine_type":           "Milling",
        "controller_type":        "Siemens",
        "cooling_system":         "Liquid",
        "lubrication_alert":      "0",
        "abnormal_noise":         "0",
        "coolant_failure":        "0",
        "maintenance_overdue":    "0",
        "wear_level":             "Low",
        "maintenance_condition":  "Good",
        "degradation_level":      "Minimal",
        "operating_temp_c":       "55.0",
        "vibration_mm_s":         "3.5",
        "spindle_speed_rpm":      "5000",
        "operating_hours":        "12000",
        "power_consumption_kw":   "8.0",
        "tool_usage_hours":       "100",
        "avg_load_pct":           "60.0",
        "coolant_temp_c":         "35.0",
        "last_maintenance_days":  "90",
        "condition":              "Normal",
    }
    if overrides:
        rec.update(overrides)
    return rec


def _make_df(records: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from a list of record dicts (string-typed)."""
    return pd.DataFrame(records)


def _csv_path() -> str:
    """Return absolute path to the real dataset CSV."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "cnc_machines.csv")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: load_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadCsv:
    def test_load_real_csv(self):
        """The real dataset CSV should load without errors."""
        df = load_csv(_csv_path())
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0, "Dataset should not be empty."

    def test_missing_file_raises(self):
        """A non-existent file path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/path/data.csv")

    def test_empty_file_raises(self):
        """An empty CSV file should raise an error (ValueError or pandas parse error)."""
        with tempfile.NamedTemporaryFile(
            suffix=".csv", mode="w", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name
        try:
            with pytest.raises((ValueError, Exception)):
                load_csv(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_all_columns_present(self):
        """The loaded CSV should contain all expected column names."""
        df = load_csv(_csv_path())
        for col in ALL_EXPECTED_COLS:
            assert col in df.columns, f"Expected column '{col}' not found."


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: validate_schema
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_valid_schema_no_errors(self):
        """A DataFrame with all expected columns should produce no errors."""
        df = _make_df([_minimal_record()])
        errors = validate_schema(df)
        assert errors == [], f"Expected no schema errors, got: {errors}"

    def test_missing_column_detected(self):
        """Removing a required column should surface as a schema error."""
        rec = _minimal_record()
        del rec["wear_level"]      # Remove an ordinal feature column
        df = _make_df([rec])
        errors = validate_schema(df)
        assert len(errors) > 0
        assert "wear_level" in errors[0]

    def test_extra_column_is_warned_not_error(self, recwarn):
        """Extra columns should emit a warning but not an error."""
        rec = _minimal_record()
        rec["mystery_column"] = "42"
        df = _make_df([rec])
        errors = validate_schema(df)
        assert errors == [], "Extra columns should not cause schema errors."


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: validate_categories
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateCategories:
    def test_valid_categories_no_issues(self):
        """A record with valid category values should have no issues."""
        df = _make_df([_minimal_record()])
        issues, df_out = validate_categories(df)
        assert len(df_out) == 1, "Valid record should not be dropped."
        cat_issues = [i for i in issues if "invalid" in i.lower()]
        assert cat_issues == []

    def test_invalid_nominal_detected_and_dropped(self):
        """An invalid machine_type value should be detected and row dropped."""
        rec = _minimal_record({"machine_type": "Grinding"})   # Not in valid set
        df = _make_df([rec])
        issues, df_out = validate_categories(df)
        assert len(df_out) == 0, "Row with invalid nominal value should be dropped."
        assert any("machine_type" in i for i in issues)

    def test_invalid_ordinal_detected(self):
        """An invalid wear_level value should be detected and row dropped."""
        rec = _minimal_record({"wear_level": "Extreme"})   # Not a valid ordinal
        df = _make_df([rec])
        issues, df_out = validate_categories(df)
        assert len(df_out) == 0
        assert any("wear_level" in i for i in issues)

    def test_invalid_target_detected(self):
        """An invalid condition value should be detected and row dropped."""
        rec = _minimal_record({"condition": "Unknown"})
        df = _make_df([rec])
        issues, df_out = validate_categories(df)
        assert len(df_out) == 0
        assert any("condition" in i for i in issues)

    def test_valid_records_preserved_alongside_invalid(self):
        """Valid rows should survive even when the same DataFrame has invalid rows."""
        valid   = _minimal_record()
        invalid = _minimal_record({"machine_type": "Grinding", "machine_id": "CNC-002"})
        df = _make_df([valid, invalid])
        _, df_out = validate_categories(df)
        assert len(df_out) == 1
        assert df_out.iloc[0]["machine_id"] == "CNC-001"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: coerce_types
# ─────────────────────────────────────────────────────────────────────────────

class TestCoerceTypes:
    def _prep(self, overrides=None):
        df = _make_df([_minimal_record(overrides)])
        return coerce_types(df)

    def test_numeric_column_is_float(self):
        df = self._prep()
        assert pd.api.types.is_float_dtype(df["operating_temp_c"])

    def test_binary_column_is_numeric(self):
        df = self._prep()
        assert pd.api.types.is_numeric_dtype(df["lubrication_alert"])

    def test_date_column_parsed(self):
        df = self._prep()
        assert pd.api.types.is_datetime64_any_dtype(df["record_date"])

    def test_empty_numeric_becomes_nan(self):
        df = _make_df([_minimal_record({"operating_temp_c": ""})])
        df = coerce_types(df)
        assert pd.isna(df["operating_temp_c"].iloc[0])

    def test_non_numeric_becomes_nan(self):
        df = _make_df([_minimal_record({"vibration_mm_s": "N/A"})])
        df = coerce_types(df)
        assert pd.isna(df["vibration_mm_s"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: validate_binary
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateBinary:
    def test_valid_binary_no_issues(self):
        df = _make_df([_minimal_record()])
        df = coerce_types(df)
        issues = validate_binary(df)
        assert issues == []

    def test_invalid_binary_value_detected(self):
        """A value of 2 in a binary column should be flagged."""
        df = _make_df([_minimal_record({"lubrication_alert": "2"})])
        df = coerce_types(df)
        issues = validate_binary(df)
        assert any("lubrication_alert" in i for i in issues)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: validate_numeric_ranges and clamp_numeric
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericRangesAndClamping:
    def test_in_range_no_warnings(self):
        df = _make_df([_minimal_record()])
        df = coerce_types(df)
        warnings = validate_numeric_ranges(df)
        assert warnings == [], f"Unexpected warnings: {warnings}"

    def test_below_min_detected(self):
        # operating_temp_c minimum is 20.0; set to 5.0
        df = _make_df([_minimal_record({"operating_temp_c": "5.0"})])
        df = coerce_types(df)
        warnings = validate_numeric_ranges(df)
        assert any("operating_temp_c" in w for w in warnings)

    def test_above_max_detected(self):
        # vibration_mm_s maximum is 25.0; set to 999.0
        df = _make_df([_minimal_record({"vibration_mm_s": "999.0"})])
        df = coerce_types(df)
        warnings = validate_numeric_ranges(df)
        assert any("vibration_mm_s" in w for w in warnings)

    def test_clamp_below_min(self):
        """Values below min should be clamped to min."""
        df = _make_df([_minimal_record({"operating_temp_c": "5.0"})])
        df = coerce_types(df)
        df = clamp_numeric(df)
        assert df["operating_temp_c"].iloc[0] == pytest.approx(20.0)

    def test_clamp_above_max(self):
        """Values above max should be clamped to max."""
        df = _make_df([_minimal_record({"vibration_mm_s": "999.0"})])
        df = coerce_types(df)
        df = clamp_numeric(df)
        assert df["vibration_mm_s"].iloc[0] == pytest.approx(25.0)

    def test_nan_not_clamped(self):
        """NaN values should remain NaN after clamping."""
        df = _make_df([_minimal_record({"operating_temp_c": ""})])
        df = coerce_types(df)
        df = clamp_numeric(df)
        assert pd.isna(df["operating_temp_c"].iloc[0])


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: encode_ordinal  (most important mathematical tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeOrdinal:
    """
    Manually verified examples for normalized rank encoding.

    wear_level has 4 levels: Low=0, Medium=1, High=2, Critical=3
        Normalized rank = raw_rank / (4 - 1) = raw_rank / 3

    maintenance_condition has 3 levels: Good=0, Fair=1, Poor=2
        Normalized rank = raw_rank / (3 - 1) = raw_rank / 2

    degradation_level has 3 levels: Minimal=0, Moderate=1, Severe=2
        Normalized rank = raw_rank / 2
    """

    def _encode(self, overrides=None):
        df = _make_df([_minimal_record(overrides)])
        return encode_ordinal(df)

    def test_wear_low_rank(self):
        df = self._encode({"wear_level": "Low"})
        assert df["wear_level_rank"].iloc[0] == pytest.approx(0.0)

    def test_wear_medium_rank(self):
        df = self._encode({"wear_level": "Medium"})
        # 1 / 3 ≈ 0.3333
        assert df["wear_level_rank"].iloc[0] == pytest.approx(1/3, rel=1e-4)

    def test_wear_high_rank(self):
        df = self._encode({"wear_level": "High"})
        # 2 / 3 ≈ 0.6667
        assert df["wear_level_rank"].iloc[0] == pytest.approx(2/3, rel=1e-4)

    def test_wear_critical_rank(self):
        df = self._encode({"wear_level": "Critical"})
        assert df["wear_level_rank"].iloc[0] == pytest.approx(1.0)

    def test_maintenance_good_rank(self):
        df = self._encode({"maintenance_condition": "Good"})
        assert df["maintenance_condition_rank"].iloc[0] == pytest.approx(0.0)

    def test_maintenance_fair_rank(self):
        df = self._encode({"maintenance_condition": "Fair"})
        # 1 / 2 = 0.5
        assert df["maintenance_condition_rank"].iloc[0] == pytest.approx(0.5)

    def test_maintenance_poor_rank(self):
        df = self._encode({"maintenance_condition": "Poor"})
        assert df["maintenance_condition_rank"].iloc[0] == pytest.approx(1.0)

    def test_degradation_minimal_rank(self):
        df = self._encode({"degradation_level": "Minimal"})
        assert df["degradation_level_rank"].iloc[0] == pytest.approx(0.0)

    def test_degradation_moderate_rank(self):
        df = self._encode({"degradation_level": "Moderate"})
        assert df["degradation_level_rank"].iloc[0] == pytest.approx(0.5)

    def test_degradation_severe_rank(self):
        df = self._encode({"degradation_level": "Severe"})
        assert df["degradation_level_rank"].iloc[0] == pytest.approx(1.0)

    def test_ordinal_rank_is_in_0_1_range(self):
        """All normalized ranks must be in [0.0, 1.0]."""
        records = [
            _minimal_record({"wear_level": w, "maintenance_condition": m,
                             "degradation_level": d, "machine_id": f"CNC-{i}"})
            for i, (w, m, d) in enumerate([
                ("Low",      "Good", "Minimal"),
                ("Medium",   "Fair", "Moderate"),
                ("High",     "Poor", "Severe"),
                ("Critical", "Poor", "Severe"),
            ])
        ]
        df = _make_df(records)
        df = encode_ordinal(df)
        for col in ["wear_level_rank", "maintenance_condition_rank", "degradation_level_rank"]:
            assert df[col].between(0.0, 1.0).all(), f"{col} has values outside [0, 1]"

    def test_original_string_columns_preserved(self):
        """encode_ordinal should add _rank columns but keep original strings."""
        df = self._encode()
        assert "wear_level" in df.columns, "Original 'wear_level' column must be preserved."
        assert "wear_level_rank" in df.columns, "'wear_level_rank' column must be added."


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: analyze_missing
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyzeMissing:
    def test_no_missing_returns_empty_df(self):
        df = _make_df([_minimal_record()])
        df = coerce_types(df)
        df = encode_ordinal(df)
        summary = analyze_missing(df)
        assert summary.empty, "No missing values should produce empty summary."

    def test_missing_counted_correctly(self):
        """Manually introduce 1 NaN in operating_temp_c and verify count."""
        df = _make_df([
            _minimal_record({"operating_temp_c": ""}),   # NaN
            _minimal_record({"machine_id": "CNC-002"}),  # Valid
        ])
        df = coerce_types(df)
        df = encode_ordinal(df)
        summary = analyze_missing(df)
        row = summary[summary["column"] == "operating_temp_c"]
        assert len(row) == 1
        assert row.iloc[0]["missing_count"] == 1
        assert row.iloc[0]["missing_pct"] == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: drop_duplicates
# ─────────────────────────────────────────────────────────────────────────────

class TestDropDuplicates:
    def test_no_duplicates_unchanged(self):
        df = _make_df([
            _minimal_record({"machine_id": "CNC-001", "record_date": "2023-01-01"}),
            _minimal_record({"machine_id": "CNC-002", "record_date": "2023-01-01"}),
        ])
        df_out, n_dupes = drop_duplicates(df)
        assert n_dupes == 0
        assert len(df_out) == 2

    def test_duplicate_dropped(self):
        """Two rows with the same machine_id and record_date → keep first, drop second."""
        df = _make_df([
            _minimal_record({"machine_id": "CNC-001", "record_date": "2023-01-01"}),
            _minimal_record({"machine_id": "CNC-001", "record_date": "2023-01-01"}),
        ])
        df_out, n_dupes = drop_duplicates(df)
        assert n_dupes == 1
        assert len(df_out) == 1

    def test_same_machine_different_date_kept(self):
        """Same machine_id but different dates → both kept (different observations)."""
        df = _make_df([
            _minimal_record({"machine_id": "CNC-001", "record_date": "2023-01-01"}),
            _minimal_record({"machine_id": "CNC-001", "record_date": "2023-06-01"}),
        ])
        df_out, n_dupes = drop_duplicates(df)
        assert n_dupes == 0
        assert len(df_out) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: chronological_split
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalSplit:
    def _make_dated_df(self, n: int) -> pd.DataFrame:
        """Create n records with sequential dates (2020-01-01, 2020-01-02, ...)."""
        from datetime import date, timedelta
        records = []
        for i in range(n):
            rec = _minimal_record({
                "machine_id":  f"CNC-{i+1:03d}",
                "record_date": (date(2020, 1, 1) + timedelta(days=i)).isoformat(),
            })
            records.append(rec)
        df = _make_df(records)
        return coerce_types(df)

    def test_correct_split_sizes(self):
        df = self._make_dated_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert len(train) == 70
        assert len(val)   == 15
        assert len(test)  == 15

    def test_no_overlap_between_splits(self):
        df = self._make_dated_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        train_ids = set(train["machine_id"])
        val_ids   = set(val["machine_id"])
        test_ids  = set(test["machine_id"])
        assert train_ids.isdisjoint(val_ids), "Train and val should not overlap."
        assert train_ids.isdisjoint(test_ids), "Train and test should not overlap."
        assert val_ids.isdisjoint(test_ids), "Val and test should not overlap."

    def test_chronological_ordering(self):
        """Training dates should all be before validation dates, which are before test dates."""
        df = self._make_dated_df(100)
        train, val, test = chronological_split(df, 0.70, 0.15)
        assert train["record_date"].max() <= val["record_date"].min(), \
            "Latest training date should be at or before earliest validation date."
        assert val["record_date"].max() <= test["record_date"].min(), \
            "Latest validation date should be at or before earliest test date."

    def test_invalid_ratio_raises(self):
        df = self._make_dated_df(10)
        with pytest.raises(ValueError):
            chronological_split(df, 0.90, 0.20)   # Sum > 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: build_feature_config
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatureConfig:
    def setup_method(self):
        self.cfg = build_feature_config()

    def test_required_keys_present(self):
        for key in ["nominal", "binary", "ordinal", "numeric",
                    "ordinal_orders", "numeric_ranges", "binary_semantics", "weights"]:
            assert key in self.cfg, f"Key '{key}' missing from feature_config."

    def test_nominal_count(self):
        assert len(self.cfg["nominal"]) == 3

    def test_binary_count(self):
        assert len(self.cfg["binary"]) == 4

    def test_ordinal_count(self):
        assert len(self.cfg["ordinal"]) == 3

    def test_numeric_count(self):
        assert len(self.cfg["numeric"]) == 9

    def test_default_weights_are_one(self):
        for col, w in self.cfg["weights"].items():
            assert w == 1.0, f"Default weight for '{col}' should be 1.0, got {w}"

    def test_ordinal_orders_structure(self):
        for col, order in self.cfg["ordinal_orders"].items():
            assert isinstance(order, list), f"Order for '{col}' should be a list."
            assert len(order) >= 2, f"Ordinal '{col}' needs at least 2 levels."

    def test_numeric_ranges_are_positive_interval(self):
        for col, (lo, hi) in self.cfg["numeric_ranges"].items():
            assert lo < hi, f"Range for '{col}' must have lo < hi."


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: load_and_preprocess (integration test)
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAndPreprocess:
    def setup_method(self):
        self.result = load_and_preprocess(_csv_path(), verbose=False)

    def test_returns_all_keys(self):
        for key in ["train", "val", "test", "feature_config", "issues", "missing_summary"]:
            assert key in self.result

    def test_no_data_leakage_target_not_in_features(self):
        """The target column must not be a feature used for similarity."""
        cfg = self.result["feature_config"]
        all_features = (
            cfg["nominal"] + cfg["binary"] + cfg["ordinal"] + cfg["numeric"]
        )
        assert TARGET_COL not in all_features, \
            "Target column 'condition' must never be a similarity feature."

    def test_train_larger_than_val_and_test(self):
        train = self.result["train"]
        val   = self.result["val"]
        test  = self.result["test"]
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_ordinal_rank_columns_present(self):
        df = self.result["train"]
        for col in ORDINAL_ORDERS:
            assert f"{col}_rank" in df.columns, \
                f"Missing rank column: {col}_rank"

    def test_ordinal_ranks_in_valid_range(self):
        df = self.result["train"]
        for col in ORDINAL_ORDERS:
            rank_col = f"{col}_rank"
            non_null = df[rank_col].dropna()
            assert (non_null >= 0.0).all() and (non_null <= 1.0).all(), \
                f"{rank_col} has values outside [0, 1]"

    def test_condition_column_present_in_splits(self):
        for split_name in ["train", "val", "test"]:
            df = self.result[split_name]
            assert TARGET_COL in df.columns, \
                f"Target column missing from {split_name} set."

    def test_no_overlap_between_train_and_test(self):
        train_ids = set(self.result["train"]["machine_id"])
        test_ids  = set(self.result["test"]["machine_id"])
        # Some machine IDs may appear in multiple splits (different dates)
        # but each individual row (machine_id + date) should be in one split
        train_keys = set(
            zip(self.result["train"]["machine_id"],
                self.result["train"]["record_date"].astype(str))
        )
        test_keys = set(
            zip(self.result["test"]["machine_id"],
                self.result["test"]["record_date"].astype(str))
        )
        assert train_keys.isdisjoint(test_keys), \
            "Individual rows (machine_id + date) must not appear in both train and test."
