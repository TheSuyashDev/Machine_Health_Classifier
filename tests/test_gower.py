"""
test_gower.py
=============
Comprehensive unit tests for src/gower_similarity.py.

Run:
    pytest tests/test_gower.py -v

Tests cover every requirement from the problem statement:

    [REQ-1]  Identical records → similarity = 1.0
    [REQ-2]  Completely different nominal attributes
    [REQ-3]  Equal and unequal symmetric binary
    [REQ-4]  Asymmetric binary joint-absence (0-0) handling
    [REQ-5]  Ordered ordinal comparisons
    [REQ-6]  Numeric similarity with known ranges
    [REQ-7]  Missing attributes (NaN, None)
    [REQ-8]  Zero-range numeric attributes (error case)
    [REQ-9]  Custom feature weights
    [REQ-10] Invalid input values (non-numeric, unknown categories)
    [REQ-11] Similarity always in [0, 1]
    [REQ-12] One complete manually-verifiable example (matches Phase 3 derivation)
    [REQ-13] Similarity matrix symmetry and diagonal
    [REQ-14] Row similarity against a DataFrame

Each test includes a comment explaining the expected value and how to verify
it by hand.
"""

import sys
import os
import math
import pytest
import numpy as np
import pandas as pd

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gower_similarity import (
    compute_similarity,
    compute_similarity_row,
    compute_similarity_matrix,
    compute_distance,
    _nominal_similarity,
    _binary_similarity,
    _ordinal_similarity,
    _numeric_similarity,
    _is_missing,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE: Minimal feature configs for isolated testing
# ─────────────────────────────────────────────────────────────────────────────

def _nominal_only_config(weight: float = 1.0) -> dict:
    """Feature config with only one nominal column: machine_type."""
    return {
        "nominal":  ["machine_type"],
        "binary":   [],
        "ordinal":  [],
        "numeric":  [],
        "binary_semantics": {},
        "numeric_ranges":   {},
        "weights":  {"machine_type": weight},
    }


def _binary_config(semantic: str = "symmetric") -> dict:
    """Feature config with one binary column."""
    return {
        "nominal":  [],
        "binary":   ["fault_flag"],
        "ordinal":  [],
        "numeric":  [],
        "binary_semantics": {"fault_flag": semantic},
        "numeric_ranges":   {},
        "weights":  {"fault_flag": 1.0},
    }


def _ordinal_config() -> dict:
    """Feature config with one ordinal column (4 levels)."""
    return {
        "nominal":  [],
        "binary":   [],
        "ordinal":  ["wear_level"],
        "numeric":  [],
        "binary_semantics": {},
        "numeric_ranges":   {},
        "weights":  {"wear_level": 1.0},
    }


def _numeric_config(lo: float = 20.0, hi: float = 120.0, weight: float = 1.0) -> dict:
    """Feature config with one numeric column: operating_temp_c."""
    return {
        "nominal":  [],
        "binary":   [],
        "ordinal":  [],
        "numeric":  ["operating_temp_c"],
        "binary_semantics": {},
        "numeric_ranges":   {"operating_temp_c": (lo, hi)},
        "weights":  {"operating_temp_c": weight},
    }


def _full_mini_config() -> dict:
    """
    The full 5-feature mini config used in the Phase 3 worked example.

    Features:
        machine_type      : nominal
        lubrication_alert : binary symmetric
        wear_level        : ordinal (rank in wear_level_rank)
        operating_temp_c  : numeric, R=100
        vibration_mm_s    : numeric, R=24.9
    """
    return {
        "nominal":  ["machine_type"],
        "binary":   ["lubrication_alert"],
        "ordinal":  ["wear_level"],
        "numeric":  ["operating_temp_c", "vibration_mm_s"],
        "binary_semantics": {"lubrication_alert": "symmetric"},
        "numeric_ranges": {
            "operating_temp_c": (20.0, 120.0),
            "vibration_mm_s":   (0.1,  25.0),
        },
        "weights": {
            "machine_type":      1.0,
            "lubrication_alert": 1.0,
            "wear_level":        1.0,
            "operating_temp_c":  1.0,
            "vibration_mm_s":    1.0,
        },
    }


def _machine_A() -> dict:
    return {
        "machine_type":       "Milling",
        "lubrication_alert":  0,
        "wear_level":         "Low",
        "wear_level_rank":    0.0,
        "operating_temp_c":   55.0,
        "vibration_mm_s":     3.5,
    }


def _machine_B() -> dict:
    return {
        "machine_type":       "Milling",
        "lubrication_alert":  1,
        "wear_level":         "High",
        "wear_level_rank":    2/3,
        "operating_temp_c":   75.0,
        "vibration_mm_s":     8.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _is_missing helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsMissing:
    def test_none_is_missing(self):
        assert _is_missing(None) is True

    def test_float_nan_is_missing(self):
        assert _is_missing(float("nan")) is True

    def test_numpy_nan_is_missing(self):
        assert _is_missing(np.nan) is True

    def test_empty_string_is_missing(self):
        assert _is_missing("") is True
        assert _is_missing("   ") is True

    def test_zero_is_not_missing(self):
        assert _is_missing(0) is False

    def test_zero_float_is_not_missing(self):
        assert _is_missing(0.0) is False

    def test_string_value_is_not_missing(self):
        assert _is_missing("Milling") is False

    def test_integer_is_not_missing(self):
        assert _is_missing(42) is False


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _nominal_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestNominalSimilarity:
    def test_same_value_returns_one(self):
        """Identical nominal values → s = 1.0, delta = 1.0"""
        s, delta = _nominal_similarity("Milling", "Milling")
        assert s == pytest.approx(1.0)
        assert delta == pytest.approx(1.0)

    def test_different_value_returns_zero(self):
        """Different nominal values → s = 0.0, delta = 1.0"""
        s, delta = _nominal_similarity("Milling", "Turning")
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_both_missing_returns_zero_delta(self):
        """Both missing → delta = 0 → excluded from calculation"""
        s, delta = _nominal_similarity(None, None)
        assert delta == pytest.approx(0.0)

    def test_one_missing_returns_zero_delta(self):
        """One missing → delta = 0"""
        s, delta = _nominal_similarity("Milling", None)
        assert delta == pytest.approx(0.0)

        s, delta = _nominal_similarity(None, "Drilling")
        assert delta == pytest.approx(0.0)

    def test_numeric_strings_compared_as_strings(self):
        """Nominal comparison is always string equality."""
        s, delta = _nominal_similarity("1", "1")
        assert s == pytest.approx(1.0)

        s, delta = _nominal_similarity("1", "2")
        assert s == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _binary_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestBinarySimilarity:
    # ── Symmetric tests ───────────────────────────────────────────────────────

    def test_symmetric_both_zero_match(self):
        """Symmetric: 0-0 → s=1, delta=1 (shared absence is a match)"""
        s, delta = _binary_similarity(0, 0, "symmetric")
        assert s == pytest.approx(1.0)
        assert delta == pytest.approx(1.0)

    def test_symmetric_both_one_match(self):
        """Symmetric: 1-1 → s=1, delta=1"""
        s, delta = _binary_similarity(1, 1, "symmetric")
        assert s == pytest.approx(1.0)
        assert delta == pytest.approx(1.0)

    def test_symmetric_zero_one_mismatch(self):
        """Symmetric: 0-1 → s=0, delta=1"""
        s, delta = _binary_similarity(0, 1, "symmetric")
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_symmetric_one_zero_mismatch(self):
        """Symmetric: 1-0 → s=0, delta=1"""
        s, delta = _binary_similarity(1, 0, "symmetric")
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    # ── Asymmetric tests ──────────────────────────────────────────────────────

    def test_asymmetric_both_zero_excluded(self):
        """
        [REQ-4] Asymmetric: 0-0 → delta=0 (shared absence excluded).

        In original Gower (1971), for asymmetric binary attributes, a pair
        of 0-0 values is excluded from the similarity calculation because
        the shared non-occurrence of a rare event is uninformative.

        Both the numerator contribution AND the denominator contribution are 0.
        """
        s, delta = _binary_similarity(0, 0, "asymmetric")
        assert delta == pytest.approx(0.0), \
            "Asymmetric 0-0: delta must be 0 (excluded from calculation)."

    def test_asymmetric_both_one_match(self):
        """Asymmetric: 1-1 → s=1, delta=1 (shared presence is informative)"""
        s, delta = _binary_similarity(1, 1, "asymmetric")
        assert s == pytest.approx(1.0)
        assert delta == pytest.approx(1.0)

    def test_asymmetric_zero_one_mismatch(self):
        """Asymmetric: 0-1 → s=0, delta=1"""
        s, delta = _binary_similarity(0, 1, "asymmetric")
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_asymmetric_one_zero_mismatch(self):
        """Asymmetric: 1-0 → s=0, delta=1"""
        s, delta = _binary_similarity(1, 0, "asymmetric")
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_binary_float_values_handled(self):
        """Binary columns may be stored as 0.0 and 1.0 (pandas int→float)."""
        s, delta = _binary_similarity(0.0, 0.0, "symmetric")
        assert s == pytest.approx(1.0)

    def test_binary_missing_excluded(self):
        """Missing value → delta=0."""
        s, delta = _binary_similarity(None, 1, "symmetric")
        assert delta == pytest.approx(0.0)

        s, delta = _binary_similarity(0, np.nan, "symmetric")
        assert delta == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _ordinal_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestOrdinalSimilarity:
    """
    [REQ-5] Ordinal similarity tests.

    wear_level has 4 levels: Low=0/3=0.000, Medium=1/3≈0.333,
                              High=2/3≈0.667, Critical=3/3=1.000

    Formula: s = 1 - |rank_i - rank_j|
    """

    def test_identical_ordinal_ranks_similarity_one(self):
        """Same rank → |rank_i - rank_j| = 0 → s = 1.0"""
        s, delta = _ordinal_similarity(0.0, 0.0)
        assert s == pytest.approx(1.0)

    def test_adjacent_levels(self):
        """
        Low (0.000) vs Medium (0.333):
        s = 1 - |0.000 - 0.333| = 1 - 0.333 = 0.667
        """
        low_rank    = 0.0
        medium_rank = 1/3
        s, delta = _ordinal_similarity(low_rank, medium_rank)
        assert s == pytest.approx(1 - 1/3, rel=1e-5)
        assert delta == pytest.approx(1.0)

    def test_two_levels_apart(self):
        """
        Low (0.000) vs High (0.667):
        s = 1 - |0.000 - 0.667| = 1 - 0.667 = 0.333
        """
        low_rank  = 0.0
        high_rank = 2/3
        s, _ = _ordinal_similarity(low_rank, high_rank)
        assert s == pytest.approx(1 - 2/3, rel=1e-5)

    def test_opposite_ends(self):
        """
        Low (0.000) vs Critical (1.000):
        s = 1 - |0.000 - 1.000| = 0.000
        """
        s, delta = _ordinal_similarity(0.0, 1.0)
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_maintenance_good_vs_poor(self):
        """
        maintenance_condition: Good=0.0, Poor=1.0
        s = 1 - |0.0 - 1.0| = 0.0
        """
        s, _ = _ordinal_similarity(0.0, 1.0)
        assert s == pytest.approx(0.0)

    def test_maintenance_good_vs_fair(self):
        """
        maintenance_condition: Good=0.0, Fair=0.5
        s = 1 - |0.0 - 0.5| = 0.5
        """
        s, _ = _ordinal_similarity(0.0, 0.5)
        assert s == pytest.approx(0.5)

    def test_ordinal_missing_excluded(self):
        """NaN rank → delta=0."""
        s, delta = _ordinal_similarity(np.nan, 0.5)
        assert delta == pytest.approx(0.0)

    def test_ordinal_result_in_0_1(self):
        """All valid rank pairs must produce s in [0, 1]."""
        ranks = [0.0, 1/3, 0.5, 2/3, 1.0]
        for r1 in ranks:
            for r2 in ranks:
                s, delta = _ordinal_similarity(r1, r2)
                assert 0.0 <= s <= 1.0, f"s={s} out of [0,1] for ranks {r1}, {r2}"
                assert delta == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _numeric_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericSimilarity:
    """
    [REQ-6] Numeric similarity tests with known ranges.

    operating_temp_c: range R = 120 - 20 = 100
    """

    def test_identical_values_similarity_one(self):
        """Same value → |x_i - x_j| = 0 → s = 1.0"""
        s, delta = _numeric_similarity(55.0, 55.0, feature_range=100.0)
        assert s == pytest.approx(1.0)
        assert delta == pytest.approx(1.0)

    def test_opposite_ends_of_range(self):
        """
        Min=20, Max=120, R=100.
        x_i=20, x_j=120 → s = 1 - |20-120|/100 = 1 - 1.0 = 0.0
        """
        s, delta = _numeric_similarity(20.0, 120.0, feature_range=100.0)
        assert s == pytest.approx(0.0)
        assert delta == pytest.approx(1.0)

    def test_phase3_example_temperature(self):
        """
        [REQ-12] From Phase 3 worked example:
        Machine A: temp=55.0, Machine B: temp=75.0, R=100
        s = 1 - |55-75|/100 = 1 - 0.20 = 0.80
        """
        s, _ = _numeric_similarity(55.0, 75.0, feature_range=100.0)
        assert s == pytest.approx(0.80, rel=1e-6)

    def test_phase3_example_vibration(self):
        """
        [REQ-12] From Phase 3 worked example:
        Machine A: vib=3.5, Machine B: vib=8.0, R=24.9
        s = 1 - |3.5-8.0|/24.9 = 1 - 4.5/24.9 ≈ 0.8193
        """
        s, _ = _numeric_similarity(3.5, 8.0, feature_range=24.9)
        expected = 1 - 4.5 / 24.9
        assert s == pytest.approx(expected, rel=1e-5)

    def test_small_difference(self):
        """
        x_i=50.0, x_j=51.0, R=100
        s = 1 - 1/100 = 0.99
        """
        s, _ = _numeric_similarity(50.0, 51.0, feature_range=100.0)
        assert s == pytest.approx(0.99, rel=1e-6)

    def test_zero_range_raises_error(self):
        """[REQ-8] A zero range is a programming error — must raise ValueError."""
        with pytest.raises(ValueError, match="feature_range"):
            _numeric_similarity(5.0, 5.0, feature_range=0.0)

    def test_negative_range_raises_error(self):
        """A negative range is also invalid."""
        with pytest.raises(ValueError):
            _numeric_similarity(5.0, 5.0, feature_range=-10.0)

    def test_missing_value_excluded(self):
        """NaN → delta=0."""
        s, delta = _numeric_similarity(np.nan, 75.0, feature_range=100.0)
        assert delta == pytest.approx(0.0)

        s, delta = _numeric_similarity(55.0, None, feature_range=100.0)
        assert delta == pytest.approx(0.0)

    def test_result_always_in_0_1(self):
        """Similarity must never go outside [0, 1] even at boundary values."""
        s, _ = _numeric_similarity(20.0, 120.0, feature_range=100.0)
        assert 0.0 <= s <= 1.0

        s, _ = _numeric_similarity(20.0, 20.0, feature_range=100.0)
        assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: compute_similarity (overall function)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSimilarity:
    """
    [REQ-1] Identical records → similarity = 1.0
    [REQ-12] Complete manually verifiable example
    """

    def test_identical_records_similarity_one(self):
        """[REQ-1] A machine compared to itself must give S = 1.0."""
        cfg = _full_mini_config()
        A = _machine_A()
        s = compute_similarity(A, A, cfg)
        assert s == pytest.approx(1.0), \
            "Self-similarity must be exactly 1.0 for all non-missing features."

    def test_complete_manually_verified_example(self):
        """
        [REQ-12] Full Phase 3 worked example.

        Expected calculation:
            s(machine_type)    = 1.000  (Milling == Milling)
            s(lubrication)     = 0.000  (0 != 1)
            s(wear_level)      = 0.333  (|0.000 - 0.667| = 0.667)
            s(operating_temp)  = 0.800  (|55.0 - 75.0| / 100 = 0.20)
            s(vibration)       = 0.819  (|3.5 - 8.0| / 24.9 ≈ 0.181)
            Numerator   = 1.000 + 0.000 + 0.333 + 0.800 + 0.819 = 2.952
            Denominator = 5
            S(A, B) = 2.952 / 5 ≈ 0.5905
        """
        cfg = _full_mini_config()
        A = _machine_A()
        B = _machine_B()

        s_nominal   = 1.000
        s_binary    = 0.000
        s_ordinal   = 1 - 2/3              # ≈ 0.3333
        s_temp      = 1 - 20.0/100.0       # = 0.800
        s_vibration = 1 - 4.5/24.9         # ≈ 0.8193

        expected = (s_nominal + s_binary + s_ordinal + s_temp + s_vibration) / 5

        computed = compute_similarity(A, B, cfg)
        assert computed == pytest.approx(expected, rel=1e-5), \
            f"Expected {expected:.6f}, got {computed:.6f}"

    def test_symmetry_s_ij_equals_s_ji(self):
        """S(A, B) must equal S(B, A) — similarity is symmetric."""
        cfg = _full_mini_config()
        A = _machine_A()
        B = _machine_B()
        assert compute_similarity(A, B, cfg) == pytest.approx(
            compute_similarity(B, A, cfg), rel=1e-10
        )

    def test_score_in_0_1_range(self):
        """[REQ-11] Similarity must always be in [0.0, 1.0]."""
        cfg = _full_mini_config()
        A = _machine_A()
        B = _machine_B()
        s = compute_similarity(A, B, cfg)
        assert 0.0 <= s <= 1.0

    def test_completely_different_nominal(self):
        """
        [REQ-2] Two records differing only in nominal attribute.
        With only one nominal feature and completely different values → s = 0.
        """
        cfg = _nominal_only_config()
        r1 = {"machine_type": "Milling"}
        r2 = {"machine_type": "Drilling"}
        s = compute_similarity(r1, r2, cfg)
        assert s == pytest.approx(0.0)

    def test_all_features_missing_returns_zero(self):
        """
        [REQ-7] When all features are missing in both records, the
        denominator is 0 and similarity defaults to 0.0.
        """
        cfg = _numeric_config()
        r1 = {"operating_temp_c": np.nan}
        r2 = {"operating_temp_c": None}
        s = compute_similarity(r1, r2, cfg)
        assert s == pytest.approx(0.0)

    def test_one_feature_missing_uses_other_features(self):
        """
        [REQ-7] When one feature is missing, the similarity is computed
        from the remaining available features only.

        Config: 2 numeric features (temp R=100, vibration R=24.9)
        Machine A: temp=55.0, vib=3.5
        Machine B: temp=75.0, vib=NaN (missing)

        Only temp is comparable:
        s = [1 * 1.0 * 0.80] / [1 * 1.0] = 0.80
        """
        cfg = {
            "nominal":  [],
            "binary":   [],
            "ordinal":  [],
            "numeric":  ["operating_temp_c", "vibration_mm_s"],
            "binary_semantics": {},
            "numeric_ranges": {
                "operating_temp_c": (20.0, 120.0),
                "vibration_mm_s":   (0.1,  25.0),
            },
            "weights": {"operating_temp_c": 1.0, "vibration_mm_s": 1.0},
        }
        A = {"operating_temp_c": 55.0, "vibration_mm_s": 3.5}
        B = {"operating_temp_c": 75.0, "vibration_mm_s": np.nan}

        s = compute_similarity(A, B, cfg)
        expected = 1 - 20.0/100.0   # = 0.80
        assert s == pytest.approx(expected, rel=1e-6)

    def test_asymmetric_binary_joint_absence_excluded(self):
        """
        [REQ-4] Asymmetric binary 0-0 pair: delta=0, excluded from denominator.

        Config: 1 asymmetric binary + 1 numeric (temp, s=0.8)
        Machine A: fault=0, temp=55.0
        Machine B: fault=0, temp=75.0

        fault_flag 0-0 → delta=0, excluded.
        Only temp contributes: s = 0.80
        """
        cfg = {
            "nominal":  [],
            "binary":   ["fault_flag"],
            "ordinal":  [],
            "numeric":  ["operating_temp_c"],
            "binary_semantics": {"fault_flag": "asymmetric"},
            "numeric_ranges":   {"operating_temp_c": (20.0, 120.0)},
            "weights":  {"fault_flag": 1.0, "operating_temp_c": 1.0},
        }
        A = {"fault_flag": 0, "operating_temp_c": 55.0}
        B = {"fault_flag": 0, "operating_temp_c": 75.0}

        s = compute_similarity(A, B, cfg)
        expected = 1 - 20.0/100.0   # = 0.80 (only temp contributes)
        assert s == pytest.approx(expected, rel=1e-6)

    def test_pandas_series_input(self):
        """compute_similarity should accept pandas Series as input."""
        cfg = _numeric_config()
        A = pd.Series({"operating_temp_c": 55.0})
        B = pd.Series({"operating_temp_c": 75.0})
        s = compute_similarity(A, B, cfg)
        expected = 1 - 20.0/100.0
        assert s == pytest.approx(expected, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Custom Feature Weights
# ─────────────────────────────────────────────────────────────────────────────

class TestCustomWeights:
    """
    [REQ-9] Feature weights control the relative importance of each attribute.

    With 2 features (temp R=100, vibration R=24.9):
        A: temp=55, vib=3.5
        B: temp=75, vib=8.0

        s_temp      = 0.800
        s_vibration = 1 - 4.5/24.9 ≈ 0.8193

    Equal weights (1, 1):
        S = (1*0.800 + 1*0.8193) / (1 + 1) ≈ 0.8096

    Weighted temp=2, vib=1:
        S = (2*0.800 + 1*0.8193) / (2 + 1) = (1.600 + 0.8193) / 3 ≈ 0.8064

    The weights change the result.
    """

    def _two_feature_config(self, w_temp: float, w_vib: float) -> dict:
        return {
            "nominal":  [],
            "binary":   [],
            "ordinal":  [],
            "numeric":  ["operating_temp_c", "vibration_mm_s"],
            "binary_semantics": {},
            "numeric_ranges": {
                "operating_temp_c": (20.0, 120.0),
                "vibration_mm_s":   (0.1,  25.0),
            },
            "weights": {
                "operating_temp_c": w_temp,
                "vibration_mm_s":   w_vib,
            },
        }

    def _records(self):
        A = {"operating_temp_c": 55.0, "vibration_mm_s": 3.5}
        B = {"operating_temp_c": 75.0, "vibration_mm_s": 8.0}
        return A, B

    def test_equal_weights_baseline(self):
        cfg = self._two_feature_config(1.0, 1.0)
        A, B = self._records()
        s_temp = 0.80
        s_vib  = 1 - 4.5/24.9
        expected = (1.0 * s_temp + 1.0 * s_vib) / 2.0
        assert compute_similarity(A, B, cfg) == pytest.approx(expected, rel=1e-5)

    def test_zero_weight_feature_excluded(self):
        """
        [REQ-9] A feature with weight=0 contributes nothing.
        Only the non-zero-weight feature determines the score.
        """
        cfg = self._two_feature_config(1.0, 0.0)  # vibration weight = 0
        A, B = self._records()
        # Only temp contributes: s = 0.80
        assert compute_similarity(A, B, cfg) == pytest.approx(0.80, rel=1e-6)

    def test_double_weight_changes_score(self):
        """Doubling one feature's weight shifts the similarity toward that feature."""
        cfg_equal    = self._two_feature_config(1.0, 1.0)
        cfg_weighted = self._two_feature_config(2.0, 1.0)
        A, B = self._records()
        s_equal    = compute_similarity(A, B, cfg_equal)
        s_weighted = compute_similarity(A, B, cfg_weighted)
        # The scores should be different
        assert s_equal != pytest.approx(s_weighted, rel=1e-5)

    def test_symmetric_weights_give_symmetric_results(self):
        """Swapping weights between two equal-score features should not change the total."""
        s_temp = 1 - 20.0/100.0   # 0.80
        s_vib  = 1 - 4.5/24.9     # ~0.8193
        A, B = self._records()

        cfg1 = self._two_feature_config(2.0, 1.0)
        cfg2 = self._two_feature_config(1.0, 2.0)
        # These should be different because s_temp != s_vib
        assert compute_similarity(A, B, cfg1) != pytest.approx(
            compute_similarity(A, B, cfg2), rel=1e-4
        )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: compute_distance
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDistance:
    def test_identical_records_distance_zero(self):
        cfg = _numeric_config()
        r = {"operating_temp_c": 55.0}
        assert compute_distance(r, r, cfg) == pytest.approx(0.0, abs=1e-10)

    def test_distance_is_one_minus_similarity(self):
        cfg = _full_mini_config()
        A = _machine_A()
        B = _machine_B()
        s = compute_similarity(A, B, cfg)
        d = compute_distance(A, B, cfg)
        assert d == pytest.approx(1.0 - s, rel=1e-10)

    def test_distance_always_in_0_1(self):
        cfg = _full_mini_config()
        A = _machine_A()
        B = _machine_B()
        d = compute_distance(A, B, cfg)
        assert 0.0 <= d <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: compute_similarity_row
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSimilarityRow:
    def _make_df(self) -> pd.DataFrame:
        """Create a 3-row reference DataFrame."""
        records = [
            {"operating_temp_c": 55.0},   # Identical to query
            {"operating_temp_c": 75.0},   # 20 degrees different → s=0.80
            {"operating_temp_c": 120.0},  # At max → s=0.35 relative to 55
        ]
        return pd.DataFrame(records)

    def test_returns_array_of_correct_length(self):
        cfg = _numeric_config()
        query = {"operating_temp_c": 55.0}
        df = self._make_df()
        scores = compute_similarity_row(query, df, cfg)
        assert len(scores) == 3

    def test_first_row_identical_score_one(self):
        """The first row is identical to query → score = 1.0"""
        cfg = _numeric_config()
        query = {"operating_temp_c": 55.0}
        df = self._make_df()
        scores = compute_similarity_row(query, df, cfg)
        assert scores[0] == pytest.approx(1.0)

    def test_second_row_score_correct(self):
        """Query=55, row=75, R=100 → s = 1 - 20/100 = 0.80"""
        cfg = _numeric_config()
        query = {"operating_temp_c": 55.0}
        df = self._make_df()
        scores = compute_similarity_row(query, df, cfg)
        assert scores[1] == pytest.approx(0.80, rel=1e-6)

    def test_scores_in_0_1_range(self):
        cfg = _numeric_config()
        query = {"operating_temp_c": 55.0}
        scores = compute_similarity_row(query, self._make_df(), cfg)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_pandas_series_query(self):
        """compute_similarity_row should accept a pandas Series as query."""
        cfg = _numeric_config()
        query = pd.Series({"operating_temp_c": 55.0})
        scores = compute_similarity_row(query, self._make_df(), cfg)
        assert scores[0] == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: compute_similarity_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSimilarityMatrix:
    def _make_df(self) -> pd.DataFrame:
        records = [
            {"operating_temp_c": 55.0},
            {"operating_temp_c": 75.0},
            {"operating_temp_c": 55.0},   # Same as row 0
        ]
        return pd.DataFrame(records)

    def test_matrix_shape(self):
        cfg = _numeric_config()
        df = self._make_df()
        matrix = compute_similarity_matrix(df, cfg)
        assert matrix.shape == (3, 3)

    def test_diagonal_is_one(self):
        """[REQ-1] Every machine compared to itself → 1.0 on diagonal."""
        cfg = _numeric_config()
        df = self._make_df()
        matrix = compute_similarity_matrix(df, cfg)
        for i in range(3):
            assert matrix[i, i] == pytest.approx(1.0), \
                f"Diagonal element [{i},{i}] should be 1.0, got {matrix[i,i]}"

    def test_matrix_is_symmetric(self):
        """S(i,j) must equal S(j,i)."""
        cfg = _numeric_config()
        df = self._make_df()
        matrix = compute_similarity_matrix(df, cfg)
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)

    def test_identical_rows_have_similarity_one(self):
        """Rows 0 and 2 have identical temp=55 → similarity = 1.0."""
        cfg = _numeric_config()
        df = self._make_df()
        matrix = compute_similarity_matrix(df, cfg)
        assert matrix[0, 2] == pytest.approx(1.0)
        assert matrix[2, 0] == pytest.approx(1.0)

    def test_all_values_in_0_1(self):
        cfg = _numeric_config()
        df = self._make_df()
        matrix = compute_similarity_matrix(df, cfg)
        assert (matrix >= 0.0).all() and (matrix <= 1.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Integration with real feature config from data_preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithRealConfig:
    """
    Use the actual feature config from data_preprocessing.py to ensure
    the full 19-feature similarity works correctly on real data.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        """Load preprocessed data once per test class."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from data_preprocessing import load_and_preprocess
        import os as _os
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(base, "data", "cnc_machines.csv")
        result = load_and_preprocess(path, verbose=False)
        self.train   = result["train"]
        self.cfg     = result["feature_config"]

    def test_self_similarity_is_one(self):
        """[REQ-1] Every training row compared to itself → S = 1.0."""
        for i in range(min(10, len(self.train))):  # Test first 10 rows
            row = self.train.iloc[i]
            s = compute_similarity(row, row, self.cfg)
            assert s == pytest.approx(1.0, abs=1e-9), \
                f"Row {i}: self-similarity = {s}, expected 1.0"

    def test_all_pairwise_scores_in_0_1(self):
        """Check a sample of pairwise scores are all in [0, 1]."""
        subset = self.train.head(20)
        for i in range(10):
            row_i = subset.iloc[i]
            for j in range(10):
                row_j = subset.iloc[j]
                s = compute_similarity(row_i, row_j, self.cfg)
                assert 0.0 <= s <= 1.0, \
                    f"S({i},{j}) = {s} is outside [0, 1]"

    def test_similarity_decreases_with_expected_features(self):
        """
        A machine with Low wear vs Critical wear should have lower
        similarity than the same machine compared to itself.
        """
        row = self.train.iloc[0].copy()

        # Make a modified version with worst-case ordinals
        modified = row.copy()
        modified["wear_level"] = "Critical"
        modified["wear_level_rank"] = 1.0
        modified["degradation_level"] = "Severe"
        modified["degradation_level_rank"] = 1.0

        s_self     = compute_similarity(row, row, self.cfg)
        s_modified = compute_similarity(row, modified, self.cfg)

        assert s_self > s_modified, \
            "Self-similarity must be greater than similarity to a degraded version."
