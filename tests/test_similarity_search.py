"""
test_similarity_search.py
=========================
Unit tests for src/similarity_search.py.

Run:
    pytest tests/test_similarity_search.py -v

Tests cover:
    - Correct ranking (highest similarity first)
    - Self-exclusion via exclude_id
    - Tie-breaking by machine_id
    - k > available machines (returns all)
    - Invalid inputs (bad k, empty reference, bad types)
    - SearchResult structure and content
    - Scores are in [0, 1]
    - format_results produces a non-empty string
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from similarity_search import search, format_results, SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_only_config() -> dict:
    """Feature config with one numeric column (operating_temp_c, R=100)."""
    return {
        "nominal":  [],
        "binary":   [],
        "ordinal":  [],
        "numeric":  ["operating_temp_c"],
        "binary_semantics": {},
        "numeric_ranges":   {"operating_temp_c": (20.0, 120.0)},
        "weights":  {"operating_temp_c": 1.0},
    }


def _make_reference_df(temps: list[float], conditions: list[str]) -> pd.DataFrame:
    """
    Create a minimal reference DataFrame.

    Parameters
    ----------
    temps      : operating_temp_c values for each row
    conditions : condition labels for each row

    Returns
    -------
    DataFrame with machine_id, operating_temp_c, condition
    """
    assert len(temps) == len(conditions)
    return pd.DataFrame({
        "machine_id":       [f"CNC-{i+1:03d}" for i in range(len(temps))],
        "operating_temp_c": temps,
        "condition":        conditions,
    })


@pytest.fixture
def simple_ref_df():
    """Reference DataFrame with 4 machines at known temperatures."""
    # Query will be at 55.0°C
    # CNC-001: 55.0 → s = 1.00 (identical)
    # CNC-002: 75.0 → s = 0.80
    # CNC-003: 90.0 → s = 0.65
    # CNC-004: 120.0 → s = 0.35
    return _make_reference_df(
        temps      = [55.0, 75.0, 90.0, 120.0],
        conditions = ["Normal", "Warning", "Warning", "Critical"],
    )


@pytest.fixture
def simple_cfg():
    return _numeric_only_config()


@pytest.fixture
def query_at_55():
    return {"operating_temp_c": 55.0, "machine_id": "QUERY"}


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Core ranking behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestRanking:
    def test_top1_is_most_similar(self, query_at_55, simple_ref_df, simple_cfg):
        """The highest-similarity machine should be rank 1."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3)
        top = result.neighbors.iloc[0]
        # CNC-001 at 55.0 is identical to query → s=1.0
        assert top["similarity"] == pytest.approx(1.0, rel=1e-6)
        assert top["machine_id"] == "CNC-001"

    def test_results_sorted_descending(self, query_at_55, simple_ref_df, simple_cfg):
        """Similarity scores must be in descending order."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4)
        sims = result.neighbors["similarity"].tolist()
        assert sims == sorted(sims, reverse=True), \
            f"Results not sorted: {sims}"

    def test_rank_column_is_sequential(self, query_at_55, simple_ref_df, simple_cfg):
        """rank column should be 1, 2, 3, ... for top-k results."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4)
        ranks = result.neighbors["rank"].tolist()
        assert ranks == list(range(1, len(ranks) + 1))

    def test_k_limits_results(self, query_at_55, simple_ref_df, simple_cfg):
        """Requesting k=2 should return exactly 2 neighbors."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=2)
        assert len(result.neighbors) == 2

    def test_k_larger_than_ref_returns_all(self, query_at_55, simple_ref_df, simple_cfg):
        """If k > len(reference), return all available machines."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=100)
        assert len(result.neighbors) == len(simple_ref_df)

    def test_similarity_scores_in_0_1(self, query_at_55, simple_ref_df, simple_cfg):
        """All similarity scores must be in [0, 1]."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4)
        for sim in result.neighbors["similarity"]:
            assert 0.0 <= sim <= 1.0

    def test_correct_similarity_values(self, query_at_55, simple_ref_df, simple_cfg):
        """
        Manually verify similarity scores for known temperature pairs.

        Query: 55.0°C, R=100

        CNC-001: 55.0 → s = 1 - |55-55|/100 = 1.000
        CNC-002: 75.0 → s = 1 - |55-75|/100 = 0.800
        CNC-003: 90.0 → s = 1 - |55-90|/100 = 0.650
        CNC-004: 120.0 → s = 1 - |55-120|/100 = 0.350
        """
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4)
        sims = result.neighbors.set_index("machine_id")["similarity"]
        assert sims["CNC-001"] == pytest.approx(1.000, rel=1e-6)
        assert sims["CNC-002"] == pytest.approx(0.800, rel=1e-6)
        assert sims["CNC-003"] == pytest.approx(0.650, rel=1e-6)
        assert sims["CNC-004"] == pytest.approx(0.350, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Self-exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestSelfExclusion:
    def test_exclude_id_removes_machine(self, query_at_55, simple_ref_df, simple_cfg):
        """
        When exclude_id matches a machine in the reference, that machine
        must not appear in the results.
        """
        # CNC-001 is in the reference and identical to query
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4,
                        exclude_id="CNC-001")
        ids = result.neighbors["machine_id"].tolist()
        assert "CNC-001" not in ids, "Excluded machine must not appear in results."

    def test_n_excluded_is_correct(self, query_at_55, simple_ref_df, simple_cfg):
        """n_excluded should reflect how many rows were removed."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3,
                        exclude_id="CNC-001")
        assert result.n_excluded == 1

    def test_no_exclusion_returns_self_as_top(self, query_at_55, simple_ref_df, simple_cfg):
        """Without exclude_id, the identical machine (CNC-001) should be rank 1."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3,
                        exclude_id=None)
        top = result.neighbors.iloc[0]
        assert top["similarity"] == pytest.approx(1.0)

    def test_nonexistent_exclude_id_no_effect(self, query_at_55, simple_ref_df, simple_cfg):
        """exclude_id that doesn't exist in reference → no rows excluded."""
        result = search(query_at_55, simple_ref_df, simple_cfg, k=4,
                        exclude_id="CNC-999")
        assert result.n_excluded == 0
        assert len(result.neighbors) == 4


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Tie-breaking
# ─────────────────────────────────────────────────────────────────────────────

class TestTieBreaking:
    def test_ties_broken_by_machine_id_alphabetically(self):
        """
        Two machines at the same temperature from the query → same similarity.
        The one with the lower machine_id (alphabetically) should rank higher.
        """
        cfg = _numeric_only_config()
        # Both at 75°C → both have s = 0.80 from query at 55°C
        ref = pd.DataFrame({
            "machine_id":       ["CNC-ZZZ", "CNC-AAA"],
            "operating_temp_c": [75.0, 75.0],
            "condition":        ["Warning", "Normal"],
        })
        query = {"operating_temp_c": 55.0}
        result = search(query, ref, cfg, k=2)
        # CNC-AAA should come first (alphabetically before CNC-ZZZ)
        assert result.neighbors.iloc[0]["machine_id"] == "CNC-AAA"
        assert result.neighbors.iloc[1]["machine_id"] == "CNC-ZZZ"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: SearchResult metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchResultMetadata:
    def test_n_searched_counts_remaining_after_exclusion(
        self, query_at_55, simple_ref_df, simple_cfg
    ):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3,
                        exclude_id="CNC-001")
        assert result.n_searched == len(simple_ref_df) - 1   # 4 - 1 = 3

    def test_n_searched_without_exclusion(self, query_at_55, simple_ref_df, simple_cfg):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3)
        assert result.n_searched == len(simple_ref_df)

    def test_query_id_set_from_query_dict(self, simple_ref_df, simple_cfg):
        query = {"machine_id": "MY-MACHINE", "operating_temp_c": 55.0}
        result = search(query, simple_ref_df, simple_cfg, k=2)
        assert result.query_id == "MY-MACHINE"

    def test_query_id_defaults_to_NEW_if_missing(self, simple_ref_df, simple_cfg):
        query = {"operating_temp_c": 55.0}  # No machine_id
        result = search(query, simple_ref_df, simple_cfg, k=2)
        assert result.query_id == "NEW"

    def test_condition_column_in_neighbors(self, query_at_55, simple_ref_df, simple_cfg):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=2)
        assert "condition" in result.neighbors.columns

    def test_distance_column_is_one_minus_similarity(self, query_at_55, simple_ref_df, simple_cfg):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=3)
        for _, row in result.neighbors.iterrows():
            assert row["distance"] == pytest.approx(1.0 - row["similarity"], abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    def test_k_zero_raises(self, query_at_55, simple_ref_df, simple_cfg):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            search(query_at_55, simple_ref_df, simple_cfg, k=0)

    def test_k_negative_raises(self, query_at_55, simple_ref_df, simple_cfg):
        with pytest.raises(ValueError):
            search(query_at_55, simple_ref_df, simple_cfg, k=-1)

    def test_k_float_raises(self, query_at_55, simple_ref_df, simple_cfg):
        with pytest.raises(ValueError):
            search(query_at_55, simple_ref_df, simple_cfg, k=3.5)

    def test_empty_reference_df_raises(self, query_at_55, simple_cfg):
        with pytest.raises(ValueError, match="empty"):
            search(query_at_55, pd.DataFrame(), simple_cfg, k=3)

    def test_invalid_query_type_raises(self, simple_ref_df, simple_cfg):
        with pytest.raises(TypeError):
            search([1, 2, 3], simple_ref_df, simple_cfg, k=3)

    def test_pandas_series_query_accepted(self, simple_ref_df, simple_cfg):
        """pandas Series should be accepted as a query."""
        query = pd.Series({"operating_temp_c": 55.0, "machine_id": "QUERY"})
        result = search(query, simple_ref_df, simple_cfg, k=2)
        assert len(result.neighbors) == 2


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: format_results
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatResults:
    def test_format_returns_non_empty_string(self, query_at_55, simple_ref_df, simple_cfg):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=2)
        formatted = format_results(result)
        assert isinstance(formatted, str)
        assert len(formatted) > 50

    def test_format_contains_similarity_note(self, query_at_55, simple_ref_df, simple_cfg):
        result = search(query_at_55, simple_ref_df, simple_cfg, k=2)
        formatted = format_results(result)
        assert "probability" in formatted.lower() or "NOTE" in formatted


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Integration with real dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestRealDataIntegration:
    @pytest.fixture(autouse=True)
    def load_data(self):
        import os as _os
        sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src"))
        from data_preprocessing import load_and_preprocess
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(base, "data", "cnc_machines.csv")
        result = load_and_preprocess(path, verbose=False)
        self.train = result["train"]
        self.cfg   = result["feature_config"]

    def test_top5_results_returned(self):
        query = self.train.iloc[0]
        result = search(query, self.train, self.cfg, k=5,
                        exclude_id=query["machine_id"])
        assert len(result.neighbors) == 5

    def test_self_not_in_results_when_excluded(self):
        query = self.train.iloc[0]
        qid = query["machine_id"]
        result = search(query, self.train, self.cfg, k=5, exclude_id=qid)
        assert qid not in result.neighbors["machine_id"].tolist()

    def test_all_similarity_scores_valid(self):
        query = self.train.iloc[0]
        result = search(query, self.train, self.cfg, k=10,
                        exclude_id=query["machine_id"])
        for sim in result.neighbors["similarity"]:
            assert 0.0 <= sim <= 1.0
