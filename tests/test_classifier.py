"""
test_classifier.py
==================
Unit tests for src/knn_classifier.py.

Run:
    pytest tests/test_classifier.py -v

Tests cover:
    - Majority voting: correct winner, vote counts
    - Weighted voting: similarity-weighted sums, correct winner
    - Tie-breaking: by top neighbor similarity, then alphabetically
    - k larger than training set (uses all available)
    - k=1 (nearest neighbour only)
    - Correct three-class predictions
    - Invalid inputs: bad k, empty train, unknown strategy, no condition col
    - Self-exclusion during training-set classification
    - Vote shares sum to 1.0
    - Vote shares are NOT probabilities (documentation test)
    - PredictionResult structure has all required fields
    - Explanation string contains prediction
    - Majority vs weighted give different results when neighbour distribution
      is unequal
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from knn_classifier import (
    predict,
    format_prediction,
    PredictionResult,
    _majority_vote,
    _weighted_vote,
    _resolve_winner,
    _compute_vote_shares,
    CONDITION_ORDER,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_cfg() -> dict:
    return {
        "nominal":  [],
        "binary":   [],
        "ordinal":  [],
        "numeric":  ["operating_temp_c"],
        "binary_semantics": {},
        "numeric_ranges":   {"operating_temp_c": (20.0, 120.0)},
        "weights":  {"operating_temp_c": 1.0},
    }


def _make_train_df(temps: list, conditions: list) -> pd.DataFrame:
    """Create a minimal training DataFrame with known temps and conditions."""
    assert len(temps) == len(conditions)
    return pd.DataFrame({
        "machine_id":       [f"CNC-{i+1:03d}" for i in range(len(temps))],
        "operating_temp_c": [float(t) for t in temps],
        "condition":        conditions,
    })


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _majority_vote internal helper
# ─────────────────────────────────────────────────────────────────────────────

class TestMajorityVote:
    def test_clear_majority(self):
        """3 Warning, 1 Normal, 1 Critical → Warning wins."""
        conditions  = ["Warning", "Warning", "Warning", "Normal", "Critical"]
        similarities = [0.9, 0.85, 0.8, 0.75, 0.7]
        winner, votes = _majority_vote(conditions, similarities)
        assert winner == "Warning"
        assert votes["Warning"] == pytest.approx(3.0)
        assert votes["Normal"]   == pytest.approx(1.0)
        assert votes["Critical"] == pytest.approx(1.0)

    def test_single_neighbor(self):
        """With k=1, the one neighbor's label is the prediction."""
        winner, votes = _majority_vote(["Critical"], [0.90])
        assert winner == "Critical"
        assert votes["Critical"] == pytest.approx(1.0)

    def test_all_same_label(self):
        """All k neighbors have the same label → that label wins."""
        winner, votes = _majority_vote(
            ["Normal", "Normal", "Normal"], [0.9, 0.8, 0.7]
        )
        assert winner == "Normal"
        assert votes["Normal"] == pytest.approx(3.0)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _weighted_vote internal helper
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedVote:
    def test_higher_similarity_dominates(self):
        """
        One Warning neighbor at 0.95 should outweigh two Normal neighbors
        at 0.30 each (Warning=0.95 > Normal=0.60).
        """
        conditions   = ["Warning", "Normal", "Normal"]
        similarities = [0.95, 0.30, 0.30]
        winner, sums = _weighted_vote(conditions, similarities)
        assert winner == "Warning"
        assert sums["Warning"] == pytest.approx(0.95)
        assert sums["Normal"]  == pytest.approx(0.60)

    def test_weighted_sums_correct(self):
        """
        Manual calculation:
        Warning: 0.90 + 0.80 = 1.70
        Normal:  0.75
        Critical: 0.60
        Winner: Warning
        """
        conditions   = ["Warning", "Warning", "Normal", "Critical"]
        similarities = [0.90, 0.80, 0.75, 0.60]
        winner, sums = _weighted_vote(conditions, similarities)
        assert winner == "Warning"
        assert sums["Warning"]  == pytest.approx(1.70, rel=1e-5)
        assert sums["Normal"]   == pytest.approx(0.75, rel=1e-5)
        assert sums["Critical"] == pytest.approx(0.60, rel=1e-5)

    def test_weighted_differs_from_majority_on_imbalanced(self):
        """
        Scenario where majority and weighted give different results:
        3 Normal (sims: 0.50, 0.50, 0.50) vs 2 Critical (sims: 0.95, 0.90)

        Majority: Normal wins (3 vs 2)
        Weighted: Normal=1.50, Critical=1.85 → Critical wins
        """
        conditions   = ["Normal", "Normal", "Normal", "Critical", "Critical"]
        similarities = [0.50, 0.50, 0.50, 0.95, 0.90]

        maj_winner, _ = _majority_vote(conditions, similarities)
        wgt_winner, _ = _weighted_vote(conditions, similarities)

        assert maj_winner == "Normal", "Majority: Normal should win (3 votes)"
        assert wgt_winner == "Critical", "Weighted: Critical should win (1.85 vs 1.50)"
        # This demonstrates why weighted voting can be more appropriate
        assert maj_winner != wgt_winner


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _resolve_winner tie-breaking
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveWinner:
    def test_clear_winner_no_tie(self):
        vote_totals = {"Normal": 1.0, "Warning": 3.0, "Critical": 1.0}
        conditions  = ["Warning", "Warning", "Warning", "Normal", "Critical"]
        sims        = [0.9, 0.8, 0.7, 0.6, 0.5]
        winner = _resolve_winner(vote_totals, conditions, sims)
        assert winner == "Warning"

    def test_tie_broken_by_top_similarity(self):
        """
        Warning and Critical both have 1 vote.
        Warning's top neighbor has sim=0.90.
        Critical's top neighbor has sim=0.70.
        → Warning wins the tie.
        """
        vote_totals = {"Warning": 1.0, "Critical": 1.0}
        conditions  = ["Warning", "Critical"]
        sims        = [0.90, 0.70]
        winner = _resolve_winner(vote_totals, conditions, sims)
        assert winner == "Warning"

    def test_tie_broken_alphabetically_as_last_resort(self):
        """
        When vote totals and top similarities are both equal,
        alphabetical order determines the winner.
        Critical < Normal < Warning alphabetically → Critical wins.
        """
        vote_totals = {"Warning": 1.0, "Critical": 1.0}
        conditions  = ["Warning", "Critical"]
        sims        = [0.80, 0.80]   # Equal top similarities
        winner = _resolve_winner(vote_totals, conditions, sims)
        # 'Critical' < 'Warning' alphabetically
        assert winner == "Critical"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _compute_vote_shares
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeVoteShares:
    def test_shares_sum_to_one(self):
        totals = {"Normal": 1.0, "Warning": 3.0, "Critical": 1.0}
        shares = _compute_vote_shares(totals)
        assert sum(shares.values()) == pytest.approx(1.0, rel=1e-9)

    def test_correct_proportions(self):
        totals = {"Normal": 1.0, "Warning": 4.0}
        shares = _compute_vote_shares(totals)
        assert shares["Normal"]  == pytest.approx(0.20, rel=1e-5)
        assert shares["Warning"] == pytest.approx(0.80, rel=1e-5)

    def test_zero_totals_returns_zero_shares(self):
        """Edge case: all votes are zero."""
        totals = {"Normal": 0.0, "Warning": 0.0}
        shares = _compute_vote_shares(totals)
        assert all(v == 0.0 for v in shares.values())


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: predict() — controlled scenarios
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:
    """
    Controlled classification tests with known outcomes.

    Strategy: Create a training set where the correct prediction is obvious,
    then verify the classifier makes that prediction.
    """

    def test_predict_normal_majority(self):
        """
        Query at 35°C. Training machines near 35°C are all Normal.
        → Should predict Normal.
        """
        cfg = _numeric_cfg()
        # 4 Normal near query temp, 1 Critical far away
        train = _make_train_df(
            temps      = [33.0, 34.0, 36.0, 37.0, 110.0],
            conditions = ["Normal", "Normal", "Normal", "Normal", "Critical"],
        )
        query = {"operating_temp_c": 35.0}
        result = predict(query, train, cfg, k=5, strategy="majority")
        assert result.predicted_condition == "Normal"

    def test_predict_critical_weighted(self):
        """
        Query at 115°C (near max 120°C). Close neighbors are Critical.
        → Should predict Critical.
        """
        cfg = _numeric_cfg()
        train = _make_train_df(
            temps      = [112.0, 114.0, 116.0, 30.0, 32.0],
            conditions = ["Critical", "Critical", "Critical", "Normal", "Normal"],
        )
        query = {"operating_temp_c": 115.0}
        result = predict(query, train, cfg, k=5, strategy="weighted")
        assert result.predicted_condition == "Critical"

    def test_predict_warning_three_class(self):
        """
        Query at 70°C (mid-range). Nearest neighbors: 2 Warning, 1 Normal.
        → Majority predicts Warning.
        """
        cfg = _numeric_cfg()
        train = _make_train_df(
            temps      = [68.0, 72.0, 65.0, 35.0, 110.0],
            conditions = ["Warning", "Warning", "Normal", "Normal", "Critical"],
        )
        query = {"operating_temp_c": 70.0}
        result = predict(query, train, cfg, k=3, strategy="majority")
        assert result.predicted_condition == "Warning"

    def test_k1_predicts_nearest_neighbor_condition(self):
        """
        k=1: prediction must equal the single nearest neighbor's condition.
        """
        cfg = _numeric_cfg()
        train = _make_train_df(
            temps      = [55.0, 90.0, 100.0],
            conditions = ["Normal", "Warning", "Critical"],
        )
        query = {"operating_temp_c": 56.0}   # Closest to 55.0 → Normal
        result = predict(query, train, cfg, k=1, strategy="majority")
        assert result.predicted_condition == "Normal"
        assert result.k_used == 1

    def test_self_exclusion_does_not_bias_result(self):
        """
        When classifying a machine in the training set with exclude_id,
        its own label should not influence the prediction.
        """
        cfg = _numeric_cfg()
        # Machine CNC-001 at 35°C is Critical, but all its neighbors are Normal
        train = pd.DataFrame({
            "machine_id":       ["CNC-001", "CNC-002", "CNC-003", "CNC-004"],
            "operating_temp_c": [35.0,       36.0,       37.0,       38.0],
            "condition":        ["Critical",  "Normal",   "Normal",   "Normal"],
        })
        query = train.iloc[0]  # CNC-001
        result = predict(
            query      = query,
            train_df   = train,
            feature_config = cfg,
            k          = 3,
            strategy   = "majority",
            exclude_id = "CNC-001",
        )
        # Without exclusion, CNC-001 (Critical) would trivially match itself
        # With exclusion, the 3 Normal neighbors should dominate
        assert result.predicted_condition == "Normal"

    def test_k_requested_stored_in_result(self):
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0, 70.0], ["Normal", "Warning", "Critical"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=3, strategy="majority")
        assert result.k_requested == 3

    def test_k_larger_than_train_uses_all(self):
        """k=100 with only 3 training machines → k_used=3."""
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0, 70.0], ["Normal", "Warning", "Critical"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=100, strategy="majority")
        assert result.k_used == 3

    def test_vote_shares_sum_to_one(self):
        """Vote shares must always sum to 1.0."""
        cfg = _numeric_cfg()
        train = _make_train_df(
            [50.0, 60.0, 70.0, 80.0, 90.0],
            ["Normal", "Warning", "Warning", "Critical", "Critical"]
        )
        query = {"operating_temp_c": 65.0}
        result = predict(query, train, cfg, k=5, strategy="weighted")
        total = sum(result.vote_shares.values())
        assert total == pytest.approx(1.0, rel=1e-9)

    def test_prediction_result_has_all_fields(self):
        """PredictionResult must contain all required fields."""
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0, 70.0], ["Normal", "Warning", "Critical"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=3, strategy="weighted")

        assert hasattr(result, "predicted_condition")
        assert hasattr(result, "vote_distribution")
        assert hasattr(result, "vote_shares")
        assert hasattr(result, "neighbors")
        assert hasattr(result, "k_used")
        assert hasattr(result, "k_requested")
        assert hasattr(result, "voting_strategy")
        assert hasattr(result, "explanation")
        assert hasattr(result, "search_result")

    def test_explanation_contains_prediction(self):
        """The explanation string must mention the predicted condition."""
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0, 70.0], ["Normal", "Warning", "Warning"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=3, strategy="majority")
        assert result.predicted_condition in result.explanation

    def test_voting_strategy_stored(self):
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0], ["Normal", "Warning"])
        query = {"operating_temp_c": 55.0}

        r_maj = predict(query, train, cfg, k=2, strategy="majority")
        r_wgt = predict(query, train, cfg, k=2, strategy="weighted")

        assert r_maj.voting_strategy == "majority"
        assert r_wgt.voting_strategy == "weighted"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Input validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictValidation:
    def _minimal_train(self):
        return _make_train_df([50.0, 60.0], ["Normal", "Warning"])

    def _cfg(self):
        return _numeric_cfg()

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            predict({"operating_temp_c": 55.0}, self._minimal_train(), self._cfg(), k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError):
            predict({"operating_temp_c": 55.0}, self._minimal_train(), self._cfg(), k=-3)

    def test_empty_train_raises(self):
        with pytest.raises(ValueError, match="empty"):
            predict({"operating_temp_c": 55.0}, pd.DataFrame(), self._cfg(), k=3)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="voting_strategy"):
            predict({"operating_temp_c": 55.0}, self._minimal_train(), self._cfg(),
                    k=2, strategy="random")

    def test_missing_condition_column_raises(self):
        train_no_label = pd.DataFrame({
            "machine_id":       ["CNC-001"],
            "operating_temp_c": [50.0],
            # No 'condition' column!
        })
        with pytest.raises(ValueError, match="condition"):
            predict({"operating_temp_c": 55.0}, train_no_label, self._cfg(), k=1)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: format_prediction
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatPrediction:
    def test_format_returns_non_empty_string(self):
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0, 70.0], ["Normal", "Warning", "Critical"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=3)
        formatted = format_prediction(result)
        assert isinstance(formatted, str)
        assert len(formatted) > 50

    def test_format_contains_predicted_condition(self):
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 51.0, 52.0], ["Normal", "Normal", "Normal"])
        query = {"operating_temp_c": 51.0}
        result = predict(query, train, cfg, k=3)
        formatted = format_prediction(result)
        assert result.predicted_condition in formatted

    def test_format_contains_probability_disclaimer(self):
        """The formatted output must include a note that vote shares != probabilities."""
        cfg = _numeric_cfg()
        train = _make_train_df([50.0, 60.0], ["Normal", "Warning"])
        query = {"operating_temp_c": 55.0}
        result = predict(query, train, cfg, k=2)
        formatted = format_prediction(result)
        assert "probab" in formatted.lower()


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Integration with real dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestRealDataIntegration:
    @pytest.fixture(autouse=True)
    def load_data(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from data_preprocessing import load_and_preprocess
        import os as _os
        base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(base, "data", "cnc_machines.csv")
        pp = load_and_preprocess(path, verbose=False)
        self.train = pp["train"]
        self.cfg   = pp["feature_config"]

    def test_prediction_is_valid_condition(self):
        """Predicted condition must be one of the three valid labels."""
        query = self.train.iloc[0]
        result = predict(query, self.train, self.cfg, k=5,
                         strategy="weighted", exclude_id=query["machine_id"])
        assert result.predicted_condition in CONDITION_ORDER

    def test_classify_first_10_rows_no_crash(self):
        """Running predictions on 10 machines should complete without error."""
        for i in range(10):
            row = self.train.iloc[i]
            result = predict(row, self.train, self.cfg, k=5,
                             strategy="weighted", exclude_id=row["machine_id"])
            assert result.predicted_condition in CONDITION_ORDER

    def test_vote_shares_always_sum_to_one(self):
        """Vote shares must sum to ~1.0 for any prediction."""
        for i in range(5):
            row = self.train.iloc[i]
            result = predict(row, self.train, self.cfg, k=5,
                             strategy="weighted", exclude_id=row["machine_id"])
            total = sum(result.vote_shares.values())
            assert total == pytest.approx(1.0, abs=1e-9), \
                f"Vote shares sum to {total} for row {i}"

    def test_both_strategies_produce_valid_results(self):
        """Both majority and weighted voting should produce valid condition labels."""
        query = self.train.iloc[0]
        qid   = query["machine_id"]

        r_maj = predict(query, self.train, self.cfg, k=5,
                        strategy="majority", exclude_id=qid)
        r_wgt = predict(query, self.train, self.cfg, k=5,
                        strategy="weighted", exclude_id=qid)

        assert r_maj.predicted_condition in CONDITION_ORDER
        assert r_wgt.predicted_condition in CONDITION_ORDER
