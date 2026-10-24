"""
knn_classifier.py
=================
Custom k-Nearest Neighbors classifier for CNC machine condition prediction.

Uses Gower-style distance (D = 1 - similarity) to find the k most similar
training machines, then predicts the query machine's condition by vote.

Two voting strategies are supported:

    1. Majority Voting (unweighted):
       Each of the k neighbors casts one vote for its condition label.
       The label with the most votes wins.

    2. Similarity-Weighted Voting:
       Each neighbor's vote is weighted by its similarity to the query.
       A neighbor with similarity 0.95 carries more influence than one
       with similarity 0.60. This is generally preferable because it
       reduces the influence of borderline neighbors.

       weighted_vote(class C) = sum of similarities of all k neighbors
                                that belong to class C

Distance Definition
-------------------
D(i, j) = 1 - S(i, j)

where S(i, j) is the Gower similarity from gower_similarity.py.
A distance of 0 means identical; a distance of 1 means maximally different.

Tie Handling
------------
When two classes have equal vote counts (majority) or equal weighted sums
(weighted), the tiebreaker is the class whose *highest individual neighbor
similarity* is greater. If that is also tied, the class label that comes
first in alphabetical order is chosen (deterministic fallback).

Output
------
The classifier returns a PredictionResult dataclass that contains:
    - predicted_condition : the winning class label
    - vote_distribution   : dict of {class: vote_count or weighted_sum}
    - neighbors           : DataFrame from similarity_search.SearchResult
    - k_used              : actual k used (may be < requested k)
    - voting_strategy     : "majority" or "weighted"
    - explanation         : plain-language description of the prediction

IMPORTANT NOTE on Vote Shares:
    The normalized vote share (e.g., Warning: 60%) is NOT a calibrated
    probability of failure. It is simply the proportion of votes cast by
    the k nearest neighbors. A calibrated probability would require fitting
    the model to real historical failure data and applying probability
    calibration techniques, which is beyond this academic scope.

Public API
----------
    predict(query, train_df, feature_config, k, strategy, exclude_id)
        -> PredictionResult
    format_prediction(result) -> str
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from similarity_search import search, SearchResult

# Condition label ordering (worst last) — used for tie-breaking display
CONDITION_ORDER = ["Normal", "Warning", "Critical"]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Structured output from the k-NN classifier.

    Attributes
    ----------
    predicted_condition : predicted class label ("Normal", "Warning", "Critical")
    vote_distribution   : dict mapping class label → vote (count or weighted sum)
    vote_shares         : dict mapping class label → fraction of total votes [0,1]
                          NOTE: NOT a calibrated failure probability.
    neighbors           : DataFrame of the k nearest neighbors with their
                          conditions and similarity scores (from SearchResult)
    k_used              : actual number of neighbors used (may be < k if
                          fewer than k machines exist in training set)
    k_requested         : the k that was requested
    voting_strategy     : "majority" or "weighted"
    explanation         : plain-language summary of the prediction
    search_result       : the underlying SearchResult from similarity_search
    """
    predicted_condition: str
    vote_distribution:   dict[str, float]
    vote_shares:         dict[str, float]
    neighbors:           pd.DataFrame
    k_used:              int
    k_requested:         int
    voting_strategy:     str
    explanation:         str
    search_result:       SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# VOTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _majority_vote(
    conditions: list[str],
    similarities: list[float],
) -> tuple[str, dict[str, float]]:
    """
    Determine the winning class by majority voting.

    Each neighbor casts exactly one vote. The class with the most votes wins.

    Parameters
    ----------
    conditions  : condition label for each of the k neighbors
    similarities: similarity score for each neighbor (used for tie-breaking)

    Returns
    -------
    (winner, vote_counts) where vote_counts maps class → count (float for
    consistency with weighted voting)
    """
    vote_counts: dict[str, float] = {}
    for label in conditions:
        vote_counts[label] = vote_counts.get(label, 0.0) + 1.0

    winner = _resolve_winner(vote_counts, conditions, similarities)
    return winner, vote_counts


def _weighted_vote(
    conditions: list[str],
    similarities: list[float],
) -> tuple[str, dict[str, float]]:
    """
    Determine the winning class by similarity-weighted voting.

    Each neighbor's vote is weighted by its similarity score. A neighbor
    with similarity 0.95 contributes 0.95 to its class's total, while a
    neighbor with similarity 0.60 contributes only 0.60.

    Why this is better than majority voting:
        Majority voting treats a similarity-0.95 neighbor identically to a
        similarity-0.55 neighbor. Weighted voting naturally discounts
        borderline neighbors.

    Parameters
    ----------
    conditions  : condition label for each of the k neighbors
    similarities: similarity score for each neighbor [0, 1]

    Returns
    -------
    (winner, weighted_sums) where weighted_sums maps class → sum of similarities
    """
    weighted_sums: dict[str, float] = {}
    for label, sim in zip(conditions, similarities):
        weighted_sums[label] = weighted_sums.get(label, 0.0) + sim

    winner = _resolve_winner(weighted_sums, conditions, similarities)
    return winner, weighted_sums


def _resolve_winner(
    vote_totals: dict[str, float],
    conditions: list[str],
    similarities: list[float],
) -> str:
    """
    Determine the winning class from vote totals, with tie-breaking.

    Tie-breaking rules (applied in order):
    1. Class with the highest vote total wins.
    2. If tied: the class whose top (highest-similarity) neighbor has the
       greater similarity wins.
    3. If still tied: alphabetical order of class label (deterministic).

    Parameters
    ----------
    vote_totals  : dict mapping class → total votes (count or weighted sum)
    conditions   : ordered list of neighbor conditions (same order as search results)
    similarities : ordered list of neighbor similarities

    Returns
    -------
    str : the winning class label
    """
    max_votes = max(vote_totals.values())
    tied_classes = [c for c, v in vote_totals.items() if v == max_votes]

    if len(tied_classes) == 1:
        return tied_classes[0]

    # Tie-break: find the highest similarity neighbor for each tied class
    class_top_sim: dict[str, float] = {}
    for label, sim in zip(conditions, similarities):
        if label in tied_classes:
            class_top_sim[label] = max(class_top_sim.get(label, 0.0), sim)

    max_top_sim = max(class_top_sim.values())
    still_tied = [c for c, s in class_top_sim.items() if s == max_top_sim]

    if len(still_tied) == 1:
        return still_tied[0]

    # Final fallback: alphabetical order
    return sorted(still_tied)[0]


def _compute_vote_shares(vote_totals: dict[str, float]) -> dict[str, float]:
    """
    Normalise vote totals to shares summing to 1.0.

    WARNING: These shares are NOT calibrated failure probabilities.
    They are simply the relative proportions of the neighbor votes.

    Parameters
    ----------
    vote_totals : dict mapping class → raw vote total

    Returns
    -------
    dict mapping class → share in [0, 1]
    """
    total = sum(vote_totals.values())
    if total == 0:
        return {c: 0.0 for c in vote_totals}
    return {c: v / total for c, v in vote_totals.items()}


# ─────────────────────────────────────────────────────────────────────────────
# EXPLANATION BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_explanation(
    predicted_condition: str,
    vote_distribution: dict[str, float],
    vote_shares: dict[str, float],
    k_used: int,
    voting_strategy: str,
    neighbors: pd.DataFrame,
) -> str:
    """
    Build a plain-language explanation of the prediction.

    This makes the k-NN classifier interpretable — a user can understand
    *why* the prediction was made without reading the code.

    Parameters
    ----------
    predicted_condition : the predicted class
    vote_distribution   : raw vote totals
    vote_shares         : normalised vote shares
    k_used              : actual k
    voting_strategy     : "majority" or "weighted"
    neighbors           : neighbor DataFrame with similarity and condition columns

    Returns
    -------
    str : human-readable explanation
    """
    strategy_name = (
        "similarity-weighted voting" if voting_strategy == "weighted"
        else "majority voting"
    )

    # Describe the vote distribution
    vote_desc_parts = []
    for label in CONDITION_ORDER:
        if label in vote_distribution:
            raw  = vote_distribution[label]
            pct  = vote_shares.get(label, 0) * 100
            if voting_strategy == "majority":
                vote_desc_parts.append(f"{label}: {int(raw)} vote(s) ({pct:.0f}%)")
            else:
                vote_desc_parts.append(f"{label}: {raw:.3f} weight ({pct:.0f}%)")
    vote_desc = ", ".join(vote_desc_parts)

    # Describe the top neighbor
    top_neighbor_desc = ""
    if len(neighbors) > 0:
        top = neighbors.iloc[0]
        top_id   = top.get("machine_id", "N/A")
        top_sim  = top.get("similarity", 0)
        top_cond = top.get("condition", "N/A")
        top_neighbor_desc = (
            f" The closest matching machine is {top_id} "
            f"(similarity={top_sim:.4f}, condition={top_cond})."
        )

    explanation = (
        f"Prediction: {predicted_condition}. "
        f"Based on {k_used} nearest neighbor(s) using {strategy_name}. "
        f"Vote distribution — {vote_desc}.{top_neighbor_desc} "
        f"NOTE: Vote shares are not calibrated failure probabilities."
    )
    return explanation


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_predict_inputs(
    k: int,
    train_df: pd.DataFrame,
    strategy: str,
) -> None:
    """
    Validate inputs to predict().

    Raises
    ------
    ValueError : invalid k, empty training set, or unknown strategy
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got k={k!r}")

    if len(train_df) == 0:
        raise ValueError("Training set is empty — cannot classify.")

    if strategy not in ("majority", "weighted"):
        raise ValueError(
            f"voting_strategy must be 'majority' or 'weighted', got {strategy!r}"
        )

    if "condition" not in train_df.columns:
        raise ValueError(
            "Training DataFrame must contain a 'condition' column with labels."
        )

    valid_conditions = set(CONDITION_ORDER)
    actual_conditions = set(train_df["condition"].dropna().unique())
    unknown = actual_conditions - valid_conditions
    if unknown:
        raise ValueError(
            f"Training set contains unknown condition labels: {unknown}. "
            f"Expected: {valid_conditions}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    query: dict | pd.Series,
    train_df: pd.DataFrame,
    feature_config: dict,
    k: int = 5,
    strategy: str = "weighted",
    exclude_id: Optional[str] = None,
) -> PredictionResult:
    """
    Predict the condition of a CNC machine using k-Nearest Neighbors.

    Parameters
    ----------
    query          : the query machine record (dict or pandas Series).
                     Must contain all feature columns, including ordinal
                     '_rank' columns if ordinal features are in feature_config.
    train_df       : labeled training DataFrame with a 'condition' column.
                     Must have been preprocessed by data_preprocessing.py.
    feature_config : feature configuration from build_feature_config().
    k              : number of nearest neighbors to use (default 5).
                     If k > len(train_df), all training machines are used
                     with a warning surfaced in the result.
    strategy       : voting strategy:
                       "weighted"  (default) — similarity-weighted votes.
                       "majority"            — one vote per neighbor.
    exclude_id     : machine_id to exclude from the neighbor search.
                     Pass the query machine's own ID when classifying a
                     machine that is already in the training set, to prevent
                     the trivial self-match.

    Returns
    -------
    PredictionResult with prediction, vote distribution, neighbor details,
    and plain-language explanation.

    Raises
    ------
    ValueError : invalid k, empty training set, or unknown strategy
    TypeError  : invalid query type

    Example
    -------
    >>> result = predict(query_row, train_df, feat_cfg, k=5, strategy="weighted")
    >>> print(result.predicted_condition)
    'Warning'
    >>> print(result.explanation)
    'Prediction: Warning. Based on 5 nearest neighbor(s)...'
    """
    _validate_predict_inputs(k, train_df, strategy)

    # ── Step 1: Similarity search (finds top-k neighbors) ───────────────────
    # We request k neighbors. If the query machine is in the training set,
    # we must exclude it to avoid a trivial self-match.
    search_result = search(
        query          = query,
        reference_df   = train_df,
        feature_config = feature_config,
        k              = k,
        exclude_id     = exclude_id,
    )

    neighbors    = search_result.neighbors
    k_used       = len(neighbors)

    if k_used == 0:
        raise ValueError(
            "No neighbors found — the reference set may be empty after exclusions."
        )

    # ── Step 2: Extract neighbor conditions and similarities ─────────────────
    conditions  = neighbors["condition"].tolist()
    similarities = neighbors["similarity"].tolist()

    # ── Step 3: Vote ─────────────────────────────────────────────────────────
    if strategy == "weighted":
        predicted_condition, vote_distribution = _weighted_vote(conditions, similarities)
    else:
        predicted_condition, vote_distribution = _majority_vote(conditions, similarities)

    # ── Step 4: Normalise votes to shares ────────────────────────────────────
    vote_shares = _compute_vote_shares(vote_distribution)

    # ── Step 5: Build explanation ────────────────────────────────────────────
    explanation = _build_explanation(
        predicted_condition = predicted_condition,
        vote_distribution   = vote_distribution,
        vote_shares         = vote_shares,
        k_used              = k_used,
        voting_strategy     = strategy,
        neighbors           = neighbors,
    )

    return PredictionResult(
        predicted_condition = predicted_condition,
        vote_distribution   = vote_distribution,
        vote_shares         = vote_shares,
        neighbors           = neighbors,
        k_used              = k_used,
        k_requested         = k,
        voting_strategy     = strategy,
        explanation         = explanation,
        search_result       = search_result,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def format_prediction(result: PredictionResult) -> str:
    """
    Format a PredictionResult as a readable text report.

    Parameters
    ----------
    result : PredictionResult from predict()

    Returns
    -------
    Formatted multi-line string for console output or dashboard display.
    """
    lines = []
    lines.append(f"\n{'='*62}")
    lines.append("  k-NN CONDITION PREDICTION")
    lines.append(f"{'='*62}")
    lines.append(f"  Predicted condition : {result.predicted_condition}")
    lines.append(f"  k used / requested  : {result.k_used} / {result.k_requested}")
    lines.append(f"  Voting strategy     : {result.voting_strategy}")

    lines.append(f"\n  Vote distribution (NOT failure probabilities):")
    for label in CONDITION_ORDER:
        if label in result.vote_distribution:
            raw = result.vote_distribution[label]
            pct = result.vote_shares.get(label, 0) * 100
            bar = "#" * int(pct / 5)
            if result.voting_strategy == "majority":
                lines.append(f"    {label:<10} {int(raw):>2} votes  ({pct:5.1f}%)  {bar}")
            else:
                lines.append(f"    {label:<10} {raw:>6.3f}  ({pct:5.1f}%)  {bar}")

    lines.append(f"\n  {result.k_used} Nearest Neighbors:")
    lines.append(f"  {'Rank':<5} {'ID':<10} {'Sim':>6} {'Condition':<12} {'Wear':<12} {'Temp(C)':>8}")
    lines.append(f"  {'-'*55}")
    for _, row in result.neighbors.iterrows():
        rank = row.get("rank", "")
        mid  = row.get("machine_id", "N/A")
        sim  = row.get("similarity", 0)
        cond = row.get("condition", "N/A")
        wear = row.get("wear_level", "N/A")
        temp = row.get("operating_temp_c", float("nan"))
        temp_str = f"{temp:.1f}" if not (isinstance(temp, float) and temp != temp) else "N/A"
        lines.append(f"  {rank:<5} {mid:<10} {sim:>6.4f} {cond:<12} {wear:<12} {temp_str:>8}")

    lines.append(f"\n  Summary: {result.explanation[:120]}...")
    lines.append(f"{'='*62}\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK DEMO when run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from src.data_preprocessing import load_and_preprocess

    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    data_path = _os.path.join(base, "data", "cnc_machines.csv")

    pp = load_and_preprocess(data_path, verbose=False)
    train_df = pp["train"]
    feat_cfg = pp["feature_config"]

    # Classify the first 5 training machines
    print(f"\nClassifying first 5 training machines (k=5, weighted voting):\n")
    for i in range(5):
        query_row = train_df.iloc[i]
        qid       = query_row["machine_id"]
        true_cond = query_row["condition"]

        result = predict(
            query          = query_row,
            train_df       = train_df,
            feature_config = feat_cfg,
            k              = 5,
            strategy       = "weighted",
            exclude_id     = qid,
        )

        correct = "OK" if result.predicted_condition == true_cond else "WRONG"
        print(
            f"  {qid}  True:{true_cond:<10}  "
            f"Predicted:{result.predicted_condition:<10}  [{correct}]"
        )
        votes = {c: f"{v:.3f}" for c, v in result.vote_distribution.items()}
        print(f"         Votes: {votes}")
        print()
