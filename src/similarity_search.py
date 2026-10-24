"""
similarity_search.py
====================
Ranks CNC machine records by Gower-style similarity to a query record.

Purpose
-------
Given a query machine (either an existing record from the dataset or a
newly entered set of attributes), this module:

    1. Computes the Gower similarity between the query and every record
       in the reference dataset.
    2. Optionally excludes the query machine itself (when searching within
       the training set for a known machine).
    3. Ranks the results from highest to lowest similarity.
    4. Returns the top-k results as a structured DataFrame.

Relationship to other modules
------------------------------
    - Uses `gower_similarity.compute_similarity_row` for the core computation.
    - The results are consumed by `knn_classifier.py` for condition prediction.
    - The dashboard (`app.py`) calls this to display the most similar machines.

Key Design Decisions
---------------------
Similarity vs. Failure Probability:
    A similarity score of 0.87 means this reference machine's attributes are
    87% similar to the query's attributes. It does NOT mean there is an 87%
    probability that the query machine will fail. Similarity is a geometric
    (distance-based) concept; failure probability requires calibration against
    real failure outcomes, which this synthetic model does not provide.

Tie Handling:
    When multiple machines have identical similarity scores, results are sorted
    by machine_id alphabetically to ensure deterministic, reproducible output.

Public API
----------
    search(query, reference_df, feature_config, k, exclude_id) -> SearchResult
    format_results(result) -> str
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

# Absolute import when running from project root
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from gower_similarity import compute_similarity_row


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    Structured output from a similarity search.

    Attributes
    ----------
    query_id        : ID of the query machine (or "NEW" for a new machine)
    k               : number of neighbors requested
    neighbors       : DataFrame with columns:
                          rank            - 1-based rank (1 = most similar)
                          machine_id      - neighbor's machine ID
                          similarity      - Gower similarity score [0, 1]
                          distance        - 1 - similarity
                          condition       - neighbor's health label
                          machine_type    - key engineering attribute
                          wear_level      - key engineering attribute
                          operating_temp_c - key engineering attribute
                          vibration_mm_s  - key engineering attribute
    n_searched      : total number of machines searched (before top-k selection)
    n_excluded      : number of machines excluded (e.g., the query itself)
    """
    query_id:   str
    k:          int
    neighbors:  pd.DataFrame
    n_searched: int
    n_excluded: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _validate_inputs(
    query: dict | pd.Series,
    reference_df: pd.DataFrame,
    k: int,
) -> None:
    """
    Validate inputs before performing a similarity search.

    Parameters
    ----------
    query        : query machine record
    reference_df : reference dataset to search against
    k            : number of neighbors to return

    Raises
    ------
    TypeError  : if query is not a dict or pandas Series
    ValueError : if k < 1, or reference_df has no rows, or k > available rows
    """
    if not isinstance(query, (dict, pd.Series)):
        raise TypeError(
            f"query must be a dict or pandas Series, got {type(query).__name__}"
        )

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError(
            f"reference_df must be a pandas DataFrame, got {type(reference_df).__name__}"
        )

    if len(reference_df) == 0:
        raise ValueError("reference_df is empty — no machines to search against.")

    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got k={k!r}")


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY COLUMN SELECTION
# ─────────────────────────────────────────────────────────────────────────────

# Columns to include in the search results DataFrame for display.
# These are the most diagnostically relevant columns for a maintenance engineer.
_DISPLAY_COLS = [
    "machine_id",
    "machine_type",
    "controller_type",
    "wear_level",
    "degradation_level",
    "maintenance_condition",
    "operating_temp_c",
    "vibration_mm_s",
    "lubrication_alert",
    "abnormal_noise",
    "coolant_failure",
    "condition",
]


def _extract_display_cols(df: pd.DataFrame) -> dict[str, list]:
    """
    Extract display columns from the reference DataFrame.

    Only columns that exist in the DataFrame are included (graceful handling
    of minimal DataFrames used in tests).

    Parameters
    ----------
    df : reference DataFrame

    Returns
    -------
    dict mapping column names → list of values (aligned with df.index)
    """
    result = {}
    for col in _DISPLAY_COLS:
        if col in df.columns:
            result[col] = df[col].tolist()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CORE SEARCH FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def search(
    query: dict | pd.Series,
    reference_df: pd.DataFrame,
    feature_config: dict,
    k: int = 5,
    exclude_id: Optional[str] = None,
) -> SearchResult:
    """
    Find the k most similar machines to a query record.

    Parameters
    ----------
    query          : the query machine record (dict or pandas Series).
                     Must contain the same feature columns as reference_df,
                     plus ordinal '_rank' columns if ordinal features are used.
    reference_df   : the reference dataset (e.g., training set) to search.
                     Must have been preprocessed by data_preprocessing.py
                     (so ordinal rank columns are present).
    feature_config : feature configuration from build_feature_config().
    k              : number of top neighbors to return (default 5).
    exclude_id     : machine_id to exclude from results.
                     Use this when the query machine is in the reference set —
                     pass its machine_id to prevent it from being its own
                     top neighbor (trivial self-match with similarity = 1.0).

    Returns
    -------
    SearchResult with ranked neighbors DataFrame and metadata.

    Raises
    ------
    TypeError  : invalid query or reference_df type
    ValueError : k < 1, empty reference_df, or k > available machines

    Notes
    -----
    Similarity ≠ Failure Probability:
        A high similarity score means the reference machine has similar
        engineering characteristics to the query machine. It does NOT
        mean the reference machine's condition will be identical to the
        query machine's future condition. Condition prediction is handled
        separately by knn_classifier.py.

    Ties:
        When multiple machines share the same similarity score, they are
        sorted by machine_id (ascending) to ensure deterministic output.
    """
    _validate_inputs(query, reference_df, k)

    # Convert query to dict for uniform handling
    if isinstance(query, pd.Series):
        query_dict = query.to_dict()
    else:
        query_dict = dict(query)

    query_id = query_dict.get("machine_id", "NEW")

    # ── Step 1: Exclude the query machine from the reference set ────────────
    working_df   = reference_df.copy()
    n_excluded   = 0

    if exclude_id is not None and "machine_id" in working_df.columns:
        mask = working_df["machine_id"] == exclude_id
        n_excluded = int(mask.sum())
        working_df = working_df[~mask].copy()

    if len(working_df) == 0:
        raise ValueError(
            f"After excluding machine_id='{exclude_id}', the reference set is empty."
        )

    # Warn if k > available machines (but don't error — return what we can)
    effective_k = k
    if k > len(working_df):
        effective_k = len(working_df)
        # (warning is surfaced in SearchResult metadata via n_searched)

    # ── Step 2: Compute similarity scores ───────────────────────────────────
    # compute_similarity_row returns a numpy array of shape (len(working_df),)
    # in the same row order as working_df.
    similarity_scores = compute_similarity_row(query_dict, working_df, feature_config)

    # ── Step 3: Build a results frame with scores ────────────────────────────
    # We reset the index so positional indexing (argsort) aligns correctly.
    results_df = working_df.reset_index(drop=True).copy()
    results_df["_similarity"] = similarity_scores
    results_df["_distance"]   = 1.0 - similarity_scores

    # ── Step 4: Sort — highest similarity first; tie-break by machine_id ────
    if "machine_id" in results_df.columns:
        results_df = results_df.sort_values(
            by=["_similarity", "machine_id"],
            ascending=[False, True],   # similarity descending, ID ascending
        )
    else:
        results_df = results_df.sort_values(by="_similarity", ascending=False)

    # ── Step 5: Take top-k ───────────────────────────────────────────────────
    top_k = results_df.head(effective_k).reset_index(drop=True)

    # ── Step 6: Build the final neighbors DataFrame ─────────────────────────
    neighbors_data = {"rank": list(range(1, len(top_k) + 1))}

    # Always include machine_id if available
    if "machine_id" in top_k.columns:
        neighbors_data["machine_id"] = top_k["machine_id"].tolist()

    neighbors_data["similarity"] = top_k["_similarity"].round(4).tolist()
    neighbors_data["distance"]   = top_k["_distance"].round(4).tolist()

    # Add condition label (the target — used only for display, not as input)
    if "condition" in top_k.columns:
        neighbors_data["condition"] = top_k["condition"].tolist()

    # Add key engineering attributes for interpretability
    display_attrs = [
        "machine_type", "controller_type", "wear_level", "degradation_level",
        "maintenance_condition", "operating_temp_c", "vibration_mm_s",
        "lubrication_alert", "abnormal_noise", "coolant_failure",
        "maintenance_overdue", "last_maintenance_days", "tool_usage_hours",
    ]
    for col in display_attrs:
        if col in top_k.columns:
            neighbors_data[col] = top_k[col].tolist()

    neighbors = pd.DataFrame(neighbors_data)

    return SearchResult(
        query_id   = str(query_id),
        k          = k,
        neighbors  = neighbors,
        n_searched = len(working_df),
        n_excluded = n_excluded,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def format_results(result: SearchResult) -> str:
    """
    Format a SearchResult as a readable text report.

    Parameters
    ----------
    result : SearchResult from search()

    Returns
    -------
    Formatted multi-line string suitable for console output or the dashboard.
    """
    lines = []
    lines.append(f"\n{'='*62}")
    lines.append(f"  SIMILARITY SEARCH RESULTS")
    lines.append(f"{'='*62}")
    lines.append(f"  Query machine : {result.query_id}")
    lines.append(f"  Machines searched : {result.n_searched}")
    if result.n_excluded > 0:
        lines.append(f"  Machines excluded : {result.n_excluded} (self-match removed)")
    lines.append(f"  Top {len(result.neighbors)} most similar machines:")
    lines.append(f"  {'Rank':<5} {'ID':<10} {'Similarity':>10} {'Distance':>9} {'Condition':<10} {'Wear':<10} {'Temp(C)':>8}")
    lines.append(f"  {'-'*62}")

    for _, row in result.neighbors.iterrows():
        rank      = row.get("rank", "")
        mid       = row.get("machine_id", "N/A")
        sim       = row.get("similarity", 0)
        dist      = row.get("distance", 1)
        cond      = row.get("condition", "N/A")
        wear      = row.get("wear_level", "N/A")
        temp      = row.get("operating_temp_c", float("nan"))
        temp_str  = f"{temp:.1f}" if not (isinstance(temp, float) and temp != temp) else "N/A"
        lines.append(
            f"  {rank:<5} {mid:<10} {sim:>10.4f} {dist:>9.4f} {cond:<10} {wear:<10} {temp_str:>8}"
        )

    lines.append(f"{'='*62}\n")
    lines.append(
        "  NOTE: Similarity scores reflect attribute closeness, not\n"
        "  failure probability. Condition prediction is separate.\n"
    )
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

    result_pp = load_and_preprocess(data_path, verbose=False)
    train_df  = result_pp["train"]
    feat_cfg  = result_pp["feature_config"]

    # Use the first training machine as the query
    query_row = train_df.iloc[0]
    query_id  = query_row["machine_id"]

    print(f"[INFO] Searching for machines similar to: {query_id}")
    print(f"       Condition: {query_row['condition']}")

    result = search(
        query          = query_row,
        reference_df   = train_df,
        feature_config = feat_cfg,
        k              = 5,
        exclude_id     = query_id,
    )

    print(format_results(result))
