"""
gower_similarity.py
===================
Custom implementation of the Gower-style mixed-attribute similarity measure.

Mathematical Foundation
-----------------------
For two machine records i and j across p features:

    S(i,j) = sum_k [ w_k * delta_ijk * s_ijk ]
             ----------------------------------
             sum_k [ w_k * delta_ijk ]

Where:
    w_k       : weight of feature k (default 1.0)
    delta_ijk : 1 if feature k is available (non-missing) for both i and j;
                0 if either value is missing.
                For asymmetric binary features: also 0 when both values are 0
                (shared absence carries no information for rare fault indicators).
    s_ijk     : similarity contribution for feature k, always in [0.0, 1.0]

Per-type similarity contributions:
    Nominal   : 1 if same category, 0 otherwise
    Binary    : symmetric → 1 if equal, 0 otherwise
                asymmetric → same as symmetric but delta=0 when both are 0
    Ordinal   : 1 - |norm_rank_i - norm_rank_j|
                (normalized ranks supplied as precomputed '_rank' columns)
    Numeric   : 1 - |x_ik - x_jk| / R_k
                (R_k is the fixed engineering range for feature k)

Distance is defined as: D(i,j) = 1 - S(i,j)

Key Design Decisions
--------------------
1. Missing values are handled via delta_ijk = 0: they fall out of both
   numerator and denominator, so they do not bias the similarity score.

2. Feature ranges (R_k) are passed in as fixed engineering constants from
   data_preprocessing.py — they are never computed from the dataset to
   prevent data leakage.

3. The function works on raw Python dicts or pandas Series, making it
   easy to use in the k-NN classifier and similarity search modules.

4. All similarity contributions are guaranteed to be in [0, 1] by the
   math of each sub-formula. We add a safety clip to catch floating-point
   edge cases.

5. The similarity of a record with itself must equal 1.0 for all
   non-missing features.

Public API
----------
    compute_similarity(record_i, record_j, feature_config) -> float
        Compute similarity between two machine records.

    compute_similarity_matrix(df, feature_config) -> np.ndarray
        Compute all pairwise similarities for a dataset.

    compute_similarity_row(query, df, feature_config) -> np.ndarray
        Compute similarity of one query record against all rows in df.
"""

from typing import Any
import math
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: Per-attribute-type similarity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_missing(value: Any) -> bool:
    """
    Return True if value is missing (None, NaN, empty string).

    We use this instead of `value is None` because pandas may represent
    missing numeric values as float('nan'), which is not `None`.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _nominal_similarity(val_i: Any, val_j: Any) -> tuple[float, float]:
    """
    Compute similarity for a nominal (categorical) feature.

    Rule: 1.0 if both values are identical, 0.0 otherwise.

    Returns
    -------
    (s, delta) where:
        s     : similarity contribution [0, 1]
        delta : 1 if both values are present, 0 if either is missing
    """
    if _is_missing(val_i) or _is_missing(val_j):
        return 0.0, 0.0   # Missing → excluded from calculation

    s = 1.0 if str(val_i) == str(val_j) else 0.0
    return s, 1.0


def _binary_similarity(
    val_i: Any,
    val_j: Any,
    semantic: str = "symmetric",
) -> tuple[float, float]:
    """
    Compute similarity for a binary (0/1) feature.

    Two semantics are supported:

    symmetric:
        Both 0-0 and 1-1 are counted as matches.
        delta = 1 if both values are present.
        s = 1 if val_i == val_j, else 0.

    asymmetric:
        A shared absence (0-0) is excluded from the similarity calculation
        (delta = 0) because the shared non-occurrence of a rare event carries
        little discriminating information.
        For 0-1, 1-0, and 1-1 pairs: delta = 1, s = 1 if equal else 0.

    Parameters
    ----------
    val_i, val_j : the binary values (expected to be 0 or 1, int or float)
    semantic     : "symmetric" (default) or "asymmetric"

    Returns
    -------
    (s, delta)
    """
    if _is_missing(val_i) or _is_missing(val_j):
        return 0.0, 0.0

    # Convert to integer for safe comparison (handles 0.0 / 1.0 floats)
    try:
        vi = int(val_i)
        vj = int(val_j)
    except (ValueError, TypeError):
        return 0.0, 0.0

    if semantic == "asymmetric":
        # Shared absence: both machines have value 0 → exclude this feature pair
        if vi == 0 and vj == 0:
            return 0.0, 0.0   # delta = 0 → excluded from numerator and denominator

    s = 1.0 if vi == vj else 0.0
    return s, 1.0


def _ordinal_similarity(
    rank_i: Any,
    rank_j: Any,
) -> tuple[float, float]:
    """
    Compute similarity for an ordinal feature using normalized ranks.

    The normalized ranks are precomputed by data_preprocessing.encode_ordinal()
    and stored in columns named '<feature>_rank'. They are always in [0, 1].

    Formula:
        s_ijk = 1 - |rank_i - rank_j|

    Since both ranks are in [0, 1], the absolute difference is also in [0, 1],
    so s_ijk is guaranteed to be in [0, 1].

    Parameters
    ----------
    rank_i, rank_j : normalized ranks (floats in [0, 1])

    Returns
    -------
    (s, delta)
    """
    if _is_missing(rank_i) or _is_missing(rank_j):
        return 0.0, 0.0

    try:
        ri = float(rank_i)
        rj = float(rank_j)
    except (ValueError, TypeError):
        return 0.0, 0.0

    s = 1.0 - abs(ri - rj)
    # Safety clip: floating-point arithmetic could push s slightly outside [0,1]
    s = max(0.0, min(1.0, s))
    return s, 1.0


def _numeric_similarity(
    val_i: Any,
    val_j: Any,
    feature_range: float,
) -> tuple[float, float]:
    """
    Compute similarity for a numeric interval-scaled feature.

    Formula:
        s_ijk = 1 - |x_ik - x_jk| / R_k

    Parameters
    ----------
    val_i, val_j   : numeric values (float or int)
    feature_range  : R_k — the valid range (max - min) for this feature.
                     Must be > 0 to avoid division by zero.

    Returns
    -------
    (s, delta)

    Raises
    ------
    ValueError : if feature_range <= 0
    """
    if _is_missing(val_i) or _is_missing(val_j):
        return 0.0, 0.0

    if feature_range <= 0:
        raise ValueError(
            f"feature_range must be > 0 to compute numeric similarity; "
            f"got {feature_range}. Check NUMERIC_RANGES in data_preprocessing.py."
        )

    try:
        vi = float(val_i)
        vj = float(val_j)
    except (ValueError, TypeError):
        return 0.0, 0.0

    s = 1.0 - abs(vi - vj) / feature_range
    # Safety clip: values near the boundary of the range could produce s ≈ 0
    # or very slightly negative due to floating-point arithmetic.
    s = max(0.0, min(1.0, s))
    return s, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: Main similarity function
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity(
    record_i: dict | pd.Series,
    record_j: dict | pd.Series,
    feature_config: dict,
) -> float:
    """
    Compute the Gower-style similarity between two CNC machine records.

    Parameters
    ----------
    record_i       : first machine record (dict or pandas Series)
    record_j       : second machine record (dict or pandas Series)
    feature_config : feature configuration dict from build_feature_config().
                     Must contain keys:
                       'nominal'          → list of nominal column names
                       'binary'           → list of binary column names
                       'ordinal'          → list of ordinal column names
                       'numeric'          → list of numeric column names
                       'binary_semantics' → dict: column → "symmetric"|"asymmetric"
                       'numeric_ranges'   → dict: column → (min, max)
                       'weights'          → dict: column → float weight
                     For ordinal features, the precomputed normalized rank is
                     expected in a column named '<feature>_rank'.

    Returns
    -------
    float : similarity score in [0.0, 1.0].
            Returns 0.0 if no features could be compared (all missing).

    Notes
    -----
    - Identical records return 1.0 for all non-missing features.
    - Missing values (None, NaN) are automatically excluded from both the
      numerator and denominator via delta = 0.
    - The distance between two records is D = 1 - S.
    """
    # Allow both dict and pandas Series as input
    if isinstance(record_i, pd.Series):
        record_i = record_i.to_dict()
    if isinstance(record_j, pd.Series):
        record_j = record_j.to_dict()

    weights        = feature_config["weights"]
    binary_sem     = feature_config["binary_semantics"]
    numeric_ranges = feature_config["numeric_ranges"]

    numerator   = 0.0   # sum of  (w_k * delta_ijk * s_ijk)
    denominator = 0.0   # sum of  (w_k * delta_ijk)

    # ── Nominal features ────────────────────────────────────────────────────
    for col in feature_config["nominal"]:
        w = weights.get(col, 1.0)
        s, delta = _nominal_similarity(
            record_i.get(col),
            record_j.get(col),
        )
        numerator   += w * delta * s
        denominator += w * delta

    # ── Binary features ──────────────────────────────────────────────────────
    for col in feature_config["binary"]:
        w        = weights.get(col, 1.0)
        semantic = binary_sem.get(col, "symmetric")
        s, delta = _binary_similarity(
            record_i.get(col),
            record_j.get(col),
            semantic=semantic,
        )
        numerator   += w * delta * s
        denominator += w * delta

    # ── Ordinal features (use precomputed '_rank' columns) ───────────────────
    for col in feature_config["ordinal"]:
        w        = weights.get(col, 1.0)
        rank_col = f"{col}_rank"
        s, delta = _ordinal_similarity(
            record_i.get(rank_col),
            record_j.get(rank_col),
        )
        numerator   += w * delta * s
        denominator += w * delta

    # ── Numeric features ─────────────────────────────────────────────────────
    for col in feature_config["numeric"]:
        w               = weights.get(col, 1.0)
        lo, hi          = numeric_ranges[col]
        feature_range   = hi - lo
        s, delta = _numeric_similarity(
            record_i.get(col),
            record_j.get(col),
            feature_range=feature_range,
        )
        numerator   += w * delta * s
        denominator += w * delta

    # ── Final score ───────────────────────────────────────────────────────────
    if denominator == 0.0:
        # No features could be compared (all missing or all asymmetric 0-0).
        # Return 0.0 — the two records are considered maximally dissimilar
        # when no information is available to compare them.
        return 0.0

    similarity = numerator / denominator
    # Final safety clip for floating-point drift
    return float(max(0.0, min(1.0, similarity)))


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: Batch similarity computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_row(
    query: dict | pd.Series,
    df: pd.DataFrame,
    feature_config: dict,
) -> np.ndarray:
    """
    Compute the similarity of one query record against every row in a DataFrame.

    This is the core operation for both the similarity search and the k-NN
    classifier. For a dataset of N rows, it returns an array of N similarity
    scores in the same row order as `df`.

    Parameters
    ----------
    query          : the query machine record (dict or pandas Series)
    df             : reference DataFrame (e.g., training set)
    feature_config : feature configuration from build_feature_config()

    Returns
    -------
    np.ndarray of shape (len(df),) with similarity scores in [0, 1]
    """
    if isinstance(query, pd.Series):
        query = query.to_dict()

    scores = np.array([
        compute_similarity(query, row, feature_config)
        for _, row in df.iterrows()
    ])
    return scores


def compute_similarity_matrix(
    df: pd.DataFrame,
    feature_config: dict,
) -> np.ndarray:
    """
    Compute the full N×N pairwise similarity matrix for a dataset.

    For N machines, this computes N*(N-1)/2 unique pairs (the matrix is
    symmetric: S(i,j) == S(j,i)) and self-similarity S(i,i) == 1.0.

    Parameters
    ----------
    df             : DataFrame of machine records
    feature_config : feature configuration from build_feature_config()

    Returns
    -------
    np.ndarray of shape (N, N) with similarities in [0, 1].
    The diagonal is all 1.0 (each machine is identical to itself).

    Note
    ----
    For large datasets (N > 1000), this matrix becomes expensive.
    For this academic project with N ≈ 300, it is fast enough.
    """
    records = [row.to_dict() for _, row in df.iterrows()]
    n = len(records)
    matrix = np.ones((n, n), dtype=float)   # Initialize diagonal to 1.0

    for i in range(n):
        for j in range(i + 1, n):   # Only compute upper triangle
            s = compute_similarity(records[i], records[j], feature_config)
            matrix[i, j] = s
            matrix[j, i] = s        # Matrix is symmetric

    return matrix


def compute_distance(
    record_i: dict | pd.Series,
    record_j: dict | pd.Series,
    feature_config: dict,
) -> float:
    """
    Compute the Gower-style distance between two records.

    Distance is defined as: D(i, j) = 1 - S(i, j)

    Parameters
    ----------
    record_i, record_j : machine records
    feature_config     : feature configuration

    Returns
    -------
    float in [0.0, 1.0], where 0.0 = identical, 1.0 = maximally different
    """
    return 1.0 - compute_similarity(record_i, record_j, feature_config)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST: Run a manually verifiable example when executed directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Manual verification example.

    Two machines with 5 features (1 nominal, 1 binary, 1 ordinal, 2 numeric).
    All weights = 1.0, no missing values.

    Feature         | Type         | Machine A    | Machine B    | s_AB
    ----------------|--------------|--------------|--------------|-------
    machine_type    | Nominal      | Milling      | Milling      | 1.000
    lubrication_    | Binary sym   | 0            | 1            | 0.000
    alert           |              |              |              |
    wear_level_rank | Ordinal      | 0.000 (Low)  | 0.667 (High) | 0.333
    operating_      | Numeric      | 55.0         | 75.0         | 0.800
    temp_c          | R=100        |              |              |
    vibration_mm_s  | Numeric      | 3.5          | 8.0          | 0.819
                    | R=24.9       |              |              |

    Numerator   = 1*1.000 + 1*0.000 + 1*0.333 + 1*0.800 + 1*0.819 = 2.952
    Denominator = 5
    S(A, B)     = 2.952 / 5 = 0.590
    D(A, B)     = 1 - 0.590 = 0.410
    """

    mini_config = {
        "nominal":  ["machine_type"],
        "binary":   ["lubrication_alert"],
        "ordinal":  ["wear_level"],
        "numeric":  ["operating_temp_c", "vibration_mm_s"],
        "binary_semantics": {"lubrication_alert": "symmetric"},
        "numeric_ranges": {
            "operating_temp_c": (20.0, 120.0),   # Range = 100
            "vibration_mm_s":   (0.1,  25.0),    # Range = 24.9
        },
        "weights": {
            "machine_type":    1.0,
            "lubrication_alert": 1.0,
            "wear_level":      1.0,
            "operating_temp_c": 1.0,
            "vibration_mm_s":  1.0,
        },
    }

    machine_A = {
        "machine_type":       "Milling",
        "lubrication_alert":  0,
        "wear_level":         "Low",
        "wear_level_rank":    0.000,      # Low = rank 0/3 = 0.000
        "operating_temp_c":   55.0,
        "vibration_mm_s":     3.5,
    }

    machine_B = {
        "machine_type":       "Milling",
        "lubrication_alert":  1,
        "wear_level":         "High",
        "wear_level_rank":    2/3,        # High = rank 2/3 ≈ 0.667
        "operating_temp_c":   75.0,
        "vibration_mm_s":     8.0,
    }

    # Manual calculation
    s_nominal   = 1.000   # Milling == Milling
    s_binary    = 0.000   # 0 != 1
    s_ordinal   = 1.000 - abs(0.000 - 2/3)        # ≈ 0.333
    s_temp      = 1.000 - abs(55.0 - 75.0) / 100  # = 0.800
    s_vibration = 1.000 - abs(3.5 - 8.0) / 24.9   # ≈ 0.819

    manual_sim = (s_nominal + s_binary + s_ordinal + s_temp + s_vibration) / 5

    # Code calculation
    computed_sim = compute_similarity(machine_A, machine_B, mini_config)

    print("── Manual Verification Example ──────────────────────────────")
    print(f"  s(machine_type)    = {s_nominal:.4f}  (Milling == Milling)")
    print(f"  s(lubrication)     = {s_binary:.4f}  (0 != 1)")
    print(f"  s(wear_level)      = {s_ordinal:.4f}  (|0.000 - 0.667| = 0.667)")
    print(f"  s(operating_temp)  = {s_temp:.4f}  (|55-75|/100 = 0.20)")
    print(f"  s(vibration)       = {s_vibration:.4f}  (|3.5-8.0|/24.9 ≈ 0.181)")
    print(f"  ------------------------------------")
    print(f"  Manual S(A,B)      = {manual_sim:.4f}")
    print(f"  Computed S(A,B)    = {computed_sim:.4f}")
    print(f"  Match: {abs(manual_sim - computed_sim) < 1e-9}")
    print(f"  D(A,B) = 1 - S(A,B) = {1 - computed_sim:.4f}")
    print("────────────────────────────────────────────────────────────")
