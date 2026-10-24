"""
data_preprocessing.py
=====================
Loads, validates, encodes, and splits the CNC machine dataset.

Responsibilities
----------------
- Load cnc_machines.csv and validate the schema.
- Detect and report missing values, duplicate records, and invalid categories.
- Check that numeric values are within their defined engineering ranges.
- Encode ordinal attributes using statically defined normalized ranks.
  (Encoding is NOT fitted from data to prevent leakage.)
- Separate features from the target column.
- Split into train / validation / test sets using chronological ordering.
- Export feature configuration that the Gower similarity engine will consume.

What this module does NOT do
------------------------------
- It does not compute any similarity scores.
- It does not train a model.
- It does not impute missing values — missing values are passed through as NaN
  so that the Gower similarity engine can exclude them from its denominator.

Data Leakage Prevention
-----------------------
The numeric feature ranges used for Gower similarity are defined as engineering
constants here — they are NOT derived from min/max of the training set. This
means the same ranges are used consistently regardless of which data split is
being processed, and no information flows from test rows back to training logic.

Usage
-----
    from src.data_preprocessing import load_and_preprocess

    result = load_and_preprocess("data/cnc_machines.csv")
    df_train    = result["train"]
    df_val      = result["val"]
    df_test     = result["test"]
    feat_config = result["feature_config"]
"""

import os
import warnings
from typing import Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# 1. SCHEMA DEFINITION
#    Every expected column with its type, attribute class, and valid range /
#    category set.  This acts as a living specification for the dataset.
# ---------------------------------------------------------------------------

# Ordered lists of valid categories for each nominal column
VALID_NOMINAL: dict[str, list[str]] = {
    "machine_type":    ["Milling", "Turning", "Drilling"],
    "controller_type": ["Siemens", "Fanuc", "Haas"],
    "cooling_system":  ["Air", "Liquid", "Hybrid"],
}

# Valid values for binary columns (all are 0 or 1 stored as int)
VALID_BINARY: list[str] = [
    "lubrication_alert",
    "abnormal_noise",
    "coolant_failure",
    "maintenance_overdue",
]

# Ordinal columns with their ORDERED categories (worst last in each list)
# The index of a category in this list becomes its raw rank (0-based).
ORDINAL_ORDERS: dict[str, list[str]] = {
    "wear_level":             ["Low", "Medium", "High", "Critical"],
    "maintenance_condition":  ["Good", "Fair", "Poor"],
    "degradation_level":      ["Minimal", "Moderate", "Severe"],
}

# Numeric columns with their valid engineering range [min, max]
# These are FIXED CONSTANTS — never derived from training data.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "operating_temp_c":      (20.0,  120.0),
    "vibration_mm_s":        (0.1,   25.0),
    "spindle_speed_rpm":     (500.0, 15000.0),
    "operating_hours":       (0.0,   50000.0),
    "power_consumption_kw":  (1.0,   30.0),
    "tool_usage_hours":      (0.0,   500.0),
    "avg_load_pct":          (10.0,  100.0),
    "coolant_temp_c":        (15.0,  80.0),
    "last_maintenance_days": (0.0,   730.0),
}

# Metadata and target columns (excluded from feature lists)
METADATA_COLS: list[str] = ["machine_id", "record_date"]
TARGET_COL: str = "condition"
VALID_CONDITIONS: list[str] = ["Normal", "Warning", "Critical"]

# All feature column names in a consistent order
FEATURE_COLS: list[str] = (
    list(VALID_NOMINAL.keys())       # 3 nominal
    + VALID_BINARY                   # 4 binary
    + list(ORDINAL_ORDERS.keys())    # 3 ordinal
    + list(NUMERIC_RANGES.keys())    # 9 numeric
)

# All expected columns (complete schema)
ALL_EXPECTED_COLS: list[str] = METADATA_COLS + FEATURE_COLS + [TARGET_COL]


# ---------------------------------------------------------------------------
# 2. FEATURE CONFIGURATION
#    This dict will be passed to the Gower similarity engine so it knows
#    which comparison rule to apply for each feature column.
# ---------------------------------------------------------------------------

# Binary attribute semantics
# symmetric  → both 0-0 and 1-1 are matches; use standard equality comparison
# asymmetric → 0-0 also counts as a match, but the 1-1 (shared fault) is the
#               most informative; in this dataset we handle both the same way
#               at the similarity engine level (delta=1 for all non-null pairs)
BINARY_SEMANTICS: dict[str, str] = {
    "lubrication_alert":   "symmetric",
    "abnormal_noise":      "symmetric",
    "coolant_failure":     "asymmetric",
    "maintenance_overdue": "symmetric",
}


def build_feature_config() -> dict[str, Any]:
    """
    Build and return the feature configuration dictionary.

    This configuration is consumed by gower_similarity.py to know:
    - Which columns are nominal / binary / ordinal / numeric
    - For ordinal columns: what the ordered categories are
    - For numeric columns: what the valid range R_k is
    - For binary columns: whether comparison is symmetric or asymmetric
    - Default feature weights (all 1.0 initially)

    Returns
    -------
    dict with keys:
        "nominal"   → list of column names
        "binary"    → list of column names
        "ordinal"   → list of column names
        "numeric"   → list of column names
        "ordinal_orders"   → dict mapping column → ordered category list
        "numeric_ranges"   → dict mapping column → (min, max)
        "binary_semantics" → dict mapping column → "symmetric"|"asymmetric"
        "weights"          → dict mapping column → float (default 1.0)
    """
    weights = {col: 1.0 for col in FEATURE_COLS}

    return {
        "nominal":          list(VALID_NOMINAL.keys()),
        "binary":           VALID_BINARY,
        "ordinal":          list(ORDINAL_ORDERS.keys()),
        "numeric":          list(NUMERIC_RANGES.keys()),
        "ordinal_orders":   ORDINAL_ORDERS,
        "numeric_ranges":   NUMERIC_RANGES,
        "binary_semantics": BINARY_SEMANTICS,
        "weights":          weights,
    }


# ---------------------------------------------------------------------------
# 3. DATA LOADING
# ---------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Load the CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : path to cnc_machines.csv

    Returns
    -------
    pd.DataFrame with all columns as loaded (no transformations yet)

    Raises
    ------
    FileNotFoundError : if the file does not exist
    ValueError        : if the file is empty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, dtype=str)   # Load everything as string first

    if df.empty:
        raise ValueError(f"Dataset is empty: {filepath}")

    return df


# ---------------------------------------------------------------------------
# 4. SCHEMA VALIDATION
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> list[str]:
    """
    Check that all expected columns are present.

    Parameters
    ----------
    df : raw DataFrame from load_csv

    Returns
    -------
    list of error message strings (empty list if schema is valid)
    """
    errors = []
    missing_cols = [c for c in ALL_EXPECTED_COLS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    extra_cols = [c for c in df.columns if c not in ALL_EXPECTED_COLS]
    if extra_cols:
        # Warn but do not error — extra columns are ignored
        warnings.warn(f"Unexpected columns (will be ignored): {extra_cols}")

    return errors


def validate_categories(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """
    Validate nominal and ordinal categorical values.

    Invalid rows are flagged with a warning. The function returns both
    the list of issues found and the DataFrame with invalid rows removed.

    Parameters
    ----------
    df : DataFrame (string-typed columns)

    Returns
    -------
    issues : list of descriptive warning strings
    df     : DataFrame with invalid categorical rows dropped
    """
    issues = []
    mask_invalid = pd.Series([False] * len(df), index=df.index)

    # Check nominal columns
    for col, valid_vals in VALID_NOMINAL.items():
        if col not in df.columns:
            continue
        bad = ~df[col].isin(valid_vals)
        if bad.any():
            bad_vals = df.loc[bad, col].unique().tolist()
            issues.append(
                f"Nominal column '{col}': {bad.sum()} invalid value(s) → {bad_vals}"
            )
            mask_invalid |= bad

    # Check ordinal columns
    for col, valid_vals in ORDINAL_ORDERS.items():
        if col not in df.columns:
            continue
        # Allow NaN (empty string from CSV) to pass — handled separately
        non_null = df[col].notna() & (df[col] != "")
        bad = non_null & ~df[col].isin(valid_vals)
        if bad.any():
            bad_vals = df.loc[bad, col].unique().tolist()
            issues.append(
                f"Ordinal column '{col}': {bad.sum()} invalid value(s) → {bad_vals}"
            )
            mask_invalid |= bad

    # Check target column
    if TARGET_COL in df.columns:
        bad = ~df[TARGET_COL].isin(VALID_CONDITIONS)
        if bad.any():
            bad_vals = df.loc[bad, TARGET_COL].unique().tolist()
            issues.append(
                f"Target column '{TARGET_COL}': {bad.sum()} invalid value(s) → {bad_vals}"
            )
            mask_invalid |= bad

    # Drop invalid rows
    n_dropped = mask_invalid.sum()
    if n_dropped > 0:
        df = df[~mask_invalid].copy()
        issues.append(f"  → Dropped {n_dropped} row(s) with invalid categories.")

    return issues, df


def validate_binary(df: pd.DataFrame) -> list[str]:
    """
    Validate that binary columns contain only 0 or 1 (after coercion to int).

    Parameters
    ----------
    df : DataFrame (binary columns already coerced to numeric)

    Returns
    -------
    list of issue strings
    """
    issues = []
    for col in VALID_BINARY:
        if col not in df.columns:
            continue
        valid = df[col].isin([0, 1, 0.0, 1.0])
        if not valid.all():
            n_bad = (~valid).sum()
            issues.append(f"Binary column '{col}': {n_bad} values not in {{0, 1}}")
    return issues


def validate_numeric_ranges(df: pd.DataFrame) -> list[str]:
    """
    Check that numeric column values fall within their defined engineering ranges.

    Out-of-range values are warned about but NOT dropped — they will be
    clamped later in type_coercion(). Missing values (NaN) are skipped.

    Parameters
    ----------
    df : DataFrame with numeric columns already coerced to float

    Returns
    -------
    list of warning strings
    """
    warnings_list = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col]
        out_of_range = series.dropna()
        out_of_range = out_of_range[(out_of_range < lo) | (out_of_range > hi)]
        if len(out_of_range) > 0:
            warnings_list.append(
                f"Numeric column '{col}': {len(out_of_range)} value(s) outside "
                f"[{lo}, {hi}] — will be clamped."
            )
    return warnings_list


# ---------------------------------------------------------------------------
# 5. TYPE COERCION
#    Convert string-typed columns to their proper Python/pandas types.
# ---------------------------------------------------------------------------

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each column to its correct dtype.

    - Binary columns → int (0 or 1); NaN stays as NaN (float)
    - Numeric columns → float; empty strings become NaN
    - Categorical/ordinal columns → str (already string)
    - record_date → datetime (for chronological splitting)

    Parameters
    ----------
    df : raw string-typed DataFrame

    Returns
    -------
    df : DataFrame with proper column types
    """
    df = df.copy()

    # Date column
    if "record_date" in df.columns:
        df["record_date"] = pd.to_datetime(df["record_date"], errors="coerce")

    # Binary columns: convert to numeric (0/1), coerce errors to NaN
    for col in VALID_BINARY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Numeric columns: convert to float, empty → NaN
    for col in NUMERIC_RANGES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clamp_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clamp numeric columns to their defined engineering ranges.

    Values below the minimum are clamped up; values above the maximum
    are clamped down. NaN values are left untouched.

    Parameters
    ----------
    df : DataFrame with numeric columns as float

    Returns
    -------
    df : DataFrame with clamped numeric columns
    """
    df = df.copy()
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ---------------------------------------------------------------------------
# 6. ORDINAL ENCODING
#    Map ordered category strings to normalized ranks in [0, 1].
#    This encoding is defined statically — it is NOT fitted from data.
# ---------------------------------------------------------------------------

def encode_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add encoded columns for each ordinal attribute.

    For a column with n ordered levels, the normalized rank of level at
    index i is: i / (n - 1).

    Example (wear_level, 4 levels):
        Low     → 0 / 3 = 0.000
        Medium  → 1 / 3 = 0.333
        High    → 2 / 3 = 0.667
        Critical → 3 / 3 = 1.000

    New columns are named with a '_rank' suffix, e.g., 'wear_level_rank'.
    The original string columns are kept for display purposes.

    Parameters
    ----------
    df : DataFrame with original ordinal string columns

    Returns
    -------
    df : DataFrame with additional '_rank' columns for each ordinal attribute
    """
    df = df.copy()

    for col, ordered_cats in ORDINAL_ORDERS.items():
        if col not in df.columns:
            continue

        n_levels = len(ordered_cats)
        rank_map = {
            cat: idx / (n_levels - 1)
            for idx, cat in enumerate(ordered_cats)
        }

        # Map original string to normalized rank; unmapped values → NaN
        rank_col_name = f"{col}_rank"
        df[rank_col_name] = df[col].map(rank_map)

    return df


# ---------------------------------------------------------------------------
# 7. MISSING VALUE ANALYSIS
# ---------------------------------------------------------------------------

def analyze_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a summary of missing values per column.

    Parameters
    ----------
    df : preprocessed DataFrame

    Returns
    -------
    pd.DataFrame with columns: ['column', 'missing_count', 'missing_pct']
    Rows are sorted by missing_pct descending.
    Only columns with at least one missing value are included.
    """
    missing_counts = df[FEATURE_COLS].isnull().sum()
    missing_pct    = missing_counts / len(df) * 100
    summary = pd.DataFrame({
        "column":       missing_counts.index,
        "missing_count": missing_counts.values,
        "missing_pct":  missing_pct.values.round(2),
    })
    summary = summary[summary["missing_count"] > 0]
    return summary.sort_values("missing_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. DUPLICATE DETECTION
# ---------------------------------------------------------------------------

def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove records that share the same machine_id AND record_date.

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df      : deduplicated DataFrame
    n_dupes : number of duplicate rows dropped
    """
    initial_len = len(df)
    subset = [c for c in ["machine_id", "record_date"] if c in df.columns]
    df = df.drop_duplicates(subset=subset, keep="first")
    n_dupes = initial_len - len(df)
    return df, n_dupes


# ---------------------------------------------------------------------------
# 9. TRAIN / VALIDATION / TEST SPLIT
#    Chronological split: oldest records for training, newest for testing.
#    This reflects realistic deployment — we train on historical data and
#    predict on future (unseen) machine states.
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically into train, validation, and test sets.

    Records are sorted by record_date (ascending). The first train_ratio
    fraction goes to training, the next val_ratio fraction to validation,
    and the remainder to test.

    Why chronological? In a real deployment, the model is trained on past
    observations and evaluated on future ones. A random split would allow
    future information to leak into the training set.

    Parameters
    ----------
    df          : preprocessed DataFrame with a valid 'record_date' column
    train_ratio : fraction of data for training (default 0.70)
    val_ratio   : fraction of data for validation (default 0.15)

    Returns
    -------
    df_train, df_val, df_test : three non-overlapping DataFrames
    """
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    # Sort by date; records with NaT dates go last (treated as most recent)
    df = df.sort_values("record_date", na_position="last").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end].copy()
    df_val   = df.iloc[train_end:val_end].copy()
    df_test  = df.iloc[val_end:].copy()

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# 10. TOP-LEVEL PIPELINE FUNCTION
# ---------------------------------------------------------------------------

def load_and_preprocess(
    filepath: str,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    verbose:     bool  = True,
) -> dict[str, Any]:
    """
    Full preprocessing pipeline: load → validate → encode → split.

    Parameters
    ----------
    filepath    : path to cnc_machines.csv
    train_ratio : fraction for training set (default 0.70)
    val_ratio   : fraction for validation set (default 0.15)
    verbose     : if True, print a summary report to stdout

    Returns
    -------
    dict with keys:
        "train"          → training DataFrame (features + target + metadata)
        "val"            → validation DataFrame
        "test"           → test DataFrame
        "feature_config" → feature configuration dict for Gower similarity
        "issues"         → list of validation warning/error strings
        "missing_summary"→ DataFrame summarising missing values in training set
    """
    issues: list[str] = []

    # Step 1: Load
    if verbose:
        print(f"[1/7] Loading dataset from: {filepath}")
    df = load_csv(filepath)
    if verbose:
        print(f"      Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Step 2: Schema validation
    if verbose:
        print("[2/7] Validating schema...")
    schema_errors = validate_schema(df)
    if schema_errors:
        raise ValueError(f"Schema errors: {schema_errors}")

    # Step 3: Category validation (drops invalid rows)
    if verbose:
        print("[3/7] Validating categories...")
    cat_issues, df = validate_categories(df)
    issues.extend(cat_issues)

    # Step 4: Type coercion
    if verbose:
        print("[4/7] Coercing column types...")
    df = coerce_types(df)

    # Step 5: Binary validation (after coercion)
    bin_issues = validate_binary(df)
    issues.extend(bin_issues)

    # Step 6: Numeric range validation and clamping
    if verbose:
        print("[5/7] Validating numeric ranges (and clamping out-of-range values)...")
    range_warnings = validate_numeric_ranges(df)
    issues.extend(range_warnings)
    df = clamp_numeric(df)

    # Step 7: Drop duplicates
    if verbose:
        print("[6/7] Checking for duplicate records...")
    df, n_dupes = drop_duplicates(df)
    if n_dupes > 0:
        issues.append(f"Dropped {n_dupes} duplicate record(s).")

    # Step 8: Ordinal encoding (static — no fitting on data)
    if verbose:
        print("[7/7] Encoding ordinal attributes...")
    df = encode_ordinal(df)

    # Step 9: Chronological split
    df_train, df_val, df_test = chronological_split(df, train_ratio, val_ratio)

    # Step 10: Missing value analysis (on training set only)
    missing_summary = analyze_missing(df_train)

    # Build feature configuration for Gower similarity
    feature_config = build_feature_config()

    # Print report
    if verbose:
        _print_report(df_train, df_val, df_test, issues, missing_summary)

    return {
        "train":           df_train,
        "val":             df_val,
        "test":            df_test,
        "feature_config":  feature_config,
        "issues":          issues,
        "missing_summary": missing_summary,
    }


# ---------------------------------------------------------------------------
# 11. REPORTING
# ---------------------------------------------------------------------------

def _print_report(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    issues:   list[str],
    missing_summary: pd.DataFrame,
) -> None:
    """Print a concise preprocessing summary report."""

    print("\n" + "=" * 60)
    print("  PREPROCESSING REPORT")
    print("=" * 60)

    print(f"\n  Split sizes:")
    print(f"    Train      : {len(df_train)} rows ({len(df_train)/( len(df_train)+len(df_val)+len(df_test))*100:.0f}%)")
    print(f"    Validation : {len(df_val)} rows")
    print(f"    Test       : {len(df_test)} rows")

    print(f"\n  Class distribution (Training set):")
    for label in VALID_CONDITIONS:
        n = (df_train[TARGET_COL] == label).sum()
        pct = n / len(df_train) * 100
        print(f"    {label:<10} {n:>4} ({pct:5.1f}%)")

    if issues:
        print(f"\n  Validation issues ({len(issues)} total):")
        for issue in issues:
            print(f"    [!] {issue}")
    else:
        print("\n  No validation issues found.")

    if not missing_summary.empty:
        print(f"\n  Missing values in training set:")
        for _, row in missing_summary.iterrows():
            print(f"    {row['column']:<28} {int(row['missing_count']):>3} ({row['missing_pct']:.1f}%)")
    else:
        print("\n  No missing values in training set.")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# UTILITY: get the list of rank columns added by encode_ordinal
# ---------------------------------------------------------------------------

def get_rank_col_names() -> list[str]:
    """Return the '_rank' column names created by encode_ordinal."""
    return [f"{col}_rank" for col in ORDINAL_ORDERS]


# ---------------------------------------------------------------------------
# QUICK SELF-TEST when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Resolve path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, "data", "cnc_machines.csv")

    result = load_and_preprocess(data_path, verbose=True)

    print("\nSample training rows (first 3):")
    sample_cols = ["machine_id", "machine_type", "wear_level", "wear_level_rank",
                   "operating_temp_c", "condition"]
    print(result["train"][sample_cols].head(3).to_string(index=False))

    print("\nFeature config keys:", list(result["feature_config"].keys()))
    print("Done.")
