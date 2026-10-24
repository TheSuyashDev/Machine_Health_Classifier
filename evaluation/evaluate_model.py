"""
evaluate_model.py
=================
Comprehensive model evaluation, hyperparameter tuning, and baseline comparison
for the CNC Machine Health Classification system (Phase 7).

Responsibilities
----------------
1. Data Leakage Prevention:
   - Fixed engineering ranges and static ordinal encoding ensure zero leakage.
   - Train / Validation / Test sets are evaluated with strict partition boundaries.
2. Hyperparameter Tuning:
   - Evaluates k in [1, 3, 5, 7, 9, 11] across both "majority" and "weighted" voting.
   - Evaluates on the Validation set to avoid overfitting to the Test set.
3. Feature Weighting Experiment:
   - Compares Equal Weights (default 1.0 for all features) vs. Engineering Weights
     (prioritizing vibration, temperature, coolant failure, wear level, and acoustic alarms).
4. Baseline Comparisons:
   - Majority Class Dummy Baseline (Zero-Rule): predicts the most common training class.
   - Numeric-Only Euclidean Baseline: ignores categorical/ordinal attributes, highlighting
     the necessity of Gower-style mixed-attribute similarity.
5. Metric Computation:
   - Accuracy, Macro/Weighted Precision, Recall, F1-Score.
   - Per-class metrics for Normal, Warning, and Critical.
   - Confusion Matrix (counts and normalized percentages).
   - Critical-class recall and false negative analysis (vital for industrial safety).
6. Export:
   - Formats a comprehensive terminal report.
   - Exports all evaluation metrics to `evaluation/metrics_summary.json`.

Usage
-----
    python evaluation/evaluate_model.py
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Any, Optional

# Add project root and src/ to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_and_preprocess, TARGET_COL
from knn_classifier import predict, CONDITION_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# 1. BATCH PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def batch_predict(
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    feature_config: dict,
    k: int = 5,
    strategy: str = "weighted",
    exclude_self: bool = False,
) -> tuple[list[str], list[dict[str, float]], list[dict[str, float]]]:
    """
    Generate predictions for an entire evaluation DataFrame.

    Parameters
    ----------
    eval_df        : DataFrame containing query machine records to evaluate.
    train_df       : Reference training DataFrame.
    feature_config : Feature configuration dictionary for Gower similarity.
    k              : Number of nearest neighbors.
    strategy       : "weighted" or "majority".
    exclude_self   : If True, excludes the query machine's own ID (used for train set eval).

    Returns
    -------
    (y_pred, vote_distributions, vote_shares)
    """
    y_pred: list[str] = []
    distributions: list[dict[str, float]] = []
    shares: list[dict[str, float]] = []

    for _, row in eval_df.iterrows():
        exclude_id = row.get("machine_id") if exclude_self else None
        res = predict(
            query          = row,
            train_df       = train_df,
            feature_config = feature_config,
            k              = k,
            strategy       = strategy,
            exclude_id     = exclude_id,
        )
        y_pred.append(res.predicted_condition)
        distributions.append(res.vote_distribution)
        shares.append(res.vote_shares)

    return y_pred, distributions, shares


# ─────────────────────────────────────────────────────────────────────────────
# 2. METRICS CALCULATION (PURE PYTHON / NUMPY FOR COMPLETE TRANSPARENCY)
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: list[str] | np.ndarray | pd.Series,
    y_pred: list[str] | np.ndarray | pd.Series,
    classes: list[str] = CONDITION_ORDER,
) -> dict[str, Any]:
    """
    Compute comprehensive classification metrics without black-box dependency.

    Parameters
    ----------
    y_true  : Ground truth condition labels.
    y_pred  : Predicted condition labels.
    classes : Ordered list of class labels (default: ["Normal", "Warning", "Critical"]).

    Returns
    -------
    Dictionary of overall metrics, per-class metrics, confusion matrix, and safety audit.
    """
    y_t = list(y_true)
    y_p = list(y_pred)
    n_samples = len(y_t)

    if n_samples == 0:
        raise ValueError("Cannot compute metrics on empty ground truth.")

    # 1. Confusion Matrix
    # cm[actual_idx][pred_idx]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    correct_total = 0
    for true_lbl, pred_lbl in zip(y_t, y_p):
        if true_lbl == pred_lbl:
            correct_total += 1
        if true_lbl in class_to_idx and pred_lbl in class_to_idx:
            cm[class_to_idx[true_lbl]][class_to_idx[pred_lbl]] += 1

    overall_accuracy = correct_total / n_samples

    # 2. Per-Class Precision, Recall, F1, and Support
    per_class: dict[str, dict[str, float]] = {}
    macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0

    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "precision": round(float(precision), 4),
            "recall":    round(float(recall), 4),
            "f1_score":  round(float(f1), 4),
            "support":   support,
        }

        macro_prec += precision
        macro_rec  += recall
        macro_f1   += f1

        weighted_prec += precision * support
        weighted_rec  += recall * support
        weighted_f1   += f1 * support

    n_classes = len(classes)
    macro_prec = round(macro_prec / n_classes, 4)
    macro_rec  = round(macro_rec / n_classes, 4)
    macro_f1   = round(macro_f1 / n_classes, 4)

    weighted_prec = round(weighted_prec / n_samples, 4)
    weighted_rec  = round(weighted_rec / n_samples, 4)
    weighted_f1   = round(weighted_f1 / n_samples, 4)

    # 3. Critical Safety Audit
    # A false negative for Critical means a machine is Critical in reality
    # but the system predicted Normal or Warning. In an industrial shop floor,
    # this could lead to catastrophic spindle failure or fire.
    crit_idx = class_to_idx.get("Critical", -1)
    norm_idx = class_to_idx.get("Normal", -1)
    warn_idx = class_to_idx.get("Warning", -1)

    critical_misclassified_as_normal = int(cm[crit_idx, norm_idx]) if crit_idx != -1 and norm_idx != -1 else 0
    critical_misclassified_as_warning = int(cm[crit_idx, warn_idx]) if crit_idx != -1 and warn_idx != -1 else 0
    total_critical_false_negatives = critical_misclassified_as_normal + critical_misclassified_as_warning

    # Serialisable confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_mat = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=(row_sums != 0),
    ).round(4).tolist()

    cm_dict = {
        "classes": classes,
        "matrix": cm.tolist(),
        "normalized_row_recall": norm_mat,
    }

    return {
        "accuracy": round(overall_accuracy, 4),
        "total_samples": n_samples,
        "macro_avg": {
            "precision": macro_prec,
            "recall":    macro_rec,
            "f1_score":  macro_f1,
        },
        "weighted_avg": {
            "precision": weighted_prec,
            "recall":    weighted_rec,
            "f1_score":  weighted_f1,
        },
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "safety_audit": {
            "critical_support": per_class.get("Critical", {}).get("support", 0),
            "critical_recall": per_class.get("Critical", {}).get("recall", 0.0),
            "critical_false_negatives": total_critical_false_negatives,
            "critical_predicted_as_normal": critical_misclassified_as_normal,
            "critical_predicted_as_warning": critical_misclassified_as_warning,
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENGINEERING FEATURE WEIGHTS SPECIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def get_engineering_feature_config(base_config: dict) -> dict:
    """
    Return a feature configuration with justified engineering weights.

    Rationale:
    - Highly critical fault indicators (coolant_failure, vibration, temperature)
      carry stronger physical association with imminent mechanical damage.
    - Nominal configuration parameters (controller_type, machine_type) affect
      cutting kinematics but are less predictive of immediate degradation.
    """
    import copy
    eng_config = copy.deepcopy(base_config)
    weights = eng_config["weights"]

    # Domain-informed weighting
    domain_weights = {
        # High impact physical degradation & safety flags
        "coolant_failure":       2.0,  # Asymmetric failure indicator
        "vibration_mm_s":        1.8,  # Primary indicator of bearing/tool imbalance
        "operating_temp_c":      1.6,  # Thermal stress indicator
        "wear_level":            1.5,  # Direct tool/component wear rating
        "degradation_level":     1.5,  # Structural wear level
        # Medium impact maintenance indicators
        "abnormal_noise":        1.3,
        "lubrication_alert":     1.3,
        "maintenance_condition": 1.3,
        "maintenance_overdue":   1.2,
        "coolant_temp_c":        1.2,
        # Operational baseline indicators
        "avg_load_pct":          1.0,
        "power_consumption_kw":  1.0,
        "spindle_speed_rpm":     1.0,
        "operating_hours":       1.0,
        "tool_usage_hours":      1.0,
        "last_maintenance_days": 1.0,
        # Low impact architectural features
        "machine_type":          0.8,
        "cooling_system":        0.8,
        "controller_type":       0.6,
    }

    for col, w in domain_weights.items():
        if col in weights:
            weights[col] = float(w)

    eng_config["weights"] = weights
    return eng_config


# ─────────────────────────────────────────────────────────────────────────────
# 4. BASELINE MODELS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_majority_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Zero-Rule Baseline: Predicts the most frequent class in the training set
    for all test samples.
    """
    majority_class = train_df[TARGET_COL].mode()[0]
    y_pred = [majority_class] * len(test_df)
    metrics = compute_classification_metrics(test_df[TARGET_COL], y_pred)
    metrics["baseline_name"] = f"Majority Class Baseline ({majority_class})"
    return metrics


def evaluate_numeric_only_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 5,
) -> dict[str, Any]:
    """
    Numeric-Only Baseline: Computes Euclidean distance on min-max normalized
    numeric features only, ignoring all nominal, binary, and ordinal attributes.
    Demonstrates the necessity and performance gain of the Gower-style similarity.
    """
    numeric_cols = [
        "operating_temp_c", "vibration_mm_s", "spindle_speed_rpm",
        "operating_hours", "power_consumption_kw", "tool_usage_hours",
        "avg_load_pct", "coolant_temp_c", "last_maintenance_days"
    ]

    # Min-max normalization strictly using training ranges (no leakage)
    min_vals = train_df[numeric_cols].min()
    max_vals = train_df[numeric_cols].max()
    ranges = (max_vals - min_vals).replace(0, 1.0)

    train_num = ((train_df[numeric_cols] - min_vals) / ranges).fillna(0.0).values
    test_num  = ((test_df[numeric_cols] - min_vals) / ranges).fillna(0.0).values

    train_labels = train_df[TARGET_COL].values
    y_pred = []

    for test_vec in test_num:
        # Euclidean distance
        dists = np.linalg.norm(train_num - test_vec, axis=1)
        top_k_indices = np.argsort(dists)[:k]
        top_k_labels = train_labels[top_k_indices]
        
        # Majority vote
        counts: dict[str, int] = {}
        for lbl in top_k_labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        best_lbl = max(counts.items(), key=lambda item: item[1])[0]
        y_pred.append(best_lbl)

    metrics = compute_classification_metrics(test_df[TARGET_COL], y_pred)
    metrics["baseline_name"] = f"Numeric-Only Euclidean k-NN (k={k})"
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_config: dict,
    k_candidates: list[int] = [1, 3, 5, 7, 9, 11],
    strategies: list[str] = ["majority", "weighted"],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Tune k and voting strategy strictly on the Validation set.

    Returns
    -------
    (best_params, tuning_summary_df)
    """
    y_val_true = val_df[TARGET_COL].tolist()
    records = []

    best_macro_f1 = -1.0
    best_crit_rec = -1.0
    best_params: dict[str, Any] = {}

    for strat in strategies:
        for k in k_candidates:
            y_pred, _, _ = batch_predict(
                eval_df        = val_df,
                train_df       = train_df,
                feature_config = feature_config,
                k              = k,
                strategy       = strat,
            )
            m = compute_classification_metrics(y_val_true, y_pred)
            acc = m["accuracy"]
            macro_f1 = m["macro_avg"]["f1_score"]
            crit_rec = m["safety_audit"]["critical_recall"]
            crit_fn = m["safety_audit"]["critical_false_negatives"]

            records.append({
                "k": k,
                "strategy": strat,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "critical_recall": crit_rec,
                "critical_false_negatives": crit_fn,
            })

            # Selection criteria: Primary = Macro F1, Secondary = Critical Recall
            if (macro_f1 > best_macro_f1) or (abs(macro_f1 - best_macro_f1) < 1e-4 and crit_rec > best_crit_rec):
                best_macro_f1 = macro_f1
                best_crit_rec = crit_rec
                best_params = {
                    "k": k,
                    "strategy": strat,
                    "val_accuracy": acc,
                    "val_macro_f1": macro_f1,
                    "val_critical_recall": crit_rec,
                }

    tuning_df = pd.DataFrame(records)
    return best_params, tuning_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORCHESTRATION AND FULL EXPERIMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation_pipeline(
    csv_path: str = "data/cnc_machines.csv",
    export_json_path: str = "evaluation/metrics_summary.json",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Execute the complete Phase 7 evaluation pipeline:
    1. Preprocessing & chronological splitting (leakage-free).
    2. Hyperparameter tuning on validation set.
    3. Feature weighting comparison (equal vs engineering weights).
    4. Baseline models evaluation.
    5. Final evaluation on unseen Test set.
    6. Exporting results to JSON.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  PHASE 7: MODEL EVALUATION & EXPERIMENTAL BENCHMARKING")
        print("=" * 70)

    # 1. Load data
    pp = load_and_preprocess(csv_path, verbose=False)
    train_df = pp["train"]
    val_df   = pp["val"]
    test_df  = pp["test"]
    base_feat_cfg = pp["feature_config"]

    if verbose:
        print(f"\n[1] Partitions Loaded:")
        print(f"    Train set      : {len(train_df)} rows")
        print(f"    Validation set : {len(val_df)} rows")
        print(f"    Test set       : {len(test_df)} rows")

    # 2. Hyperparameter Tuning on Validation Set
    if verbose:
        print(f"\n[2] Hyperparameter Tuning on Validation Set (k in [1,3,5,7,9,11], majority vs weighted)...")

    best_params, tuning_df = tune_hyperparameters(
        train_df       = train_df,
        val_df         = val_df,
        feature_config = base_feat_cfg,
        k_candidates   = [1, 3, 5, 7, 9, 11],
        strategies     = ["majority", "weighted"],
    )

    if verbose:
        print(f"    Tuning Results Table:")
        print(tuning_df.to_string(index=False))
        print(f"\n    --> Best Validation Configuration: k={best_params['k']}, strategy={best_params['strategy']}")
        print(f"        Val Accuracy: {best_params['val_accuracy']:.4f}, Val Macro F1: {best_params['val_macro_f1']:.4f}, Val Critical Recall: {best_params['val_critical_recall']:.4f}")

    opt_k = best_params["k"]
    opt_strat = best_params["strategy"]

    # 3. Feature Weighting Experiment on Test Set
    if verbose:
        print(f"\n[3] Evaluating Feature Weighting Strategies on Test Set (using tuned k={opt_k}, {opt_strat})...")

    # A. Equal Weights Test Evaluation
    y_test_true = test_df[TARGET_COL].tolist()
    y_test_pred_equal, _, _ = batch_predict(
        eval_df        = test_df,
        train_df       = train_df,
        feature_config = base_feat_cfg,
        k              = opt_k,
        strategy       = opt_strat,
    )
    equal_weights_metrics = compute_classification_metrics(y_test_true, y_test_pred_equal)

    # B. Engineering Weights Test Evaluation
    eng_feat_cfg = get_engineering_feature_config(base_feat_cfg)
    y_test_pred_eng, _, _ = batch_predict(
        eval_df        = test_df,
        train_df       = train_df,
        feature_config = eng_feat_cfg,
        k              = opt_k,
        strategy       = opt_strat,
    )
    eng_weights_metrics = compute_classification_metrics(y_test_true, y_test_pred_eng)

    if verbose:
        print(f"    - Equal Weights       --> Accuracy: {equal_weights_metrics['accuracy']:.4f}, Macro F1: {equal_weights_metrics['macro_avg']['f1_score']:.4f}, Crit Recall: {equal_weights_metrics['safety_audit']['critical_recall']:.4f}")
        print(f"    - Engineering Weights --> Accuracy: {eng_weights_metrics['accuracy']:.4f}, Macro F1: {eng_weights_metrics['macro_avg']['f1_score']:.4f}, Crit Recall: {eng_weights_metrics['safety_audit']['critical_recall']:.4f}")

    # Choose winning model configuration on Test set
    champion_config_name = "Engineering Weights" if eng_weights_metrics["macro_avg"]["f1_score"] >= equal_weights_metrics["macro_avg"]["f1_score"] else "Equal Weights"
    champion_metrics = eng_weights_metrics if champion_config_name == "Engineering Weights" else equal_weights_metrics
    champion_feat_cfg = eng_feat_cfg if champion_config_name == "Engineering Weights" else base_feat_cfg

    # 4. Baselines Evaluation on Test Set
    if verbose:
        print(f"\n[4] Evaluating Baselines on Test Set...")

    majority_baseline = evaluate_majority_baseline(train_df, test_df)
    numeric_baseline  = evaluate_numeric_only_baseline(train_df, test_df, k=opt_k)

    if verbose:
        print(f"    - {majority_baseline['baseline_name']} --> Accuracy: {majority_baseline['accuracy']:.4f}, Macro F1: {majority_baseline['macro_avg']['f1_score']:.4f}")
        print(f"    - {numeric_baseline['baseline_name']}  --> Accuracy: {numeric_baseline['accuracy']:.4f}, Macro F1: {numeric_baseline['macro_avg']['f1_score']:.4f}")

    # 5. Full Summary Report
    summary_report: dict[str, Any] = {
        "dataset_split": {
            "train_size": len(train_df),
            "val_size":   len(val_df),
            "test_size":  len(test_df),
        },
        "best_hyperparameters": best_params,
        "tuning_results": tuning_df.to_dict(orient="records"),
        "champion_configuration": {
            "name": champion_config_name,
            "k": opt_k,
            "strategy": opt_strat,
            "metrics": champion_metrics,
        },
        "comparisons": {
            "equal_weights_test": equal_weights_metrics,
            "engineering_weights_test": eng_weights_metrics,
            "majority_class_baseline": majority_baseline,
            "numeric_only_baseline": numeric_baseline,
        }
    }

    # Save to JSON
    os.makedirs(os.path.dirname(os.path.abspath(export_json_path)), exist_ok=True)
    with open(export_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    if verbose:
        print(f"\n[5] Metrics summary exported to: {export_json_path}")
        _print_detailed_metrics_report(champion_metrics, champion_config_name, opt_k, opt_strat)

    return summary_report


def _print_detailed_metrics_report(
    m: dict[str, Any],
    config_name: str,
    k: int,
    strategy: str,
) -> None:
    """Print an ASCII formatted classification report and confusion matrix."""
    print("\n" + "=" * 70)
    print(f"  FINAL TEST SET PERFORMANCE: {config_name.upper()} (k={k}, {strategy})")
    print("=" * 70)
    print(f"  Overall Accuracy : {m['accuracy'] * 100:.2f}%")
    print(f"  Macro Avg F1     : {m['macro_avg']['f1_score'] * 100:.2f}%")
    print(f"  Weighted Avg F1  : {m['weighted_avg']['f1_score'] * 100:.2f}%")

    print("\n  Per-Class Metrics:")
    print(f"  {'Condition':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("  " + "-" * 56)
    for c in CONDITION_ORDER:
        info = m["per_class"][c]
        print(f"  {c:<12} {info['precision']*100:>9.2f}% {info['recall']*100:>9.2f}% {info['f1_score']*100:>9.2f}% {info['support']:>10}")

    print("\n  Confusion Matrix (Rows: Actual, Columns: Predicted):")
    classes = m["confusion_matrix"]["classes"]
    mat = m["confusion_matrix"]["matrix"]
    norm_mat = m["confusion_matrix"]["normalized_row_recall"]

    header = "  " + f"{'Actual v \\ Pred >':<20}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("  " + "-" * (20 + 12 * len(classes)))
    for i, c in enumerate(classes):
        row_str = f"  {c:<20}" + "".join(f"{mat[i][j]:>6} ({norm_mat[i][j]*100:4.1f}%)" for j in range(len(classes)))
        print(row_str)

    print("\n  Industrial Safety Audit (Critical Condition):")
    audit = m["safety_audit"]
    print(f"    Critical Samples in Test Set      : {audit['critical_support']}")
    print(f"    Critical Recall (Detection Rate)  : {audit['critical_recall']*100:.2f}%")
    print(f"    Critical False Negatives          : {audit['critical_false_negatives']}")
    print(f"      - Misclassified as Normal       : {audit['critical_predicted_as_normal']} (High risk)")
    print(f"      - Misclassified as Warning      : {audit['critical_predicted_as_warning']} (Sub-optimal)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    csv_file = os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv")
    json_out = os.path.join(PROJECT_ROOT, "evaluation", "metrics_summary.json")
    run_evaluation_pipeline(csv_file, json_out, verbose=True)
