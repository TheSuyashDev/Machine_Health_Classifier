"""
test_evaluation.py
==================
Unit tests for evaluation/evaluate_model.py (Phase 7).

Run:
    pytest tests/test_evaluation.py -v

Tests cover:
    - compute_classification_metrics: perfect accuracy, zero accuracy, per-class metrics
    - Critical safety audit: false negative counting, critical recall
    - Confusion matrix generation and normalization
    - Engineering feature weights configuration
    - Majority class zero-rule baseline
    - Numeric-only baseline
    - Hyperparameter tuning selection
    - Data leakage prevention checks
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

for p in (EVAL_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_model import (
    compute_classification_metrics,
    get_engineering_feature_config,
    evaluate_majority_baseline,
    evaluate_numeric_only_baseline,
    tune_hyperparameters,
)
from data_preprocessing import load_and_preprocess, TARGET_COL
from knn_classifier import CONDITION_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# 1. METRICS COMPUTATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_classification(self):
        y_true = ["Normal", "Warning", "Critical", "Normal", "Warning", "Critical"]
        y_pred = ["Normal", "Warning", "Critical", "Normal", "Warning", "Critical"]
        m = compute_classification_metrics(y_true, y_pred)

        assert m["accuracy"] == 1.0
        assert m["macro_avg"]["precision"] == 1.0
        assert m["macro_avg"]["recall"] == 1.0
        assert m["macro_avg"]["f1_score"] == 1.0
        assert m["safety_audit"]["critical_recall"] == 1.0
        assert m["safety_audit"]["critical_false_negatives"] == 0

    def test_zero_accuracy_classification(self):
        y_true = ["Normal", "Warning", "Critical"]
        y_pred = ["Warning", "Critical", "Normal"]
        m = compute_classification_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.0
        assert m["macro_avg"]["precision"] == 0.0
        assert m["macro_avg"]["recall"] == 0.0
        assert m["macro_avg"]["f1_score"] == 0.0
        assert m["safety_audit"]["critical_recall"] == 0.0
        assert m["safety_audit"]["critical_false_negatives"] == 1

    def test_critical_false_negatives_audit(self):
        # 3 critical machines: 1 predicted correctly, 1 as warning, 1 as normal
        y_true = ["Critical", "Critical", "Critical", "Normal", "Warning"]
        y_pred = ["Critical", "Warning",  "Normal",   "Normal", "Warning"]
        m = compute_classification_metrics(y_true, y_pred)

        audit = m["safety_audit"]
        assert audit["critical_support"] == 3
        assert audit["critical_recall"] == pytest.approx(1 / 3, abs=1e-4)
        assert audit["critical_false_negatives"] == 2
        assert audit["critical_predicted_as_normal"] == 1
        assert audit["critical_predicted_as_warning"] == 1

    def test_confusion_matrix_shape_and_sums(self):
        y_true = ["Normal", "Normal", "Warning", "Critical"]
        y_pred = ["Normal", "Warning", "Warning", "Critical"]
        m = compute_classification_metrics(y_true, y_pred)

        cm = np.array(m["confusion_matrix"]["matrix"])
        assert cm.shape == (3, 3)
        assert cm.sum() == 4
        # First row is Normal: 1 predicted Normal, 1 predicted Warning, 0 Critical
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[0, 2] == 0

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError):
            compute_classification_metrics([], [])


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE WEIGHTS CONFIGURATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineeringWeights:
    def test_engineering_weights_assigned(self):
        base_cfg = {
            "weights": {
                "coolant_failure": 1.0,
                "vibration_mm_s": 1.0,
                "operating_temp_c": 1.0,
                "machine_type": 1.0,
            }
        }
        eng_cfg = get_engineering_feature_config(base_cfg)

        # Original dict should not be mutated
        assert base_cfg["weights"]["coolant_failure"] == 1.0
        # Engineering weights should reflect domain priorities
        assert eng_cfg["weights"]["coolant_failure"] == 2.0
        assert eng_cfg["weights"]["vibration_mm_s"] == 1.8
        assert eng_cfg["weights"]["operating_temp_c"] == 1.6
        assert eng_cfg["weights"]["machine_type"] == 0.8


# ─────────────────────────────────────────────────────────────────────────────
# 3. BASELINE EVALUATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestBaselines:
    @pytest.fixture
    def mock_data(self):
        train = pd.DataFrame({
            "machine_id": ["M01", "M02", "M03", "M04"],
            "operating_temp_c": [50.0, 60.0, 70.0, 80.0],
            "vibration_mm_s": [1.0, 2.0, 5.0, 8.0],
            "spindle_speed_rpm": [1000, 2000, 3000, 4000],
            "operating_hours": [100, 200, 300, 400],
            "power_consumption_kw": [5, 10, 15, 20],
            "tool_usage_hours": [10, 20, 30, 40],
            "avg_load_pct": [30, 40, 50, 60],
            "coolant_temp_c": [25, 30, 35, 40],
            "last_maintenance_days": [10, 20, 30, 40],
            "condition": ["Normal", "Warning", "Warning", "Critical"],
        })
        test = pd.DataFrame({
            "machine_id": ["M05", "M06"],
            "operating_temp_c": [52.0, 78.0],
            "vibration_mm_s": [1.1, 7.5],
            "spindle_speed_rpm": [1100, 3900],
            "operating_hours": [110, 390],
            "power_consumption_kw": [6, 19],
            "tool_usage_hours": [12, 38],
            "avg_load_pct": [32, 58],
            "coolant_temp_c": [26, 39],
            "last_maintenance_days": [12, 38],
            "condition": ["Normal", "Critical"],
        })
        return train, test

    def test_majority_baseline_predicts_mode(self, mock_data):
        train, test = mock_data
        # Warning is the mode in train (appears twice)
        m = evaluate_majority_baseline(train, test)
        assert "Majority Class Baseline (Warning)" in m["baseline_name"]
        assert m["total_samples"] == 2
        # Exactly 0 Warning in test, so accuracy = 0.0
        assert m["accuracy"] == 0.0

    def test_numeric_only_baseline_runs(self, mock_data):
        train, test = mock_data
        m = evaluate_numeric_only_baseline(train, test, k=1)
        assert "Numeric-Only Euclidean" in m["baseline_name"]
        assert m["total_samples"] == 2
        assert 0.0 <= m["accuracy"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. TUNING AND LEAKAGE CHECKS
# ─────────────────────────────────────────────────────────────────────────────

class TestTuningAndLeakage:
    def test_data_leakage_safety_checks(self):
        csv_path = os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv")
        pp = load_and_preprocess(csv_path, verbose=False)
        train_df = pp["train"]
        val_df   = pp["val"]
        test_df  = pp["test"]
        feat_cfg = pp["feature_config"]

        # 1. Target is strictly absent from all feature configuration lists
        for ftype in ("nominal", "binary", "ordinal", "numeric"):
            assert TARGET_COL not in feat_cfg[ftype], f"Target leaked into {ftype}!"

        # 2. Strict chronological ordering (no machine overlap across timestamps)
        max_train_date = train_df["record_date"].max()
        min_val_date   = val_df["record_date"].min()
        max_val_date   = val_df["record_date"].max()
        min_test_date  = test_df["record_date"].min()

        assert max_train_date <= min_val_date, "Train and Validation dates overlap!"
        assert max_val_date <= min_test_date, "Validation and Test dates overlap!"

    def test_tuning_finds_valid_k(self):
        csv_path = os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv")
        pp = load_and_preprocess(csv_path, verbose=False)
        best_params, tuning_df = tune_hyperparameters(
            train_df=pp["train"],
            val_df=pp["val"],
            feature_config=pp["feature_config"],
            k_candidates=[1, 3, 5],
            strategies=["majority"],
        )
        assert best_params["k"] in [1, 3, 5]
        assert best_params["strategy"] == "majority"
        assert not tuning_df.empty
