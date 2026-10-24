"""
main.py
=======
Command-Line End-to-End Demonstration Pipeline.

Orchestrates the entire system:
1. Loads and validates the CNC machine dataset (leakage-free).
2. Allows querying by Machine ID, Row Index, or a Custom Synthetic Sensor Profile.
3. Performs top-k similarity search using the custom Gower engine.
4. Classifies machine health condition using the custom k-NN classifier.
5. Diagnoses underlying sensory evidence (vibration, heat, acoustic chatter, wear).
6. Issues actionable, prioritized predictive maintenance work-orders.

Usage
-----
    # Run interactive demo on default sample machine
    python src/main.py

    # Query a specific machine by ID
    python src/main.py --machine_id M042 --k 5

    # Test an emergency critical situation
    python src/main.py --scenario critical
"""

from __future__ import annotations

import os
import sys
import argparse
import pandas as pd

# Add project root and src/ to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_and_preprocess, TARGET_COL
from knn_classifier import predict, format_prediction
from maintenance_recommendations import generate_recommendation, format_recommendation


# Predefined illustrative testing scenarios for easy demonstration/viva
SCENARIOS: dict[str, dict] = {
    "normal": {
        "machine_id": "SIM_NORMAL_01",
        "machine_type": "Milling",
        "controller_type": "Siemens",
        "cooling_system": "Liquid",
        "lubrication_alert": 0,
        "abnormal_noise": 0,
        "coolant_failure": 0,
        "maintenance_overdue": 0,
        "wear_level": "Low",
        "wear_level_rank": 0.0,
        "maintenance_condition": "Good",
        "maintenance_condition_rank": 0.0,
        "degradation_level": "Minimal",
        "degradation_level_rank": 0.0,
        "operating_temp_c": 52.4,
        "vibration_mm_s": 2.1,
        "spindle_speed_rpm": 6200.0,
        "operating_hours": 12500.0,
        "power_consumption_kw": 12.4,
        "tool_usage_hours": 85.0,
        "avg_load_pct": 55.0,
        "coolant_temp_c": 28.5,
        "last_maintenance_days": 45.0,
    },
    "warning": {
        "machine_id": "SIM_WARNING_01",
        "machine_type": "Turning",
        "controller_type": "Fanuc",
        "cooling_system": "Hybrid",
        "lubrication_alert": 1,
        "abnormal_noise": 1,
        "coolant_failure": 0,
        "maintenance_overdue": 1,
        "wear_level": "High",
        "wear_level_rank": 0.667,
        "maintenance_condition": "Fair",
        "maintenance_condition_rank": 0.5,
        "degradation_level": "Moderate",
        "degradation_level_rank": 0.5,
        "operating_temp_c": 79.5,
        "vibration_mm_s": 12.8,
        "spindle_speed_rpm": 9400.0,
        "operating_hours": 32000.0,
        "power_consumption_kw": 23.5,
        "tool_usage_hours": 380.0,
        "avg_load_pct": 78.0,
        "coolant_temp_c": 58.0,
        "last_maintenance_days": 210.0,
    },
    "critical": {
        "machine_id": "SIM_CRITICAL_01",
        "machine_type": "Milling",
        "controller_type": "Haas",
        "cooling_system": "Liquid",
        "lubrication_alert": 1,
        "abnormal_noise": 1,
        "coolant_failure": 1,
        "maintenance_overdue": 1,
        "wear_level": "Critical",
        "wear_level_rank": 1.0,
        "maintenance_condition": "Poor",
        "maintenance_condition_rank": 1.0,
        "degradation_level": "Severe",
        "degradation_level_rank": 1.0,
        "operating_temp_c": 96.8,
        "vibration_mm_s": 19.4,
        "spindle_speed_rpm": 12800.0,
        "operating_hours": 46500.0,
        "power_consumption_kw": 28.2,
        "tool_usage_hours": 485.0,
        "avg_load_pct": 92.0,
        "coolant_temp_c": 74.5,
        "last_maintenance_days": 340.0,
    },
}


def run_pipeline(
    csv_path: str = "data/cnc_machines.csv",
    machine_id: str | None = None,
    scenario: str | None = None,
    k: int = 5,
    strategy: str = "weighted",
) -> None:
    """
    Execute the full end-to-end diagnosis pipeline on a target machine.
    """
    print("\n" + "=" * 70)
    print("  CNC MACHINE PREDICTIVE MAINTENANCE SYSTEM: CLI PIPELINE")
    print("=" * 70)

    # Step 1: Preprocessing & Data Loading
    print(f"\n[1/4] Loading and preprocessing fleet telemetry from '{csv_path}'...")
    pp = load_and_preprocess(csv_path, verbose=False)
    train_df = pp["train"]
    test_df  = pp["test"]
    feat_cfg = pp["feature_config"]
    print(f"      Reference training fleet: {len(train_df)} machines.")

    # Step 2: Determine Query Machine
    exclude_id = None
    if scenario and scenario.lower() in SCENARIOS:
        print(f"\n[2/4] Query machine configured from synthetic scenario: '{scenario.upper()}'")
        query_record = SCENARIOS[scenario.lower()]
        ground_truth = scenario.capitalize()
    elif machine_id:
        print(f"\n[2/4] Locating machine '{machine_id}' in dataset...")
        # Check test first, then train
        match = test_df[test_df["machine_id"] == machine_id]
        if not match.empty:
            query_record = match.iloc[0]
            ground_truth = query_record.get("condition", "Unknown")
        else:
            match_train = train_df[train_df["machine_id"] == machine_id]
            if match_train.empty:
                raise ValueError(f"Machine ID '{machine_id}' not found in dataset.")
            query_record = match_train.iloc[0]
            ground_truth = query_record.get("condition", "Unknown")
            exclude_id = machine_id  # Prevent self-match
    else:
        # Default to first machine in test partition
        query_record = test_df.iloc[0]
        ground_truth = query_record.get("condition", "Unknown")
        print(f"\n[2/4] Defaulting to first test machine: '{query_record['machine_id']}'")

    print(f"      Machine ID: {query_record.get('machine_id')}")
    print(f"      Type: {query_record.get('machine_type')}, Controller: {query_record.get('controller_type')}")
    print(f"      Known True Condition: {ground_truth}")

    # Step 3: Similarity Search & k-NN Classification
    print(f"\n[3/4] Performing Gower similarity search & k-NN classification (k={k}, {strategy})...")
    prediction_result = predict(
        query          = query_record,
        train_df       = train_df,
        feature_config = feat_cfg,
        k              = k,
        strategy       = strategy,
        exclude_id     = exclude_id,
    )

    print(format_prediction(prediction_result))

    # Step 4: Maintenance Recommendation
    print(f"[4/4] Generating actionable maintenance recommendation...")
    reco = generate_recommendation(
        predicted_condition = prediction_result.predicted_condition,
        machine_record      = query_record,
        neighbor_df         = prediction_result.neighbors,
    )

    print(format_recommendation(reco))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNC Machine Similarity Analysis & Predictive Maintenance Pipeline."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv"),
        help="Path to cnc_machines.csv",
    )
    parser.add_argument(
        "--machine_id",
        type=str,
        default=None,
        help="Machine ID to evaluate (e.g., M005)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["normal", "warning", "critical"],
        default=None,
        help="Run against a predefined diagnostic scenario",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=3,
        help="Number of nearest neighbors (default: 3)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["majority", "weighted"],
        default="majority",
        help="k-NN voting strategy (default: majority)",
    )

    args = parser.parse_args()
    run_pipeline(
        csv_path   = args.csv,
        machine_id = args.machine_id,
        scenario   = args.scenario,
        k          = args.k,
        strategy   = args.strategy,
    )


if __name__ == "__main__":
    main()
