"""
test_recommendations.py
=======================
Unit tests for src/maintenance_recommendations.py (Phase 8).

Run:
    pytest tests/test_recommendations.py -v

Tests cover:
    - Condition mapping: Normal (Routine), Warning (Scheduled), Critical (Immediate)
    - Diagnostic evidence extraction across vibration, temp, coolant, wear, alerts
    - Dynamic checklist injection
    - Custom policy and threshold overrides
    - Historical neighbor context summarization
    - Safety disclaimer and report formatting
    - Error handling on invalid conditions
"""

import os
import sys
import pytest
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from maintenance_recommendations import (
    generate_recommendation,
    format_recommendation,
    extract_sensor_evidence,
    summarize_neighbor_history,
    MaintenanceRecommendation,
    SAFETY_DISCLAIMER,
)


class TestRecommendationConditions:
    def test_normal_condition_routing(self):
        rec = {"machine_id": "M_NORM", "vibration_mm_s": 2.1, "operating_temp_c": 45.0}
        reco = generate_recommendation("Normal", rec)

        assert reco.predicted_condition == "Normal"
        assert reco.urgency_level == "ROUTINE"
        assert "30 Days" in reco.time_horizon
        assert "Routine Operations" in reco.primary_action
        assert any("within normal limits" in ev for ev in reco.supporting_evidence)

    def test_warning_condition_routing(self):
        rec = {"machine_id": "M_WARN", "vibration_mm_s": 12.0, "operating_temp_c": 78.0}
        reco = generate_recommendation("Warning", rec)

        assert reco.predicted_condition == "Warning"
        assert reco.urgency_level == "SCHEDULED"
        assert "24-48" in reco.time_horizon
        assert "Inspection" in reco.primary_action

    def test_critical_condition_routing(self):
        rec = {"machine_id": "M_CRIT", "vibration_mm_s": 19.5, "operating_temp_c": 98.0, "coolant_failure": 1}
        reco = generate_recommendation("Critical", rec)

        assert reco.predicted_condition == "Critical"
        assert reco.urgency_level == "IMMEDIATE"
        assert "Lockout-Tagout" in reco.primary_action
        assert any("[CRITICAL CHECK]" in step for step in reco.action_checklist)

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown condition 'Catastrophic'"):
            generate_recommendation("Catastrophic", {})


class TestEvidenceExtraction:
    def test_vibration_critical_and_warning(self):
        # Above 15 mm/s -> Critical
        ev_crit = extract_sensor_evidence({"vibration_mm_s": 16.5})
        assert any("Severe Vibration" in ev for ev in ev_crit)

        # Between 10 and 15 mm/s -> Warning
        ev_warn = extract_sensor_evidence({"vibration_mm_s": 11.2})
        assert any("Elevated Vibration" in ev for ev in ev_warn)

    def test_temperature_critical_and_warning(self):
        ev_crit = extract_sensor_evidence({"operating_temp_c": 92.0})
        assert any("Spindle Overheating" in ev for ev in ev_crit)

        ev_warn = extract_sensor_evidence({"operating_temp_c": 80.0})
        assert any("Elevated Spindle Temp" in ev for ev in ev_warn)

    def test_coolant_failure_and_temperature(self):
        ev = extract_sensor_evidence({"coolant_failure": 1, "coolant_temp_c": 72.0})
        assert any("Coolant delivery system failure" in e for e in ev)
        assert any("Coolant Overheating" in e for e in ev)

    def test_acoustic_and_lubrication_alarms(self):
        ev = extract_sensor_evidence({"abnormal_noise": 1, "lubrication_alert": 1})
        assert any("Acoustic Sensor" in e for e in ev)
        assert any("Lubrication Alarm" in e for e in ev)

    def test_tool_wear_evidence(self):
        ev_crit = extract_sensor_evidence({"wear_level": "Critical"})
        assert any("Tool Wear Level: Critical" in e for e in ev_crit)

        ev_high = extract_sensor_evidence({"wear_level": "High"})
        assert any("Tool Wear Level: High" in e for e in ev_high)


class TestCustomPoliciesAndThresholds:
    def test_custom_threshold_override(self):
        # Default vibration warning is 10.0, we override to 4.0
        custom_th = {"vibration_warning_mm_s": 4.0}
        ev = extract_sensor_evidence({"vibration_mm_s": 5.5}, thresholds=custom_th)
        assert any("Elevated Vibration: 5.50 mm/s" in e for e in ev)

    def test_custom_policy_override(self):
        custom_pol = {
            "Critical": {
                "urgency_level": "RED_ALERT",
                "time_horizon": "15 Minutes",
            }
        }
        reco = generate_recommendation("Critical", {"machine_id": "M99"}, custom_policy=custom_pol)
        assert reco.urgency_level == "RED_ALERT"
        assert reco.time_horizon == "15 Minutes"


class TestNeighborContextAndFormatting:
    def test_neighbor_summary_generation(self):
        neighbor_df = pd.DataFrame({
            "machine_id": ["M01", "M02", "M03"],
            "condition": ["Warning", "Warning", "Critical"],
            "similarity": [0.94, 0.91, 0.88],
            "coolant_failure": [0, 1, 1],
            "abnormal_noise": [1, 0, 1],
        })
        summary = summarize_neighbor_history(neighbor_df)
        assert "2 Warning, 1 Critical" in summary
        assert "Closest match was M01" in summary
        assert "2/3 neighbors had coolant failure" in summary

    def test_empty_neighbor_handled(self):
        summary = summarize_neighbor_history(None)
        assert "No historical nearest neighbor context available" in summary

    def test_format_recommendation_contains_key_sections(self):
        reco = generate_recommendation("Critical", {"machine_id": "CNC-400", "vibration_mm_s": 17.0})
        report = format_recommendation(reco)

        assert "PREDICTIVE MAINTENANCE WORK-ORDER ADVICE: MACHINE CNC-400" in report
        assert "CRITICAL" in report
        assert "[IMMEDIATE]" in report
        assert "Diagnostic Physical Evidence:" in report
        assert "Recommended Maintenance Checklist:" in report
        assert "DISCLAIMER" in report
