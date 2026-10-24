"""
app.py
======
CNC Care: Machine Health & Predictive Maintenance Dashboard.

A clean, intuitive, and professional frontend for monitoring CNC machine health,
calculating Gower mixed-attribute similarity, predicting health conditions (Normal,
Warning, Critical), and generating actionable prescriptive maintenance guidance.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Setup python path to include project root, src/, and evaluation/
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
for p in (PROJECT_ROOT, SRC_DIR, EVAL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data_preprocessing import (
    load_and_preprocess,
    VALID_NOMINAL,
    VALID_BINARY,
    ORDINAL_ORDERS,
    NUMERIC_RANGES,
    TARGET_COL,
)
from src.knn_classifier import predict, CONDITION_ORDER
from src.maintenance_recommendations import (
    generate_recommendation,
    SAFETY_DISCLAIMER,
)
from evaluation.evaluate_model import get_engineering_feature_config


# ─────────────────────────────────────────────────────────────────────────────
# 1. PAGE CONFIGURATION & CLEAN PROFESSIONAL INDUSTRIAL THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CNC Care - Machine Health",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom minimal styling: light background, clean white cards, blue accent, subtle borders
st.markdown(
    """
    <style>
    /* Global application styling */
    .stApp {
        background-color: #f8fafc;
        color: #1e293b;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Clean Card Container */
    .cnc-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }

    /* Summary Metric Cards */
    .summary-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1px solid #e2e8f0;
    }
    .summary-normal {
        border-left: 4px solid #16a34a;
        background-color: #f0fdf4;
    }
    .summary-warning {
        border-left: 4px solid #d97706;
        background-color: #fffbeb;
    }
    .summary-critical {
        border-left: 4px solid #dc2626;
        background-color: #fef2f2;
    }

    .summary-count {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1;
        margin-top: 4px;
    }
    .summary-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }

    /* Condition Status Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-normal {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
    }

    /* Action Banner */
    .action-box {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
    }

    /* Evidence list */
    .evidence-item {
        background-color: #f1f5f9;
        border-left: 3px solid #64748b;
        padding: 6px 12px;
        margin: 4px 0;
        border-radius: 0 4px 4px 0;
        font-size: 0.9rem;
    }

    /* Disclaimer box */
    .disclaimer-text {
        font-size: 0.76rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 12px;
        margin-top: 24px;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING (REUSING EXISTING ZERO-LEAKAGE PIPELINE)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """Load preprocessed partitions and raw fleet data."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv")
    pp = load_and_preprocess(csv_path, verbose=False)
    raw_df = pd.read_csv(csv_path)
    return pp, raw_df


pp_data, raw_fleet_df = load_data()
train_df = pp_data["train"]
base_feat_cfg = pp_data["feature_config"]
eng_feat_cfg  = get_engineering_feature_config(base_feat_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 3. HEADER & MINIMAL NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

# Navigation state
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "Dashboard"
if "selected_machine_for_analysis" not in st.session_state:
    st.session_state["selected_machine_for_analysis"] = None

# Top Header Bar
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 4px;'>
            <span style='font-size: 1.8rem;'>⚙️</span>
            <div>
                <h2 style='margin: 0; font-size: 1.45rem; font-weight: 700; color: #0f172a;'>CNC Care</h2>
                <p style='margin: 0; font-size: 0.88rem; color: #64748b;'>Machine Health & Predictive Maintenance</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_col2:
    nav_choice = st.radio(
        "Navigation",
        ["Dashboard", "Analyze Machine"],
        index=0 if st.session_state["current_view"] == "Dashboard" else 1,
        horizontal=True,
        label_visibility="collapsed",
    )
    if nav_choice != st.session_state["current_view"]:
        st.session_state["current_view"] = nav_choice
        st.rerun()

st.markdown("<hr style='margin: 8px 0 20px 0; border: none; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state["current_view"] == "Dashboard":
    # ── A. Summary Cards (Normal, Warning, Critical counts) ─────────────────
    n_normal   = int((raw_fleet_df["condition"] == "Normal").sum())
    n_warning  = int((raw_fleet_df["condition"] == "Warning").sum())
    n_critical = int((raw_fleet_df["condition"] == "Critical").sum())

    card_col1, card_col2, card_col3 = st.columns(3)

    with card_col1:
        st.markdown(
            f"""
            <div class='summary-card summary-normal'>
                <div>
                    <div class='summary-title'>Normal Condition</div>
                    <div class='summary-count' style='color: #16a34a;'>{n_normal}</div>
                </div>
                <div style='font-size: 1.8rem;'>🟢</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col2:
        st.markdown(
            f"""
            <div class='summary-card summary-warning'>
                <div>
                    <div class='summary-title'>Warning Condition</div>
                    <div class='summary-count' style='color: #d97706;'>{n_warning}</div>
                </div>
                <div style='font-size: 1.8rem;'>🟡</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card_col3:
        st.markdown(
            f"""
            <div class='summary-card summary-critical'>
                <div>
                    <div class='summary-title'>Critical Condition</div>
                    <div class='summary-count' style='color: #dc2626;'>{n_critical}</div>
                </div>
                <div style='font-size: 1.8rem;'>🔴</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")  # Spacing

    # ── B. Primary Action Bar ────────────────────────────────────────────────
    action_col1, action_col2 = st.columns([3, 1])
    with action_col1:
        st.markdown(
            "<h3 style='margin:0; font-size:1.15rem; color:#1e293b; font-weight:600;'>Machines Requiring Attention</h3>"
            "<p style='margin:0; font-size:0.82rem; color:#64748b;'>Equipment flagged with elevated vibration, high temperatures, or maintenance alerts.</p>",
            unsafe_allow_html=True,
        )
    with action_col2:
        if st.button("🔍 Analyze Machine", type="primary", use_container_width=True):
            st.session_state["current_view"] = "Analyze Machine"
            st.session_state["selected_machine_for_analysis"] = None
            st.rerun()

    # ── C. Machines Requiring Attention Table ────────────────────────────────
    attention_df = raw_fleet_df[raw_fleet_df["condition"].isin(["Critical", "Warning"])].copy()

    if attention_df.empty:
        st.success("✅ All machines are operating within normal engineering limits.")
    else:
        # Sort Critical first, then Warning
        attention_df["sort_order"] = attention_df["condition"].map({"Critical": 0, "Warning": 1})
        attention_df = attention_df.sort_values(["sort_order", "operating_temp_c"], ascending=[True, False]).drop(columns=["sort_order"])

        # Create concise recommended action text based on rules
        def quick_action(row):
            if row["condition"] == "Critical":
                if row.get("coolant_failure") == 1:
                    return "Immediate Stop (Coolant Failure)"
                elif row.get("vibration_mm_s", 0) >= 15.0:
                    return "Immediate Stop (Severe Vibration)"
                return "Immediate Controlled Stop & LOTO"
            else:
                if row.get("abnormal_noise") == 1:
                    return "Inspect Spindle Bearing (Noise Alert)"
                elif row.get("wear_level") in ("High", "Critical"):
                    return "Inspect Cutting Tool & Wear"
                return "Schedule Inspection within 48h"

        attention_df["Recommended Action"] = attention_df.apply(quick_action, axis=1)

        # Format display columns
        display_df = attention_df[[
            "machine_id", "condition", "operating_temp_c", "vibration_mm_s", "Recommended Action"
        ]].head(15).copy()

        display_df.rename(columns={
            "machine_id": "Machine ID",
            "condition": "Condition",
            "operating_temp_c": "Temperature (°C)",
            "vibration_mm_s": "Vibration (mm/s)",
        }, inplace=True)

        display_df["Temperature (°C)"] = display_df["Temperature (°C)"].map(lambda x: f"{x:.1f}")
        display_df["Vibration (mm/s)"] = display_df["Vibration (mm/s)"].map(lambda x: f"{x:.2f}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption("Showing up to 15 prioritized machines. Click **Analyze Machine** above to diagnose any machine in detail.")


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2: SIMPLIFIED MACHINE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

elif st.session_state["current_view"] == "Analyze Machine":
    st.markdown(
        "<h3 style='margin:0; font-size:1.2rem; color:#1e293b; font-weight:600;'>Machine Health Analysis</h3>"
        "<p style='margin:0 0 16px 0; font-size:0.85rem; color:#64748b;'>Select an existing machine or enter current sensor readings to evaluate condition.</p>",
        unsafe_allow_html=True,
    )

    # 1. Input Mode Selection (Simple & Clear)
    input_type = st.radio(
        "Input Mode:",
        ["Select Existing Machine from Fleet", "Enter Custom Sensor Readings"],
        horizontal=True,
    )

    query_dict: dict = {}
    exclude_id = None

    if input_type == "Select Existing Machine from Fleet":
        all_ids = raw_fleet_df["machine_id"].tolist()
        default_index = 0
        if st.session_state.get("selected_machine_for_analysis") in all_ids:
            default_index = all_ids.index(st.session_state["selected_machine_for_analysis"])

        selected_id = st.selectbox("Select Machine ID to Analyze", all_ids, index=default_index)
        row = raw_fleet_df[raw_fleet_df["machine_id"] == selected_id].iloc[0]
        query_dict = row.to_dict()
        # Ensure normalized rank features are present
        query_dict["wear_level_rank"] = ORDINAL_ORDERS["wear_level"].index(row["wear_level"]) / 3.0
        query_dict["maintenance_condition_rank"] = ORDINAL_ORDERS["maintenance_condition"].index(row["maintenance_condition"]) / 2.0
        query_dict["degradation_level_rank"] = ORDINAL_ORDERS["degradation_level"].index(row["degradation_level"]) / 2.0
        exclude_id = selected_id

        # Quick preview of selected machine
        with st.expander("👁️ View Machine Attributes", expanded=False):
            p1, p2, p3 = st.columns(3)
            with p1:
                st.write(f"**Type:** {row['machine_type']}")
                st.write(f"**Controller:** {row['controller_type']}")
                st.write(f"**Cooling:** {row['cooling_system']}")
            with p2:
                st.write(f"**Temperature:** {row['operating_temp_c']:.1f} °C")
                st.write(f"**Vibration:** {row['vibration_mm_s']:.2f} mm/s")
                st.write(f"**Wear Level:** {row['wear_level']}")
            with p3:
                st.write(f"**Coolant Failure:** {'Yes' if row['coolant_failure'] == 1 else 'No'}")
                st.write(f"**Abnormal Noise:** {'Yes' if row['abnormal_noise'] == 1 else 'No'}")
                st.write(f"**Operating Hours:** {row['operating_hours']:.0f} hrs")

    else:
        # Clean, structured grouped inputs
        st.markdown("##### 1. Machine & Sensor Readings")
        col1, col2, col3 = st.columns(3)

        with col1:
            q_type = st.selectbox("Machine Type", VALID_NOMINAL["machine_type"])
            q_ctrl = st.selectbox("Controller", VALID_NOMINAL["controller_type"])
            q_cool = st.selectbox("Cooling System", VALID_NOMINAL["cooling_system"])
            q_temp = st.number_input("Spindle Temperature (°C)", min_value=20.0, max_value=120.0, value=65.0, step=1.0)
            q_vib  = st.number_input("Vibration (mm/s)", min_value=0.1, max_value=25.0, value=4.5, step=0.1)

        with col2:
            q_rpm   = st.number_input("Spindle Speed (RPM)", min_value=500.0, max_value=15000.0, value=7500.0, step=250.0)
            q_pwr   = st.number_input("Power Consumption (kW)", min_value=1.0, max_value=30.0, value=12.0, step=0.5)
            q_load  = st.number_input("Average Load (%)", min_value=10.0, max_value=100.0, value=50.0, step=5.0)
            q_ctemp = st.number_input("Coolant Temperature (°C)", min_value=15.0, max_value=80.0, value=30.0, step=1.0)
            q_thrs  = st.number_input("Tool Usage Hours", min_value=0.0, max_value=500.0, value=120.0, step=10.0)

        with col3:
            q_wear  = st.selectbox("Tool Wear Level", ORDINAL_ORDERS["wear_level"])
            q_mcond = st.selectbox("Maintenance Condition", ORDINAL_ORDERS["maintenance_condition"])
            q_deg   = st.selectbox("Structural Degradation", ORDINAL_ORDERS["degradation_level"])
            q_ophrs = st.number_input("Total Operating Hours", min_value=100.0, max_value=50000.0, value=14000.0, step=500.0)
            q_days  = st.number_input("Days Since Service", min_value=0.0, max_value=730.0, value=60.0, step=10.0)

        st.markdown("##### 2. Active Sensor Alarms")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            q_cfail = st.checkbox("Coolant System Failure")
        with a2:
            q_noise = st.checkbox("Abnormal Mechanical Noise")
        with a3:
            q_lube  = st.checkbox("Lubrication Alert")
        with a4:
            q_over  = st.checkbox("Maintenance Overdue")

        query_dict = {
            "machine_id": "MANUAL_QUERY",
            "machine_type": q_type, "controller_type": q_ctrl, "cooling_system": q_cool,
            "lubrication_alert": int(q_lube), "abnormal_noise": int(q_noise),
            "coolant_failure": int(q_cfail), "maintenance_overdue": int(q_over),
            "wear_level": q_wear,
            "wear_level_rank": ORDINAL_ORDERS["wear_level"].index(q_wear) / 3.0,
            "maintenance_condition": q_mcond,
            "maintenance_condition_rank": ORDINAL_ORDERS["maintenance_condition"].index(q_mcond) / 2.0,
            "degradation_level": q_deg,
            "degradation_level_rank": ORDINAL_ORDERS["degradation_level"].index(q_deg) / 2.0,
            "operating_temp_c": q_temp, "vibration_mm_s": q_vib, "spindle_speed_rpm": q_rpm,
            "operating_hours": q_ophrs, "power_consumption_kw": q_pwr, "tool_usage_hours": q_thrs,
            "avg_load_pct": q_load, "coolant_temp_c": q_ctemp, "last_maintenance_days": q_days,
        }

    # Collapsed Advanced Settings (Hidden from normal users as requested)
    with st.expander("⚙️ Advanced Algorithm Settings (Optional)", expanded=False):
        k_val = st.slider("Nearest Neighbors (k)", min_value=1, max_value=9, value=3, step=2)
        strat_val = st.selectbox("Voting Strategy", ["majority", "weighted"], index=0)
        use_eng = st.checkbox("Apply Domain Engineering Weights", value=False)
        active_config = eng_feat_cfg if use_eng else base_feat_cfg

    # Prominent Primary Action Button
    st.write("")
    analyze_clicked = st.button("🚀 Analyze Machine", type="primary", use_container_width=True)

    # Automatically trigger analysis on load or button click
    if analyze_clicked or query_dict:
        # Run prediction
        res = predict(
            query          = query_dict,
            train_df       = train_df,
            feature_config = active_config if "active_config" in locals() else base_feat_cfg,
            k              = k_val if "k_val" in locals() else 3,
            strategy       = strat_val if "strat_val" in locals() else "majority",
            exclude_id     = exclude_id,
        )

        reco = generate_recommendation(
            predicted_condition = res.predicted_condition,
            machine_record      = query_dict,
            neighbor_df         = res.neighbors,
        )

        st.markdown("<hr style='margin: 20px 0; border: none; border-top: 2px solid #e2e8f0;' />", unsafe_allow_html=True)
        st.markdown("### 📋 Analysis Results")

        # ── Result 1: Predicted Machine Condition ───────────────────────────
        cond = res.predicted_condition
        badge_style = "badge-normal" if cond == "Normal" else ("badge-warning" if cond == "Warning" else "badge-critical")
        icon = "🟢" if cond == "Normal" else ("🟡" if cond == "Warning" else "🔴")

        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.markdown(
                f"""
                <div style='background-color:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:16px; text-align:center;'>
                    <div style='font-size:0.8rem; font-weight:600; text-transform:uppercase; color:#64748b;'>Predicted Condition</div>
                    <div style='margin-top:8px;'>
                        <span class='badge {badge_style}' style='font-size:1.3rem; padding:6px 16px;'>
                            {icon} {cond.upper()}
                        </span>
                    </div>
                    <div style='font-size:0.8rem; color:#64748b; margin-top:8px;'>Urgency: <b>{reco.urgency_level}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_c2:
            # ── Result 2: Short Explanation ─────────────────────────────────
            st.markdown("##### 📝 Summary Explanation")
            # Present concise evidence summary
            clean_evidence = [e for e in reco.supporting_evidence if "normal limits" not in e.lower()]
            if clean_evidence:
                ev_str = "; ".join(clean_evidence[:3])
                st.write(f"The machine exhibits **{cond}** state driven by: {ev_str}.")
            else:
                st.write("All operational parameters and alarms are currently within standard baseline specifications.")

            # ── Result 3: Recommended Action ────────────────────────────────
            st.markdown(
                f"""
                <div class='action-box'>
                    <div style='font-weight:600; color:#1d4ed8; font-size:0.85rem;'>RECOMMENDED ACTION ({reco.time_horizon}):</div>
                    <div style='color:#1e293b; font-size:1rem; font-weight:600; margin-top:4px;'>{reco.primary_action}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Result 4: Most Similar Machines ─────────────────────────────────
        st.markdown("##### 👥 Most Similar Machines in Fleet")
        sim_df = res.neighbors[[
            "rank", "machine_id", "similarity", "condition", "operating_temp_c", "vibration_mm_s", "wear_level"
        ]].copy()

        sim_df.rename(columns={
            "rank": "Rank",
            "machine_id": "Machine ID",
            "similarity": "Similarity Score",
            "condition": "Condition",
            "operating_temp_c": "Temp (°C)",
            "vibration_mm_s": "Vibration (mm/s)",
            "wear_level": "Wear",
        }, inplace=True)

        sim_df["Similarity Score"] = sim_df["Similarity Score"].map(lambda s: f"{s:.1%}")
        sim_df["Temp (°C)"] = sim_df["Temp (°C)"].map(lambda t: f"{t:.1f}")
        sim_df["Vibration (mm/s)"] = sim_df["Vibration (mm/s)"].map(lambda v: f"{v:.2f}")

        st.dataframe(sim_df, use_container_width=True, hide_index=True)

        # ── Result 5: Technical Details (Optional Expandable) ────────────────
        with st.expander("🔬 Technical Model Details (k-NN Voting Breakdown)", expanded=False):
            t1, t2 = st.columns(2)
            with t1:
                st.write(f"**Neighbors Used:** {res.k_used} (Requested k={res.k_requested})")
                st.write(f"**Voting Strategy:** {res.voting_strategy.capitalize()}")
                st.write("**Vote Distribution:**")
                for c_label in CONDITION_ORDER:
                    v_raw = res.vote_distribution.get(c_label, 0)
                    v_pct = res.vote_shares.get(c_label, 0) * 100
                    st.write(f"- {c_label}: {v_raw:.1f} votes ({v_pct:.1f}%)")
            with t2:
                st.write("**Detailed Diagnostic Checklist:**")
                for s_idx, step in enumerate(reco.action_checklist, 1):
                    st.write(f"{s_idx}. {step}")

        # Safety Disclaimer
        st.markdown(
            f"<div class='disclaimer-text'><b>Safety Notice:</b> {SAFETY_DISCLAIMER}</div>",
            unsafe_allow_html=True,
        )
