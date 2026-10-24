# CNC Machine Health Dataset — Data Dictionary

## Overview

This dictionary describes every column in `cnc_machines.csv`.  
The dataset contains synthetic records representing CNC machine operating
states. Labels were generated using transparent simulation rules documented
in `SYNTHETIC_LABEL_RULES.md`.

**Total columns:** 22  
**Target column:** `condition`  
**ID / metadata columns:** `machine_id`, `record_date`  
**Feature columns:** 19 (used as model inputs)

---

## Column Reference Table

| # | Column Name | Role | Data Type | Attribute Type | Unit / Values | Valid Range | Description |
|---|---|---|---|---|---|---|---|
| 1 | `machine_id` | Metadata | string | — | e.g. `CNC-001` | Unique per machine | Unique machine identifier. Never used as a feature. |
| 2 | `record_date` | Metadata | date string | — | `YYYY-MM-DD` | 2021-01-01 to present | Date the observation was recorded. Used for chronological splitting. |
| 3 | `machine_type` | Feature | string | **Nominal** | `Milling`, `Turning`, `Drilling` | One of three values | Category of CNC operation. No ordering exists between types. |
| 4 | `controller_type` | Feature | string | **Nominal** | `Siemens`, `Fanuc`, `Haas` | One of three values | CNC controller brand. No ordering. |
| 5 | `cooling_system` | Feature | string | **Nominal** | `Air`, `Liquid`, `Hybrid` | One of three values | Cooling mechanism in use. Affects thermal behavior. |
| 6 | `lubrication_alert` | Feature | integer | **Binary (symmetric)** | `0`=No alert, `1`=Alert | {0, 1} | Lubrication system issue flag. |
| 7 | `abnormal_noise` | Feature | integer | **Binary (symmetric)** | `0`=No noise, `1`=Detected | {0, 1} | Acoustic anomaly indicator. |
| 8 | `coolant_failure` | Feature | integer | **Binary (asymmetric)** | `0`=Normal, `1`=Failure | {0, 1} | Coolant system failure flag. Rare event — shared absence is meaningful but two machines both failing is most informative. |
| 9 | `maintenance_overdue` | Feature | integer | **Binary (symmetric)** | `0`=Current, `1`=Overdue | {0, 1} | Whether scheduled maintenance has been missed. |
| 10 | `wear_level` | Feature | string | **Ordinal** | `Low`, `Medium`, `High`, `Critical` | Ordered (4 levels) | Physical wear on cutting tools. Low < Medium < High < Critical. |
| 11 | `maintenance_condition` | Feature | string | **Ordinal** | `Good`, `Fair`, `Poor` | Ordered (3 levels) | Overall maintenance state from last inspection. Good < Fair < Poor. |
| 12 | `degradation_level` | Feature | string | **Ordinal** | `Minimal`, `Moderate`, `Severe` | Ordered (3 levels) | Physical degradation of machine components. Minimal < Moderate < Severe. |
| 13 | `operating_temp_c` | Feature | float | **Numeric (interval)** | °C | 20.0 – 120.0 | Spindle area operating temperature. Higher = more thermal stress. |
| 14 | `vibration_mm_s` | Feature | float | **Numeric (interval)** | mm/s | 0.1 – 25.0 | RMS vibration level. High vibration suggests imbalance or bearing wear. |
| 15 | `spindle_speed_rpm` | Feature | float | **Numeric (interval)** | RPM | 500 – 15000 | Current spindle rotation speed. |
| 16 | `operating_hours` | Feature | float | **Numeric (interval)** | hours | 0 – 50000 | Cumulative operating hours since installation. |
| 17 | `power_consumption_kw` | Feature | float | **Numeric (interval)** | kW | 1.0 – 30.0 | Real-time power draw. Abnormal usage can signal motor issues. |
| 18 | `tool_usage_hours` | Feature | float | **Numeric (interval)** | hours | 0 – 500 | Hours since last tool change. |
| 19 | `avg_load_pct` | Feature | float | **Numeric (interval)** | % | 10.0 – 100.0 | Average load on servo system as % of rated capacity. |
| 20 | `coolant_temp_c` | Feature | float | **Numeric (interval)** | °C | 15.0 – 80.0 | Temperature of coolant fluid. High value reduces cooling efficiency. |
| 21 | `last_maintenance_days` | Feature | float | **Numeric (interval)** | days | 0 – 730 | Days since last scheduled maintenance. |
| 22 | `condition` | **Target** | string | — | `Normal`, `Warning`, `Critical` | One of three values | Machine health label. Never used as an input feature. |

---

## Attribute Type Summary

| Attribute Type | Count | Columns |
|---|---|---|
| Nominal | 3 | `machine_type`, `controller_type`, `cooling_system` |
| Binary (symmetric) | 3 | `lubrication_alert`, `abnormal_noise`, `maintenance_overdue` |
| Binary (asymmetric) | 1 | `coolant_failure` |
| Ordinal | 3 | `wear_level`, `maintenance_condition`, `degradation_level` |
| Numeric (interval) | 9 | `operating_temp_c`, `vibration_mm_s`, `spindle_speed_rpm`, `operating_hours`, `power_consumption_kw`, `tool_usage_hours`, `avg_load_pct`, `coolant_temp_c`, `last_maintenance_days` |
| Metadata / ID | 2 | `machine_id`, `record_date` |
| Target | 1 | `condition` |

---

## Ordinal Rank Encoding Reference

For ordinal attributes, similarity uses normalized ranks in [0, 1].
Formula: normalized_rank = raw_rank / (n_levels - 1)

### `wear_level`
| Category | Raw Rank | Normalized Rank |
|---|---|---|
| Low | 0 | 0.000 |
| Medium | 1 | 0.333 |
| High | 2 | 0.667 |
| Critical | 3 | 1.000 |

### `maintenance_condition`
| Category | Raw Rank | Normalized Rank |
|---|---|---|
| Good | 0 | 0.000 |
| Fair | 1 | 0.500 |
| Poor | 2 | 1.000 |

### `degradation_level`
| Category | Raw Rank | Normalized Rank |
|---|---|---|
| Minimal | 0 | 0.000 |
| Moderate | 1 | 0.500 |
| Severe | 2 | 1.000 |

---

## Numeric Feature Ranges (Fixed Engineering Constants)

> **Important:** These ranges are fixed from engineering domain knowledge.
> They are NOT computed from the training data to prevent data leakage.

| Column | Min | Max | Range (R_k) used in similarity |
|---|---|---|---|
| `operating_temp_c` | 20.0 | 120.0 | 100.0 |
| `vibration_mm_s` | 0.1 | 25.0 | 24.9 |
| `spindle_speed_rpm` | 500 | 15000 | 14500 |
| `operating_hours` | 0 | 50000 | 50000 |
| `power_consumption_kw` | 1.0 | 30.0 | 29.0 |
| `tool_usage_hours` | 0 | 500 | 500 |
| `avg_load_pct` | 10.0 | 100.0 | 90.0 |
| `coolant_temp_c` | 15.0 | 80.0 | 65.0 |
| `last_maintenance_days` | 0 | 730 | 730 |

---

## Binary Attribute Semantics: Symmetric vs Asymmetric

**Symmetric binary** (lubrication_alert, abnormal_noise, maintenance_overdue):
Both machines sharing a 0 (neither has the issue) counts as a match, just as
both sharing a 1 (both have the issue) counts as a match. All four combinations
(0-0, 0-1, 1-0, 1-1) are handled uniformly: similarity = 1 if equal, else 0.

**Asymmetric binary** (coolant_failure):
Coolant failure is a rare fault event. The asymmetric treatment means:
- 0-0 (both normal): counted as a match
- 1-1 (both failing): counted as a match (strongest shared signal)
- 0-1 or 1-0 (one failing, one not): counted as a mismatch

For this dataset, we treat coolant_failure as asymmetric but still count 0-0
as similar, since a shared normal operating state is informative.

---

## Features Used for Similarity vs. Reserved Columns

| Column | Used for Similarity? | Reason |
|---|---|---|
| `machine_id` | No | Metadata identifier |
| `record_date` | No | Used only for train/test splitting |
| All 19 feature columns | Yes | Engineering characteristics |
| `condition` | No (Target) | What we are predicting |

---

*Project: CNC Machine Health Classifier*
*Data type: Synthetic — see SYNTHETIC_LABEL_RULES.md for label generation rules*
