# Synthetic Label Generation Rules

## Purpose

This document transparently describes how the `condition` target labels were
assigned to the synthetic CNC machine dataset. These rules are for academic
purposes only. They do NOT represent certified engineering thresholds or
validated real-world failure models.

## Label Categories

| Label | Meaning |
|---|---|
| Normal | Machine operating within acceptable parameters |
| Warning | Degradation or anomaly detected; requires attention |
| Critical | Severe condition; immediate inspection required |

## Scoring System

Each machine record receives a numeric risk score based on its sensor readings
and status flags. The score is then mapped to one of the three condition labels.

### Score Contribution Rules

| Condition | Score Added |
|---|---|
| wear_level == 'Critical' | +4 |
| wear_level == 'High' | +2 |
| wear_level == 'Medium' | +1 |
| degradation_level == 'Severe' | +4 |
| degradation_level == 'Moderate' | +2 |
| maintenance_condition == 'Poor' | +3 |
| maintenance_condition == 'Fair' | +1 |
| operating_temp_c > 90 | +3 |
| operating_temp_c > 70 | +1 |
| vibration_mm_s > 15 | +3 |
| vibration_mm_s > 8 | +1 |
| coolant_failure == 1 | +4 |
| abnormal_noise == 1 | +2 |
| lubrication_alert == 1 | +2 |
| maintenance_overdue == 1 | +2 |
| last_maintenance_days > 365 | +2 |
| last_maintenance_days > 180 | +1 |
| tool_usage_hours > 400 | +2 |
| tool_usage_hours > 200 | +1 |
| avg_load_pct > 90 | +2 |
| coolant_temp_c > 65 | +2 |
| operating_hours > 40000 | +1 |

### Label Assignment Thresholds

| Total Score | Assigned Label |
|---|---|
| 0 – 4 | Normal |
| 5 – 9 | Warning |
| 10+ | Critical |

A small amount of random noise (±1 point with 10% probability) is applied
before thresholding to simulate real-world labeling uncertainty.

## Limitations

1. These rules are invented for teaching purposes — they are plausible but
   not validated against real machine failure data.
2. The model trained on this data cannot be used in production without
   validation against real inspection outcomes.
3. Class distribution may be skewed (e.g., more Normal than Critical records).
   This is realistic but requires attention during evaluation.
4. The simulation assumes all features are independent — real machines have
   complex correlations not captured here.
