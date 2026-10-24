"""
generate_dataset.py
===================
One-time script to generate the synthetic CNC machine dataset.

Run:
    python data/generate_dataset.py

Output:
    data/cnc_machines.csv  — 300 synthetic machine records

Design notes:
    - Each record represents one snapshot observation of a CNC machine.
    - Feature values are sampled from distributions that reflect realistic
      CNC operating conditions.
    - The `condition` label is deterministically derived from a risk score
      computed from the feature values, as documented in SYNTHETIC_LABEL_RULES.md.
    - A small amount of noise (10% probability, ±1 point) is added to the
      risk score to simulate real-world label uncertainty.
    - Random seed is fixed for reproducibility.

IMPORTANT: This data is synthetic. Do not claim the trained model has been
validated against real industrial failure data.
"""

import random
import csv
import os
from datetime import date, timedelta

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── Dataset size ───────────────────────────────────────────────────────────────
N_MACHINES = 300   # Total number of records to generate

# ── Output path ───────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "cnc_machines.csv")

# ── Nominal attribute value pools ─────────────────────────────────────────────
MACHINE_TYPES   = ["Milling", "Turning", "Drilling"]
CONTROLLER_TYPES = ["Siemens", "Fanuc", "Haas"]
COOLING_SYSTEMS  = ["Air", "Liquid", "Hybrid"]

# ── Ordinal attribute value pools (ordered from best to worst) ────────────────
WEAR_LEVELS          = ["Low", "Medium", "High", "Critical"]
MAINTENANCE_CONDITIONS = ["Good", "Fair", "Poor"]
DEGRADATION_LEVELS   = ["Minimal", "Moderate", "Severe"]

# ── Numeric attribute ranges ───────────────────────────────────────────────────
# Each tuple is (min, max). Values sampled uniformly within range.
NUMERIC_RANGES = {
    "operating_temp_c":     (20.0,  120.0),
    "vibration_mm_s":       (0.1,   25.0),
    "spindle_speed_rpm":    (500,   15000),
    "operating_hours":      (0,     50000),
    "power_consumption_kw": (1.0,   30.0),
    "tool_usage_hours":     (0,     500),
    "avg_load_pct":         (10.0,  100.0),
    "coolant_temp_c":       (15.0,  80.0),
    "last_maintenance_days": (0,    730),
}


def _rand_float(low: float, high: float, decimals: int = 2) -> float:
    """Sample a uniformly distributed float, rounded to `decimals` places."""
    return round(random.uniform(low, high), decimals)


def _rand_int(low: int, high: int) -> int:
    """Sample a uniformly distributed integer in [low, high] inclusive."""
    return random.randint(low, high)


def _sample_binary(prob_of_one: float) -> int:
    """Return 1 with probability `prob_of_one`, else 0."""
    return 1 if random.random() < prob_of_one else 0


def _sample_categorical(pool: list, weights: list = None):
    """
    Sample one value from `pool`.

    Parameters
    ----------
    pool    : list of possible values
    weights : optional list of relative probabilities (same length as pool).
              If None, uniform sampling is used.
    """
    if weights is None:
        return random.choice(pool)
    # random.choices returns a list; we take the first element
    return random.choices(pool, weights=weights, k=1)[0]


def _random_date(start_year: int = 2021, end_year: int = 2024) -> str:
    """Generate a random date string between start_year and end_year."""
    start = date(start_year, 1, 1)
    end   = date(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


def _compute_risk_score(record: dict) -> int:
    """
    Compute an integer risk score for a machine record.

    Rules are documented in data/SYNTHETIC_LABEL_RULES.md.
    A higher score indicates a worse machine condition.

    Parameters
    ----------
    record : dict containing feature values for a single machine

    Returns
    -------
    int : cumulative risk score
    """
    score = 0

    # ── Ordinal wear level ─────────────────────────────────────────────────
    wear = record["wear_level"]
    if   wear == "Critical": score += 4
    elif wear == "High":     score += 2
    elif wear == "Medium":   score += 1
    # "Low" adds 0

    # ── Ordinal degradation level ──────────────────────────────────────────
    deg = record["degradation_level"]
    if   deg == "Severe":   score += 4
    elif deg == "Moderate": score += 2
    # "Minimal" adds 0

    # ── Ordinal maintenance condition ──────────────────────────────────────
    mc = record["maintenance_condition"]
    if   mc == "Poor": score += 3
    elif mc == "Fair": score += 1
    # "Good" adds 0

    # ── Numeric: operating temperature ────────────────────────────────────
    temp = record["operating_temp_c"]
    if   temp > 90: score += 3
    elif temp > 70: score += 1

    # ── Numeric: vibration ────────────────────────────────────────────────
    vib = record["vibration_mm_s"]
    if   vib > 15: score += 3
    elif vib > 8:  score += 1

    # ── Binary fault flags ────────────────────────────────────────────────
    if record["coolant_failure"]     == 1: score += 4
    if record["abnormal_noise"]      == 1: score += 2
    if record["lubrication_alert"]   == 1: score += 2
    if record["maintenance_overdue"] == 1: score += 2

    # ── Numeric: days since last maintenance ───────────────────────────────
    days = record["last_maintenance_days"]
    if   days > 365: score += 2
    elif days > 180: score += 1

    # ── Numeric: tool usage ────────────────────────────────────────────────
    tool = record["tool_usage_hours"]
    if   tool > 400: score += 2
    elif tool > 200: score += 1

    # ── Numeric: load percentage ───────────────────────────────────────────
    if record["avg_load_pct"] > 90: score += 2

    # ── Numeric: coolant temperature ───────────────────────────────────────
    if record["coolant_temp_c"] > 65: score += 2

    # ── Numeric: operating hours (machine age proxy) ───────────────────────
    if record["operating_hours"] > 40000: score += 1

    return score


def _assign_condition(risk_score: int) -> str:
    """
    Map a numeric risk score to a condition label.

    Thresholds (documented in SYNTHETIC_LABEL_RULES.md):
        0  – 4  → Normal
        5  – 9  → Warning
        10+     → Critical
    """
    if   risk_score >= 10: return "Critical"
    elif risk_score >= 5:  return "Warning"
    else:                  return "Normal"


def _maybe_add_noise(score: int, prob: float = 0.10) -> int:
    """
    With probability `prob`, randomly adjust the score by ±1.

    This simulates real-world label uncertainty (e.g., an inspector
    might rate the same machine slightly differently on different days).
    """
    if random.random() < prob:
        direction = random.choice([-1, 1])
        score = max(0, score + direction)   # score cannot go below 0
    return score


def _maybe_null(value, prob_null: float = 0.03):
    """
    Return None with probability `prob_null`, otherwise return `value`.

    This introduces realistic missing values (~3% rate per numeric column).
    """
    if random.random() < prob_null:
        return None
    return value


def generate_record(machine_idx: int) -> dict:
    """
    Generate one synthetic CNC machine observation record.

    Parameters
    ----------
    machine_idx : int used to build the machine_id (e.g., 1 → "CNC-001")

    Returns
    -------
    dict : a single record with all 22 columns
    """
    machine_id = f"CNC-{machine_idx:03d}"
    record_date = _random_date()

    # ── Sample nominal attributes ──────────────────────────────────────────
    machine_type    = _sample_categorical(MACHINE_TYPES)
    controller_type = _sample_categorical(CONTROLLER_TYPES)
    cooling_system  = _sample_categorical(COOLING_SYSTEMS)

    # ── Sample binary attributes ───────────────────────────────────────────
    # The probabilities below reflect realistic failure rates:
    #   lubrication_alert ~20% of machines have an active alert
    #   abnormal_noise    ~15%
    #   coolant_failure   ~8%  (rarer — treated as asymmetric)
    #   maintenance_overdue ~30%
    lubrication_alert   = _sample_binary(0.20)
    abnormal_noise      = _sample_binary(0.15)
    coolant_failure     = _sample_binary(0.08)
    maintenance_overdue = _sample_binary(0.30)

    # ── Sample ordinal attributes ──────────────────────────────────────────
    # Weights skew toward healthier states (realistic fleet: most machines OK)
    wear_level = _sample_categorical(
        WEAR_LEVELS,
        weights=[0.45, 0.30, 0.17, 0.08]   # Low most common, Critical rarest
    )
    maintenance_condition = _sample_categorical(
        MAINTENANCE_CONDITIONS,
        weights=[0.55, 0.30, 0.15]          # Good most common, Poor least
    )
    degradation_level = _sample_categorical(
        DEGRADATION_LEVELS,
        weights=[0.50, 0.33, 0.17]          # Minimal most common, Severe least
    )

    # ── Sample numeric attributes ──────────────────────────────────────────
    # Biased toward normal operating ranges:
    #   temperature: most machines run 30-70 °C (not at max 120)
    #   vibration:   most < 10 mm/s under normal load
    #   tool usage:  uniformly distributed across tool life
    #   maintenance: most services done within 6 months
    operating_temp_c      = _rand_float(30.0, 90.0)      # Normal ops range
    vibration_mm_s        = _rand_float(0.1,  14.0)      # Biased below alarm
    spindle_speed_rpm     = _rand_float(1000, 12000, decimals=0)
    operating_hours       = _rand_float(0,    40000, decimals=1)
    power_consumption_kw  = _rand_float(2.0,  22.0)
    tool_usage_hours      = _rand_float(0,    400,   decimals=1)
    avg_load_pct          = _rand_float(20.0, 85.0)
    coolant_temp_c        = _rand_float(20.0, 65.0)
    last_maintenance_days = _rand_float(0,    400,   decimals=0)

    # Build record dict (needed for risk score computation before applying nulls)
    record = {
        "machine_id":             machine_id,
        "record_date":            record_date,
        "machine_type":           machine_type,
        "controller_type":        controller_type,
        "cooling_system":         cooling_system,
        "lubrication_alert":      lubrication_alert,
        "abnormal_noise":         abnormal_noise,
        "coolant_failure":        coolant_failure,
        "maintenance_overdue":    maintenance_overdue,
        "wear_level":             wear_level,
        "maintenance_condition":  maintenance_condition,
        "degradation_level":      degradation_level,
        "operating_temp_c":       operating_temp_c,
        "vibration_mm_s":         vibration_mm_s,
        "spindle_speed_rpm":      spindle_speed_rpm,
        "operating_hours":        operating_hours,
        "power_consumption_kw":   power_consumption_kw,
        "tool_usage_hours":       tool_usage_hours,
        "avg_load_pct":           avg_load_pct,
        "coolant_temp_c":         coolant_temp_c,
        "last_maintenance_days":  last_maintenance_days,
    }

    # ── Compute risk score and assign label ────────────────────────────────
    # We compute the label BEFORE introducing missing values so that the
    # risk score is always computed on complete feature data.
    score = _compute_risk_score(record)
    score = _maybe_add_noise(score)
    condition = _assign_condition(score)

    # ── Introduce missing values in numeric columns AFTER labeling ─────────
    # This prevents the label from being affected by missing values, while
    # still giving us realistic incomplete data for preprocessing practice.
    record["operating_temp_c"]      = _maybe_null(operating_temp_c)
    record["vibration_mm_s"]        = _maybe_null(vibration_mm_s)
    record["spindle_speed_rpm"]     = _maybe_null(spindle_speed_rpm)
    record["operating_hours"]       = _maybe_null(operating_hours)
    record["power_consumption_kw"]  = _maybe_null(power_consumption_kw)
    record["tool_usage_hours"]      = _maybe_null(tool_usage_hours)
    record["avg_load_pct"]          = _maybe_null(avg_load_pct)
    record["coolant_temp_c"]        = _maybe_null(coolant_temp_c)
    record["last_maintenance_days"] = _maybe_null(last_maintenance_days)

    record["condition"] = condition
    return record


def generate_dataset(n: int = N_MACHINES) -> list[dict]:
    """
    Generate `n` synthetic CNC machine records.

    Parameters
    ----------
    n : number of records to generate

    Returns
    -------
    list of dicts, one per machine record
    """
    return [generate_record(i + 1) for i in range(n)]


def save_to_csv(records: list[dict], output_path: str) -> None:
    """
    Write a list of record dicts to a CSV file.

    Parameters
    ----------
    records     : list of dicts (one per machine)
    output_path : absolute or relative path for the CSV file
    """
    if not records:
        raise ValueError("No records to save.")

    fieldnames = list(records[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"[OK] Saved {len(records)} records to: {output_path}")


def print_summary(records: list[dict]) -> None:
    """Print a basic summary of the generated dataset to stdout."""
    from collections import Counter

    conditions = [r["condition"] for r in records]
    counts = Counter(conditions)

    print("\n-- Dataset Summary -------------------------------------------")
    print(f"  Total records   : {len(records)}")
    print(f"  Columns         : {len(records[0])}")
    print(f"\n  Condition distribution:")
    for label in ["Normal", "Warning", "Critical"]:
        n = counts.get(label, 0)
        pct = n / len(records) * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:<10} {n:>4} records ({pct:5.1f}%)  {bar}")

    # Count missing values per column
    null_counts = {}
    for col in records[0]:
        nulls = sum(1 for r in records if r[col] is None)
        if nulls > 0:
            null_counts[col] = nulls

    if null_counts:
        print(f"\n  Missing values (numeric columns only):")
        for col, n in null_counts.items():
            pct = n / len(records) * 100
            print(f"    {col:<28} {n:>3} ({pct:.1f}%)")
    else:
        print("\n  No missing values.")

    print("--------------------------------------------------------------\n")


if __name__ == "__main__":
    print("[INFO] Generating synthetic CNC machine dataset...")
    print(f"       Random seed  : {RANDOM_SEED}")
    print(f"       Records      : {N_MACHINES}")
    print(f"       Output path  : {OUTPUT_PATH}")

    records = generate_dataset(N_MACHINES)
    print_summary(records)
    save_to_csv(records, OUTPUT_PATH)
