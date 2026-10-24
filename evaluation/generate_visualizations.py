"""
generate_visualizations.py
===========================
Generates publication-quality figures and plots for the CNC Machine Health
Classification evaluation report (Phase 7).

Plots Generated:
1. `confusion_matrix.png`:
   Heatmap of test set confusion matrix with raw counts and recall percentages.
2. `k_tuning_curve.png`:
   Validation accuracy, Macro F1, and Critical Recall across values of k
   comparing Majority vs. Weighted voting.
3. `per_class_metrics.png`:
   Grouped bar chart showing Precision, Recall, and F1 for Normal, Warning, Critical.
4. `feature_weighting_comparison.png`:
   Comparison of Equal Weights vs. Domain-informed Engineering Weights.
5. `model_vs_baselines.png`:
   Benchmarking the Champion Gower k-NN against Majority Baseline and Numeric-Only Baseline.

Output Directory:
    evaluation/figures/

Usage:
    python evaluation/generate_visualizations.py
"""

from __future__ import annotations

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="sans-serif")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
FIG_DIR = os.path.join(EVAL_DIR, "figures")
METRICS_JSON = os.path.join(EVAL_DIR, "metrics_summary.json")

# Visual Styling Configuration
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})

# Class Colors
CLASS_COLORS = {
    "Normal": "#2ecc71",    # Emerald Green
    "Warning": "#f39c12",   # Industrial Amber
    "Critical": "#e74c3c",  # Safety Red
}


def load_metrics_data(json_path: str = METRICS_JSON) -> dict:
    """Load precomputed metrics from JSON, or invoke evaluation pipeline if absent."""
    if not os.path.exists(json_path):
        from evaluate_model import run_evaluation_pipeline
        csv_file = os.path.join(PROJECT_ROOT, "data", "cnc_machines.csv")
        return run_evaluation_pipeline(csv_file, json_path, verbose=False)

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(metrics_data: dict, output_dir: str = FIG_DIR) -> str:
    """Plot an annotated confusion matrix heatmap."""
    champ = metrics_data["champion_configuration"]
    cm_info = champ["metrics"]["confusion_matrix"]
    classes = cm_info["classes"]
    cm = np.array(cm_info["matrix"])
    norm_cm = np.array(cm_info["normalized_row_recall"])

    # Create formatted annotations: "Count\n(Recall%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(len(classes)):
        for j in range(len(classes)):
            pct = norm_cm[i, j] * 100
            annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(7, 6))
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=classes,
            yticklabels=classes,
            linewidths=1.2,
            linecolor="#e0e0e0",
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
    else:
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Machine Count", weight="bold")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, annot[i, j], ha="center", va="center", color=text_color, fontweight="bold", fontsize=11)

    ax.set_title(
        f"Test Set Confusion Matrix\n({champ['name']}, k={champ['k']}, {champ['strategy'].capitalize()} Voting)",
        pad=15,
        weight="bold"
    )
    ax.set_xlabel("Predicted Health Condition", labelpad=10, weight="bold")
    ax.set_ylabel("Actual Ground Truth", labelpad=10, weight="bold")

    out_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. HYPERPARAMETER TUNING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_k_tuning_curves(metrics_data: dict, output_dir: str = FIG_DIR) -> str:
    """Plot Validation Accuracy, Macro F1, and Critical Recall across k values."""
    tuning_records = metrics_data["tuning_results"]
    df = pd.DataFrame(tuning_records)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # Plot 1: Macro F1 vs k
    for strat, color, marker in [("majority", "#2980b9", "o"), ("weighted", "#8e44ad", "s")]:
        sub = df[df["strategy"] == strat].sort_values("k")
        ax1.plot(
            sub["k"], sub["macro_f1"] * 100,
            marker=marker, linewidth=2.2, markersize=8,
            label=f"{strat.capitalize()} Voting", color=color
        )

    best_k = metrics_data["best_hyperparameters"]["k"]
    ax1.axvline(best_k, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Selected k={best_k}")
    ax1.set_title("Validation Macro F1 vs. k", weight="bold")
    ax1.set_xlabel("Number of Neighbors (k)", weight="bold")
    ax1.set_ylabel("Macro F1-Score (%)", weight="bold")
    ax1.set_xticks(sorted(df["k"].unique()))
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Critical Recall vs k
    for strat, color, marker in [("majority", "#2980b9", "o"), ("weighted", "#8e44ad", "s")]:
        sub = df[df["strategy"] == strat].sort_values("k")
        ax2.plot(
            sub["k"], sub["critical_recall"] * 100,
            marker=marker, linewidth=2.2, markersize=8,
            label=f"{strat.capitalize()} Voting", color=color
        )

    ax2.axvline(best_k, color="#e74c3c", linestyle="--", alpha=0.7, label=f"Selected k={best_k}")
    ax2.set_title("Validation Critical Recall (Safety) vs. k", weight="bold")
    ax2.set_xlabel("Number of Neighbors (k)", weight="bold")
    ax2.set_ylabel("Critical Recall (%)", weight="bold")
    ax2.set_xticks(sorted(df["k"].unique()))
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Hyperparameter Tuning on Validation Set", fontsize=15, weight="bold", y=1.02)
    out_path = os.path.join(output_dir, "k_tuning_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. PER-CLASS PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(metrics_data: dict, output_dir: str = FIG_DIR) -> str:
    """Grouped bar chart of Precision, Recall, and F1 per condition class."""
    per_class = metrics_data["champion_configuration"]["metrics"]["per_class"]
    classes = list(per_class.keys())

    precisions = [per_class[c]["precision"] * 100 for c in classes]
    recalls    = [per_class[c]["recall"] * 100 for c in classes]
    f1_scores  = [per_class[c]["f1_score"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width, precisions, width, label="Precision", color="#3498db", edgecolor="#2980b9")
    rects2 = ax.bar(x,         recalls,    width, label="Recall",    color="#2ecc71", edgecolor="#27ae60")
    rects3 = ax.bar(x + width, f1_scores,  width, label="F1-Score",  color="#e67e22", edgecolor="#d35400")

    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, weight="bold"
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_title("Test Set Performance by Health Condition", weight="bold", pad=15)
    ax.set_ylabel("Score (%)", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, weight="bold", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    out_path = os.path.join(output_dir, "per_class_metrics.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE WEIGHTING COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_weighting_comparison(metrics_data: dict, output_dir: str = FIG_DIR) -> str:
    """Compare Equal Weights vs Engineering Weights on the Test Set."""
    eq_m = metrics_data["comparisons"]["equal_weights_test"]
    eng_m = metrics_data["comparisons"]["engineering_weights_test"]

    metrics_names = ["Overall Accuracy", "Macro F1-Score", "Critical Recall"]
    eq_vals = [
        eq_m["accuracy"] * 100,
        eq_m["macro_avg"]["f1_score"] * 100,
        eq_m["safety_audit"]["critical_recall"] * 100,
    ]
    eng_vals = [
        eng_m["accuracy"] * 100,
        eng_m["macro_avg"]["f1_score"] * 100,
        eng_m["safety_audit"]["critical_recall"] * 100,
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width/2, eq_vals,  width, label="Equal Feature Weights (1.0 each)", color="#95a5a6", edgecolor="#7f8c8d")
    rects2 = ax.bar(x + width/2, eng_vals, width, label="Domain Engineering Weights",     color="#1abc9c", edgecolor="#16a085")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    ax.set_title("Test Set Impact: Equal vs. Engineering Feature Weights", weight="bold", pad=15)
    ax.set_ylabel("Score (%)", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, weight="bold", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    out_path = os.path.join(output_dir, "feature_weighting_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL VS BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_vs_baselines(metrics_data: dict, output_dir: str = FIG_DIR) -> str:
    """Benchmark final model against Majority baseline and Numeric-only baseline."""
    maj_m  = metrics_data["comparisons"]["majority_class_baseline"]
    num_m  = metrics_data["comparisons"]["numeric_only_baseline"]
    champ_m = metrics_data["champion_configuration"]["metrics"]

    models = [
        "Majority Baseline\n(Zero-Rule)",
        "Numeric-Only\nEuclidean k-NN",
        "Champion Model\n(Gower-KNN)",
    ]

    accuracies = [
        maj_m["accuracy"] * 100,
        num_m["accuracy"] * 100,
        champ_m["accuracy"] * 100,
    ]

    macro_f1s = [
        maj_m["macro_avg"]["f1_score"] * 100,
        num_m["macro_avg"]["f1_score"] * 100,
        champ_m["macro_avg"]["f1_score"] * 100,
    ]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rects1 = ax.bar(x - width/2, accuracies, width, label="Accuracy", color="#34495e", edgecolor="#2c3e50")
    rects2 = ax.bar(x + width/2, macro_f1s,   width, label="Macro F1", color="#e74c3c", edgecolor="#c0392b")

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    ax.set_title("Benchmarking: Gower k-NN vs. Baseline Alternatives", weight="bold", pad=15)
    ax.set_ylabel("Score (%)", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, weight="bold", fontsize=10)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    out_path = os.path.join(output_dir, "model_vs_baselines.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN GENERATOR FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_visualizations(
    metrics_path: str = METRICS_JSON,
    output_dir: str = FIG_DIR
) -> list[str]:
    """Generate all evaluation figures and save to figures directory."""
    os.makedirs(output_dir, exist_ok=True)
    data = load_metrics_data(metrics_path)

    generated = [
        plot_confusion_matrix(data, output_dir),
        plot_k_tuning_curves(data, output_dir),
        plot_per_class_metrics(data, output_dir),
        plot_weighting_comparison(data, output_dir),
        plot_model_vs_baselines(data, output_dir),
    ]

    print(f"\nGenerated {len(generated)} visualization figures in '{output_dir}':")
    for p in generated:
        print(f"  - {os.path.basename(p)}")

    return generated


if __name__ == "__main__":
    generate_all_visualizations()
