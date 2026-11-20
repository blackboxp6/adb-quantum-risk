
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Project paths
ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "outputs" / "fair_classical_qmlae_comparison.csv"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load benchmark results
df = pd.read_csv(INPUT_FILE)

# Methods and metrics
methods = [
    "Monte Carlo",
    "Stratified Monte Carlo",
    "Importance Sampling",
    "Sobol",
    "QMLAE",
]

metrics = [
    "Expected principal loss",
    "Both countries pay",
    "Portfolio loss exceeds 80m",
    "Expected stop-loss",
    "Both bonds exhausted",
]

# Check that the expected columns and methods exist
required_columns = {"Method", "Metric", "Budget", "Relative MAE (%)"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing CSV columns: {missing_columns}")

available_methods = set(df["Method"].unique())
missing_methods = set(methods) - available_methods

if missing_methods:
    raise ValueError(f"Missing methods in CSV: {missing_methods}")

# Generate one plot per risk metric
for metric in metrics:
    metric_df = df[df["Metric"] == metric]

    if metric_df.empty:
        print(f"WARNING: No data found for {metric}")
        continue

    plt.figure(figsize=(9, 6))

    for method in methods:
        method_df = metric_df[
            metric_df["Method"] == method
        ].sort_values("Budget")

        if method_df.empty:
            continue

        plt.plot(
            method_df["Budget"],
            method_df["Relative MAE (%)"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=method,
        )

    plt.xscale("log", base=2)
    plt.xticks(
        [1024, 4096, 8192, 16384],
        ["1,024", "4,096", "8,192", "16,384"],
    )

    plt.xlabel("Evaluation Budget", fontsize=12)
    plt.ylabel("Relative MAE (%)", fontsize=12)
    plt.title(
        f"Estimation Error vs Computational Resources\n{metric}",
        fontsize=13,
    )

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace("$", "")
        .replace(" ", "_")
        .replace("-", "_")
        + "_error_vs_resources.png"
    )

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {filename}")

print("\nAll available metric plots generated.")
