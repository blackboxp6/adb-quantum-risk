
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input files
MC_FILE = OUTPUT_DIR / "convergence_results.csv"
STRATIFIED_FILE = OUTPUT_DIR / "stratified_results.csv"
SOBOL_FILE = OUTPUT_DIR / "sobol_results.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "importance_sampling_results.csv"


def load_classical_results():
    """Load the largest available run for each method."""

    results = []

    # 1. Ordinary Monte Carlo
    mc = pd.read_csv(MC_FILE)

    mc = mc.sort_values("Sample Size").groupby(
        "Metric", as_index=False
    ).tail(1)

    for _, row in mc.iterrows():
        results.append({
            "Method": "Monte Carlo",
            "Metric": row["Metric"],
            "Estimate": row["Monte Carlo Estimate"],
            "Exact Value": row["Exact Value"],
            "Relative Error (%)": row["Relative Error (%)"],
            "Standard Error": np.nan,
            "Sample Budget": int(row["Sample Size"])
        })

    # 2. Stratified Monte Carlo
    stratified = pd.read_csv(STRATIFIED_FILE)

    for _, row in stratified.iterrows():
        results.append({
            "Method": "Stratified MC",
            "Metric": row["Metric"],
            "Estimate": row["Stratified Estimate"],
            "Exact Value": row["Exact Value"],
            "Relative Error (%)": row["Relative Error (%)"],
            "Standard Error": pd.to_numeric(
                row["Standard Error"], errors="coerce"
            ),
            "Sample Budget": 300_000
        })

    # 3. Scrambled Sobol
    sobol = pd.read_csv(SOBOL_FILE)

    sobol = sobol.sort_values(
        "Total points across scrambles"
    ).groupby("Metric", as_index=False).tail(1)

    for _, row in sobol.iterrows():
        results.append({
            "Method": "Sobol",
            "Metric": row["Metric"],
            "Estimate": row["Sobol estimate"],
            "Exact Value": row["Exact value"],
            "Relative Error (%)": row["Relative error (%)"],
            "Standard Error": row["Standard error"],
            "Sample Budget": int(row["Total points across scrambles"])
        })

    # 4. Importance sampling
    importance = pd.read_csv(IMPORTANCE_FILE)

    for _, row in importance.iterrows():
        results.append({
            "Method": "Importance Sampling",
            "Metric": row["Metric"],
            "Estimate": row["Importance Sampling Estimate"],
            "Exact Value": row["Exact Value"],
            "Relative Error (%)": row["Relative Error (%)"],
            "Standard Error": pd.to_numeric(
                row["Standard Error"], errors="coerce"
            ),
            "Sample Budget": 300_000
        })

    return pd.DataFrame(results)


def create_comparison_table(results):
    """Print and save the complete comparison."""

    metric_order = [
        "Expected principal loss",
        "Both countries pay",
        "Portfolio loss exceeds 80m",
        "Expected stop-loss",
        "Both bonds exhausted"
    ]

    method_order = [
        "Monte Carlo",
        "Stratified MC",
        "Sobol",
        "Importance Sampling"
    ]

    results["Metric"] = pd.Categorical(
        results["Metric"],
        categories=metric_order,
        ordered=True
    )

    results["Method"] = pd.Categorical(
        results["Method"],
        categories=method_order,
        ordered=True
    )

    results = results.sort_values(["Metric", "Method"])

    output_file = OUTPUT_DIR / "classical_comparison.csv"
    results.to_csv(output_file, index=False)

    print("\nCLASSICAL METHOD COMPARISON")
    print("===========================")

    for metric in metric_order:
        subset = results[results["Metric"] == metric]

        print(f"\n{metric}")
        print("-" * len(metric))

        for _, row in subset.iterrows():
            print(
                f"{row['Method']:<22} "
                f"Estimate: {row['Estimate']:.8g} | "
                f"Relative error: {row['Relative Error (%)']:.4f}% | "
                f"Samples: {row['Sample Budget']:,}"
            )

    print(f"\nSaved comparison table: {output_file}")

    return results


def create_error_plot(results):
    """Compare relative errors across methods."""

    pivot = results.pivot(
        index="Metric",
        columns="Method",
        values="Relative Error (%)"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(13, 7),
        width=0.8
    )

    ax.set_yscale("log")
    ax.set_ylabel("Relative error (%) — logarithmic scale")
    ax.set_xlabel("ADB risk metric")
    ax.set_title("Classical Sampling Method Accuracy")
    ax.legend(title="Method")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    output_file = OUTPUT_DIR / "classical_method_comparison.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Saved comparison graph: {output_file}")


if __name__ == "__main__":

    results = load_classical_results()

    results = create_comparison_table(results)

    create_error_plot(results)

    print("\nSUCCESS: Classical comparison completed.")
