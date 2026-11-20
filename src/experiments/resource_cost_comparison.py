
from pathlib import Path
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

classical_file = OUTPUT / "fair_classical_summary.csv"
qmlae_file = OUTPUT / "fair_qmlae_summary.csv"
runtime_file = OUTPUT / "fair_qmlae_runtime.csv"


# --------------------------------------------------
# 2. LOAD EXISTING EXPERIMENT RESULTS
# --------------------------------------------------

classical = pd.read_csv(classical_file)
qmlae = pd.read_csv(qmlae_file)
qmlae_runtime = pd.read_csv(runtime_file)

print("\nClassical methods:", classical["Method"].unique())
print("QMLAE metrics:", qmlae["Metric"].unique())


# --------------------------------------------------
# 3. PREPARE CLASSICAL RESULTS
# --------------------------------------------------

# Obtain exact reference values from the QMLAE results.
references = (
    qmlae[["Metric", "Exact value"]]
    .drop_duplicates(subset=["Metric"])
)

classical = classical.merge(
    references,
    on="Metric",
    how="left"
)

classical["Relative MAE (%)"] = (
    classical["MAE"] / classical["Exact value"].abs()
) * 100

classical["Relative RMSE (%)"] = (
    classical["RMSE"] / classical["Exact value"].abs()
) * 100

classical["Resource type"] = "Classical simulation"
classical["Shots"] = np.nan
classical["Qubits"] = np.nan
classical["Grover applications"] = np.nan
classical["Grover powers"] = "Not applicable"


# --------------------------------------------------
# 4. PREPARE QMLAE RESULTS
# --------------------------------------------------

quantum = qmlae.copy()

quantum["Method"] = "QMLAE"
quantum["Resource type"] = "Quantum circuit estimate"

# Average Grover applications is the total application
# count used by one QMLAE estimate in this benchmark.
quantum["Grover applications"] = (
    quantum["Average Grover applications"]
)

quantum["Shots"] = quantum["Average shots"]

quantum["Mean runtime (seconds)"] = np.nan

quantum["Bias"] = np.nan
quantum["Standard deviation"] = np.nan
quantum["Repetitions"] = quantum["Repetitions"]

quantum["Resource type"] = (
    "Ideal statistical QMLAE simulation"
)


# --------------------------------------------------
# 5. CONSOLIDATE RESULTS
# --------------------------------------------------

classical_columns = [
    "Method",
    "Budget",
    "Metric",
    "MAE",
    "RMSE",
    "Relative MAE (%)",
    "Relative RMSE (%)",
    "Mean runtime (seconds)",
    "Repetitions",
    "Resource type",
    "Shots",
    "Qubits",
    "Grover applications",
    "Grover powers",
]

quantum_columns = [
    "Method",
    "Budget",
    "Metric",
    "MAE",
    "RMSE",
    "Relative MAE (%)",
    "Relative RMSE (%)",
    "Mean runtime (seconds)",
    "Repetitions",
    "Resource type",
    "Shots",
    "Qubits",
    "Grover applications",
    "Grover powers",
]

combined = pd.concat(
    [
        classical[classical_columns],
        quantum[quantum_columns],
    ],
    ignore_index=True
)

combined = combined.sort_values(
    ["Metric", "Budget", "Method"]
)

combined.to_csv(
    OUTPUT / "resource_cost_comparison.csv",
    index=False
)


# --------------------------------------------------
# 6. CREATE QUANTUM RESOURCE SUMMARY
# --------------------------------------------------

quantum_resources = quantum[
    [
        "Metric",
        "Budget",
        "Qubits",
        "Shots",
        "Grover powers",
        "Grover applications",
        "MAE",
        "RMSE",
        "Relative MAE (%)",
        "Relative RMSE (%)",
    ]
].copy()

quantum_resources.to_csv(
    OUTPUT / "quantum_resource_summary.csv",
    index=False
)


# --------------------------------------------------
# 7. PLOT RELATIVE ESTIMATION ERROR
# --------------------------------------------------

metrics = combined["Metric"].dropna().unique()
budgets = sorted(combined["Budget"].unique())

fig, axes = plt.subplots(
    len(metrics),
    1,
    figsize=(11, 5 * len(metrics))
)

if len(metrics) == 1:
    axes = [axes]

for ax, metric in zip(axes, metrics):

    subset = combined[
        combined["Metric"] == metric
    ]

    for method in subset["Method"].unique():

        method_data = subset[
            subset["Method"] == method
        ].sort_values("Budget")

        ax.plot(
            method_data["Budget"],
            method_data["Relative MAE (%)"],
            marker="o",
            label=method
        )

    ax.set_title(metric)
    ax.set_xlabel("Computational budget")
    ax.set_ylabel("Relative MAE (%)")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle(
    "Classical Methods vs QMLAE: Relative Estimation Error",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "resource_cost_by_budget.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# 8. PRINT RESOURCE INFORMATION
# --------------------------------------------------

print("\n========== CLASSICAL RUNTIME ==========")

runtime_summary = (
    classical.groupby("Method")["Mean runtime (seconds)"]
    .mean()
    .sort_values()
)

print(runtime_summary.to_string())


print("\n========== QMLAE RUNTIME ==========")

print(qmlae_runtime.to_string(index=False))


print("\n========== QUANTUM RESOURCES ==========")

print(
    quantum_resources[
        [
            "Metric",
            "Budget",
            "Qubits",
            "Shots",
            "Grover applications",
            "Relative MAE (%)",
        ]
    ].to_string(index=False)
)


print("\n========== OUTPUT FILES ==========")

print("1. outputs/resource_cost_comparison.csv")
print("2. outputs/quantum_resource_summary.csv")
print("3. outputs/resource_cost_by_budget.png")

print("\nResource comparison completed.")
