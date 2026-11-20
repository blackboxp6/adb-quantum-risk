
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"

cost_file = OUTPUT / "resource_cost_comparison.csv"
quantum_file = OUTPUT / "quantum_resource_results.csv"
component_file = OUTPUT / "quantum_component_resources.csv"

cost = pd.read_csv(cost_file)
quantum = pd.read_csv(quantum_file)
components = pd.read_csv(component_file)

# --------------------------------------------------
# GRAPH 4A: Classical runtime vs budget
# --------------------------------------------------

classical = cost[
    cost["Method"].isin([
        "Monte Carlo",
        "Stratified Monte Carlo",
        "Importance Sampling",
        "Sobol",
    ])
].copy()

runtime = (
    classical.groupby(["Method", "Budget"], as_index=False)
    ["Mean runtime (seconds)"].first()
)

plt.figure(figsize=(9, 6))

for method in runtime["Method"].unique():
    data = runtime[runtime["Method"] == method].sort_values("Budget")

    plt.plot(
        data["Budget"],
        data["Mean runtime (seconds)"],
        marker="o",
        linewidth=2,
        label=method,
    )

plt.xlabel("Classical Sample Budget")
plt.ylabel("Mean Runtime (seconds)")
plt.title("Classical Simulation Runtime vs Sample Budget")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "classical_runtime_vs_budget.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# GRAPH 4B: Quantum gate executions
# --------------------------------------------------

quantum = quantum.sort_values("Grover iterations")

plt.figure(figsize=(9, 6))

plt.plot(
    quantum["Grover iterations"],
    quantum["Total CX gates across shots"],
    marker="o",
    linewidth=2,
    label="CX gate executions",
)

plt.plot(
    quantum["Grover iterations"],
    quantum["Total gates across shots"],
    marker="s",
    linewidth=2,
    label="Total gate executions",
)

plt.xlabel("Grover Iterations")
plt.ylabel("Gate Executions Across Shots")
plt.title("Quantum Gate Execution Cost")
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT / "quantum_gate_execution_cost.png",
    dpi=300,
)
plt.close()


# --------------------------------------------------
# GRAPH 4C: Quantum circuit component resources
# --------------------------------------------------

plt.figure(figsize=(10, 6))

labels = components["Component"]
x = range(len(labels))

plt.bar(
    x,
    components["CX gates"],
    label="CX gates",
)

plt.bar(
    x,
    components["U gates"],
    bottom=components["CX gates"],
    label="U gates",
)

plt.xticks(x, labels, rotation=15, ha="right")
plt.ylabel("Transpiled Gate Count")
plt.title("Quantum Circuit Resource Breakdown")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "quantum_component_resource_breakdown.png",
    dpi=300,
)
plt.close()

print("Saved: classical_runtime_vs_budget.png")
print("Saved: quantum_gate_execution_cost.png")
print("Saved: quantum_component_resource_breakdown.png")
print("\nComputational cost plots generated successfully.")
