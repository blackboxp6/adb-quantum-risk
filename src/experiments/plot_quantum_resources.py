
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "outputs" / "quantum_resource_results.csv"
OUTPUT = ROOT / "outputs" / "quantum_resource_scaling.png"

# Load compiled circuit resource results
df = pd.read_csv(INPUT)

# Remove accidental whitespace from column names
df.columns = df.columns.str.strip()

# Sort by Grover iteration count
df = df.sort_values("Grover iterations")

# Create two-panel figure
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --------------------------------------------------
# Panel 1: Per-circuit resources
# --------------------------------------------------

axes[0].plot(
    df["Grover iterations"],
    df["Circuit depth"],
    marker="o",
    label="Circuit depth"
)

axes[0].plot(
    df["Grover iterations"],
    df["Total gates per circuit"],
    marker="s",
    label="Total gates per circuit"
)

axes[0].set_title("Per-circuit resource requirements")
axes[0].set_xlabel("Grover iterations")
axes[0].set_ylabel("Number of operations")
axes[0].set_xticks(df["Grover iterations"])
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# --------------------------------------------------
# Panel 2: Cumulative resources across all shots
# --------------------------------------------------

axes[1].plot(
    df["Grover iterations"],
    df["Total CX gates across shots"],
    marker="o",
    label="CX gate executions"
)

axes[1].plot(
    df["Grover iterations"],
    df["Total gates across shots"],
    marker="s",
    label="Total gate executions"
)

axes[1].set_title("Cumulative gate executions")
axes[1].set_xlabel("Grover iterations")
axes[1].set_ylabel("Gate executions across shots")
axes[1].set_xticks(df["Grover iterations"])
axes[1].set_yscale("log")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

fig.suptitle(
    "Two-Country QAE: Quantum Circuit Resource Requirements",
    fontsize=15
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nFigure saved to: {OUTPUT}")
print("\nResource summary:")
print(
    df[
        [
            "Grover iterations",
            "Shots",
            "Circuit depth",
            "CX gates per circuit",
            "Total CX gates across shots",
            "Total gates across shots",
        ]
    ].to_string(index=False)
)
