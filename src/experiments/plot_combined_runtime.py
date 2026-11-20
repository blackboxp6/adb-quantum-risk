
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. CLASSICAL METHODS
# Mean runtime per repetition at each budget (seconds)
# --------------------------------------------------

budgets = [1024, 4096, 8192, 16384]

classical_runtime = {
    "Monte Carlo": [
        0.000579892,
        0.001768561,
        0.003219126,
        0.006747197
    ],
    "Stratified MC": [
        0.001046377,
        0.002007007,
        0.003429090,
        0.006502650
    ],
    "Importance Sampling": [
        0.000692955,
        0.002123534,
        0.004136053,
        0.008279491
    ],
    "Sobol": [
        0.001574167,
        0.0025020109,
        0.003962040,
        0.007513922
    ]
}

# Number of repetitions in the classical benchmark
REPETITIONS = 100

# Sum the four budget runtimes and scale by repetitions
total_runtime = {
    method: sum(times) * REPETITIONS
    for method, times in classical_runtime.items()
}

# --------------------------------------------------
# 2. QUANTUM METHOD
# --------------------------------------------------

# Total Python runtime reported by fair_qmlae_runtime.csv
# This is an ideal statistical simulation, NOT QPU runtime.

qmlae_runtime = 1.086546199861914

total_runtime["Ideal statistical QMLAE"] = qmlae_runtime

# --------------------------------------------------
# 3. CREATE COMPARISON TABLE
# --------------------------------------------------

df = pd.DataFrame({
    "Method": list(total_runtime.keys()),
    "Total runtime (seconds)": list(total_runtime.values())
})

df = df.sort_values("Total runtime (seconds)", ascending=True)

print("\nTOTAL COMPUTATIONAL RUNTIME")
print(df.to_string(index=False))

# Save comparison data
csv_path = OUTPUT_DIR / "combined_classical_quantum_runtime.csv"
df.to_csv(csv_path, index=False)

# --------------------------------------------------
# 4. CREATE PLOT
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    df["Method"],
    df["Total runtime (seconds)"]
)

ax.set_xlabel("Total benchmark runtime (seconds)")
ax.set_ylabel("Computational method")

ax.set_title(
    "Classical Methods vs Ideal Statistical QMLAE: "
    "Total Runtime"
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.35
)

ax.set_axisbelow(True)

# Add runtime labels
for bar, value in zip(
    bars,
    df["Total runtime (seconds)"]
):
    ax.text(
        value + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f} s",
        va="center",
        fontsize=9
    )

ax.set_xlim(
    0,
    max(df["Total runtime (seconds)"]) * 1.18
)

# Scientific interpretation and caveat
fig.text(
    0.01,
    0.01,
    "Classical totals: sum of mean runtimes across four budgets "
    "multiplied by 100 repetitions. "
    "QMLAE: reported Python runtime for 100 repetitions, "
    "four budgets, and five metrics. "
    "QMLAE is an ideal statistical simulation, not quantum-hardware runtime.",
    ha="left",
    fontsize=8,
    wrap=True
)

plt.tight_layout(rect=[0, 0.09, 1, 1])

# Save plot
plot_path = OUTPUT_DIR / "classical_vs_qmlae_total_runtime.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFiles saved:")
print(plot_path)
print(csv_path)
