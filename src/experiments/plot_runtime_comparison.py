
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

# Classical benchmark data
budgets = np.array([1024, 4096, 8192, 16384])

classical = {
    "Monte Carlo": [0.000579892, 0.001768561, 0.003219126, 0.006747197],
    "Stratified MC": [0.001046377, 0.002007007, 0.003429090, 0.006502650],
    "Importance Sampling": [0.000692955, 0.002123534, 0.004136053, 0.008279491],
    "Sobol": [0.001574167, 0.0025020109, 0.003962040, 0.007513922],
}

# QMLAE complete benchmark runtime
qmlae_total_runtime = 1.086546199861914

# Create two-panel figure
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(13, 5)
)

# Panel 1: Classical runtime by budget
for method, times in classical.items():
    ax1.plot(
        budgets,
        np.array(times) * 1000,
        marker="o",
        linewidth=2,
        label=method
    )

ax1.set_title("Classical Runtime vs Sample Budget")
ax1.set_xlabel("Sample budget")
ax1.set_ylabel("Mean runtime per repetition (milliseconds)")
ax1.set_xticks(budgets)
ax1.set_xticklabels(["1,024", "4,096", "8,192", "16,384"])
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.legend(fontsize=8)

# Panel 2: QMLAE total runtime
ax2.bar(
    ["Ideal statistical\nQMLAE"],
    [qmlae_total_runtime]
)

ax2.set_title("QMLAE Complete Benchmark Runtime")
ax2.set_ylabel("Total Python runtime (seconds)")
ax2.set_ylim(0, 1.25)
ax2.grid(axis="y", linestyle="--", alpha=0.4)

ax2.text(
    0,
    qmlae_total_runtime + 0.025,
    f"{qmlae_total_runtime:.3f} s",
    ha="center"
)

fig.suptitle(
    "Computational Runtime of Classical Methods and QMLAE",
    fontsize=14
)

fig.text(
    0.01, 0.01,
    "Classical panel: mean runtime per repetition at each sample budget. "
    "QMLAE panel: total runtime for 100 repetitions, four budgets, and five metrics. "
    "QMLAE is an ideal statistical simulation, not quantum-hardware runtime. "
    "The panels are intentionally separate because their timing scopes differ.",
    fontsize=8
)

plt.tight_layout(rect=[0, 0.09, 1, 0.93])

# Save figure
plot_path = OUTPUT / "runtime_comparison_two_panel.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved to: {plot_path}")
