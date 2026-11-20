
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

resource_file = OUTPUT_DIR / "quantum_resource_results.csv"
noise_file = OUTPUT_DIR / "two_country_noise_repeated_summary.csv"

output_file = OUTPUT_DIR / "quantum_resources_and_noise_final.png"

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

resources = pd.read_csv(resource_file)
noise = pd.read_csv(noise_file)

resources.columns = resources.columns.str.strip()
noise.columns = noise.columns.str.strip()

resources = resources.sort_values("Grover iterations")
noise = noise.sort_values("Noise rate")

# Convert decimal rates to percentages
noise["Noise rate (%)"] = noise["Noise rate"] * 100
noise["Mean invalid-state rate (%)"] = (
    noise["Mean invalid-state rate"] * 100
)
noise["SD invalid-state rate (pp)"] = (
    noise["SD invalid-state rate"] * 100
)

# --------------------------------------------------
# 3. Create four-panel figure
# --------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 11)
)

# --------------------------------------------------
# PANEL A: Per-circuit resources
# --------------------------------------------------

ax = axes[0, 0]

ax.plot(
    resources["Grover iterations"],
    resources["Circuit depth"],
    marker="o",
    linewidth=2,
    label="Circuit depth"
)

ax.plot(
    resources["Grover iterations"],
    resources["Total gates per circuit"],
    marker="s",
    linewidth=2,
    label="Total gates per circuit"
)

ax.set_title("(A) Per-circuit resource requirements")
ax.set_xlabel("Grover iterations")
ax.set_ylabel("Number of operations")
ax.set_xticks(resources["Grover iterations"])
ax.grid(True, alpha=0.3)
ax.legend()

# --------------------------------------------------
# PANEL B: Cumulative gate executions
# --------------------------------------------------

ax = axes[0, 1]

ax.plot(
    resources["Grover iterations"],
    resources["Total CX gates across shots"],
    marker="o",
    linewidth=2,
    label="CX gate executions"
)

ax.plot(
    resources["Grover iterations"],
    resources["Total gates across shots"],
    marker="s",
    linewidth=2,
    label="Total gate executions"
)

ax.set_title("(B) Cumulative gate executions")
ax.set_xlabel("Grover iterations")
ax.set_ylabel("Gate executions across shots")
ax.set_xticks(resources["Grover iterations"])
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

# --------------------------------------------------
# PANEL C: Estimation error under noise
# --------------------------------------------------

ax = axes[1, 0]

ax.errorbar(
    noise["Noise rate (%)"],
    noise["Mean absolute error (pp)"],
    yerr=noise["SD absolute error (pp)"],
    marker="o",
    capsize=4,
    linewidth=2,
    label="Mean absolute error"
)

ax.set_title("(C) Estimation error under noise")
ax.set_xlabel("Depolarizing / readout error rate (%)")
ax.set_ylabel("Mean absolute error (percentage points)")
ax.grid(True, alpha=0.3)
ax.legend()

# --------------------------------------------------
# PANEL D: Invalid-state rate under noise
# --------------------------------------------------

ax = axes[1, 1]

ax.errorbar(
    noise["Noise rate (%)"],
    noise["Mean invalid-state rate (%)"],
    yerr=noise["SD invalid-state rate (pp)"],
    marker="s",
    capsize=4,
    linewidth=2,
    label="Mean invalid-state rate"
)

ax.set_title("(D) Invalid-state rate under noise")
ax.set_xlabel("Depolarizing / readout error rate (%)")
ax.set_ylabel("Invalid-state rate (%)")
ax.grid(True, alpha=0.3)
ax.legend()

# --------------------------------------------------
# 4. Final publication formatting
# --------------------------------------------------

fig.suptitle(
    "Two-Country QAE: Resource Requirements and NISQ Feasibility",
    fontsize=17,
    fontweight="bold",
    y=0.98
)

fig.subplots_adjust(
    top=0.84,
    bottom=0.12,
    left=0.15,
    right=0.97,
    hspace=0.42,
    wspace=0.32
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.3
)

plt.show()

print("\nSaved figure:")
print(output_file)

print("\nNoise experiment summary:")
print(
    noise[
        [
            "Noise rate (%)",
            "Mean absolute error (pp)",
            "Mean invalid-state rate (%)"
        ]
    ].to_string(index=False)
)
