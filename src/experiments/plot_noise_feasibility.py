
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"

INPUT_FILE = (
    OUTPUT_DIR / "two_country_noise_repeated_summary.csv"
)

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("Noise rate")

# Convert noise rates to percentages for readable labels
noise_percent = df["Noise rate"] * 100

# --------------------------------------------------
# Plot 1: Mean relative estimation error vs noise
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.errorbar(
    noise_percent,
    df["Mean relative error (%)"],
    yerr=df["SD relative error (%)"],
    marker="o",
    markersize=7,
    linewidth=2,
    capsize=5,
)

plt.xlabel("Per-gate / Readout Noise Probability (%)")
plt.ylabel("Mean Relative Estimation Error (%)")
plt.title("QMLAE Estimation Error vs Quantum Noise")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "qmlae_error_vs_quantum_noise.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Plot 2: Invalid-state rate vs noise
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.errorbar(
    noise_percent,
    df["Mean invalid-state rate"] * 100,
    yerr=df["SD invalid-state rate"] * 100,
    marker="s",
    markersize=7,
    linewidth=2,
    capsize=5,
)

plt.xlabel("Per-gate / Readout Noise Probability (%)")
plt.ylabel("Mean Invalid-State Rate (%)")
plt.title("Invalid-State Rate vs Quantum Noise")

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "qmlae_invalid_state_vs_noise.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print("Saved: qmlae_error_vs_quantum_noise.png")
print("Saved: qmlae_invalid_state_vs_noise.png")
print("\nNoise feasibility plots generated successfully.")
