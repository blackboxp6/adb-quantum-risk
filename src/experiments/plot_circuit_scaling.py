
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = ROOT / "outputs" / "quantum_circuit_scaling.csv"
OUTPUT_DIR = ROOT / "outputs"

df = pd.read_csv(INPUT_FILE)
df = df.sort_values("countries")

# Plot 1: Circuit depth vs number of countries
plt.figure(figsize=(9, 6))

plt.plot(
    df["countries"],
    df["depth"],
    marker="o",
    linewidth=2,
    markersize=8,
)

plt.xlabel("Number of Countries")
plt.ylabel("Transpiled Circuit Depth")
plt.title("Quantum Circuit Depth vs Number of Countries")
plt.xticks(df["countries"])
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "circuit_depth_vs_countries.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()


# Plot 2: Two-qubit gate count vs number of countries
plt.figure(figsize=(9, 6))

plt.plot(
    df["countries"],
    df["cx_gates"],
    marker="o",
    linewidth=2,
    markersize=8,
)

plt.xlabel("Number of Countries")
plt.ylabel("Number of CX Gates")
plt.title("Two-Qubit Gate Count vs Number of Countries")
plt.xticks(df["countries"])
plt.yscale("log")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cx_gates_vs_countries.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Saved: circuit_depth_vs_countries.png")
print("Saved: cx_gates_vs_countries.png")
print("\nCircuit scaling plots generated successfully.")
