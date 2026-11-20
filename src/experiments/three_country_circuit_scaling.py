
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(exist_ok=True)

# --------------------------------------------------
# 1. SCALABILITY SETTINGS
# --------------------------------------------------

COUNTRY_COUNTS = [2, 3, 4, 5]

rng = np.random.default_rng(42)

results = []

print("=" * 65)
print("QUANTUM STATE-PREPARATION SCALABILITY EXPERIMENT")
print("=" * 65)

# --------------------------------------------------
# 2. TEST EACH COUNTRY COUNT
# --------------------------------------------------

for n_countries in COUNTRY_COUNTS:

    outcomes = 10 ** n_countries

    qubits = int(np.ceil(np.log2(outcomes)))

    dimension = 2 ** qubits

    unused = dimension - outcomes

    print(f"\nCountries: {n_countries}")
    print(f"Possible outcomes: {outcomes:,}")
    print(f"Required qubits: {qubits}")
    print(f"Quantum dimension: {dimension:,}")

    # Generate a reproducible synthetic probability distribution.
    # This is a scalability test, not an ADB-calibrated model.

    probabilities = rng.random(outcomes)

    probabilities = probabilities / probabilities.sum()

    padded = np.zeros(dimension)
    padded[:outcomes] = probabilities

    amplitudes = np.sqrt(padded)

    # Construct state-preparation circuit
    preparation = StatePreparation(amplitudes)

    circuit = QuantumCircuit(qubits)
    circuit.append(preparation, range(qubits))

    # Transpile into elementary basis gates
    print("Transpiling...")

    compiled = transpile(
    circuit,
    basis_gates=["u", "cx"],
    optimization_level=1,
    seed_transpiler=42
    )

    operations = compiled.count_ops()

    depth = compiled.depth()
    total_gates = compiled.size()
    cx_gates = operations.get("cx", 0)
    u_gates = operations.get("u", 0)

    print(f"Depth: {depth:,}")
    print(f"Total gates: {total_gates:,}")
    print(f"CX gates: {cx_gates:,}")
    print(f"U gates: {u_gates:,}")

    results.append({
        "countries": n_countries,
        "outcomes": outcomes,
        "qubits": qubits,
        "dimension": dimension,
        "unused_states": unused,
        "depth": depth,
        "total_gates": total_gates,
        "cx_gates": cx_gates,
        "u_gates": u_gates
    })

# --------------------------------------------------
# 3. SAVE RESULTS
# --------------------------------------------------

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT / "quantum_circuit_scaling.csv",
    index=False
)

# --------------------------------------------------
# 4. PLOT CIRCUIT DEPTH
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df["countries"],
    df["depth"],
    marker="o",
    linewidth=2
)

for _, row in df.iterrows():
    plt.annotate(
        f'{row["depth"]:,}',
        (row["countries"], row["depth"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

plt.yscale("log")

plt.xlabel("Number of countries")
plt.ylabel("Transpiled circuit depth (log scale)")

plt.title(
    "Synthetic State-Preparation Circuit Depth vs Countries"
)

plt.xticks(COUNTRY_COUNTS)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "quantum_depth_vs_countries.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 5. PLOT CX GATE COUNT
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    df["countries"],
    df["cx_gates"],
    marker="o",
    linewidth=2
)

for _, row in df.iterrows():
    plt.annotate(
        f'{row["cx_gates"]:,}',
        (row["countries"], row["cx_gates"]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center"
    )

plt.yscale("log")

plt.xlabel("Number of countries")
plt.ylabel("Number of CX gates (log scale)")

plt.title(
    "Synthetic State-Preparation CX Gates vs Countries"
)

plt.xticks(COUNTRY_COUNTS)
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT / "quantum_cx_vs_countries.png",
    dpi=300
)

plt.close()
