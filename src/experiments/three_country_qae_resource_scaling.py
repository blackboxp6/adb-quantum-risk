
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from qiskit import QuantumCircuit, transpile, qpy

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "outputs"
OUTPUT = ROOT / "outputs"

POWERS = [0, 1, 2, 4, 8]

OUTPUT.mkdir(exist_ok=True)

# ============================================================
# 2. LOAD PREPARATION AND GROVER CIRCUITS
# ============================================================

with open(INPUT / "three_country_state_preparation.qpy", "rb") as f:
    A = qpy.load(f)[0]

with open(INPUT / "three_country_grover.qpy", "rb") as f:
    G = qpy.load(f)[0]

n = A.num_qubits

print("=" * 65)
print("THREE-COUNTRY QAE RESOURCE SCALING")
print("=" * 65)

print(f"Qubits: {n}")
print(f"Grover powers: {POWERS}")

# ============================================================
# 3. TRANSPILATION SETTINGS
# ============================================================

BASIS_GATES = ["u", "cx"]

def compile_circuit(circuit):

    return transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=1,
        seed_transpiler=42
    )

# Compile A once for reporting its cost.
compiled_A = compile_circuit(A)

A_depth = compiled_A.depth()
A_gates = compiled_A.size()
A_cx = compiled_A.count_ops().get("cx", 0)
A_u = compiled_A.count_ops().get("u", 0)

print("\nSTATE PREPARATION COST")
print(f"Depth: {A_depth:,}")
print(f"Total gates: {A_gates:,}")
print(f"CX gates: {A_cx:,}")
print(f"U gates: {A_u:,}")

# ============================================================
# 4. BUILD AND MEASURE EACH QAE CIRCUIT
# ============================================================

results = []

for power in POWERS:

    print("\n" + "-" * 60)
    print(f"QAE POWER: {power}")
    print("-" * 60)

    circuit = QuantumCircuit(n)

    # Prepare initial probability distribution.
    circuit.compose(A, inplace=True)

    # Apply Grover operator repeatedly.
    for _ in range(power):
        circuit.compose(G, inplace=True)

    print("Transpiling complete QAE circuit...")

    compiled = compile_circuit(circuit)

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

        "countries": 3,
        "qubits": n,
        "qae_power": power,
        "grover_applications": power,

        "state_preparation_depth": A_depth,
        "state_preparation_gates": A_gates,
        "state_preparation_cx": A_cx,

        "total_depth": depth,
        "total_gates": total_gates,
        "cx_gates": cx_gates,
        "u_gates": u_gates

    })

# ============================================================
# 5. SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)

csv_path = OUTPUT / "three_country_qae_resource_scaling.csv"

df.to_csv(csv_path, index=False)

print("\nRESOURCE SCALING TABLE")
print(df.to_string(index=False))

# ============================================================
# 6. PLOT CIRCUIT DEPTH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df["qae_power"],
    df["total_depth"],
    marker="o",
    linewidth=2
)

plt.yscale("log")

plt.xlabel("Number of Grover applications")
plt.ylabel("Complete circuit depth (log scale)")

plt.title(
    "Three-Country QAE Circuit Depth vs Grover Applications"
)

plt.xticks(POWERS)
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT / "three_country_qae_depth_vs_power.png",
    dpi=300
)

plt.close()

# ============================================================
# 7. PLOT CX GATE COUNT
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    df["qae_power"],
    df["cx_gates"],
    marker="o",
    linewidth=2
)

plt.yscale("log")

plt.xlabel("Number of Grover applications")
plt.ylabel("CX gate count (log scale)")

plt.title(
    "Three-Country QAE CX Gates vs Grover Applications"
)

plt.xticks(POWERS)
plt.grid(True, which="both", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT / "three_country_qae_cx_vs_power.png",
    dpi=300
)

plt.close()

# ============================================================
# 8. FINAL OUTPUT
# ============================================================

print("\nSAVED FILES")
print(csv_path)
print(OUTPUT / "three_country_qae_depth_vs_power.png")
print(OUTPUT / "three_country_qae_cx_vs_power.png")

print("\nQAE RESOURCE SCALING COMPLETE.")
