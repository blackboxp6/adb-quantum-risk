
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, DiagonalGate


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "outputs" / "quantum_input_distribution.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

N_QUBITS = 7
DIMENSION = 2 ** N_QUBITS

SHOTS = 4096
GROVER_POWERS = [0, 1, 2, 4]

BASIS_GATES = ["u", "cx"]


# --------------------------------------------------
# 2. LOAD QUANTUM DISTRIBUTION
# --------------------------------------------------

print("\nQUANTUM RESOURCE ANALYSIS")
print("=========================")

data = pd.read_csv(INPUT_FILE)

amplitudes = data["Amplitude"].to_numpy(dtype=float)
marked = data["Marked"].astype(bool).to_numpy()

assert len(amplitudes) == DIMENSION
assert len(marked) == DIMENSION
assert np.isclose(np.sum(amplitudes ** 2), 1.0)

print("Qubits:", N_QUBITS)
print("Marked states:", np.sum(marked))
print("Shots per circuit:", SHOTS)


# --------------------------------------------------
# 3. STATE PREPARATION
# --------------------------------------------------

A = QuantumCircuit(N_QUBITS)

A.append(
    StatePreparation(amplitudes),
    range(N_QUBITS)
)


# --------------------------------------------------
# 4. PHASE ORACLE
# --------------------------------------------------

diagonal = np.ones(DIMENSION)
diagonal[marked] = -1

oracle = QuantumCircuit(N_QUBITS)

oracle.append(
    DiagonalGate(diagonal),
    range(N_QUBITS)
)


# --------------------------------------------------
# 5. ZERO-STATE REFLECTION
# --------------------------------------------------

def create_zero_reflection():

    circuit = QuantumCircuit(N_QUBITS)

    circuit.x(range(N_QUBITS))

    circuit.h(N_QUBITS - 1)

    circuit.mcx(
        list(range(N_QUBITS - 1)),
        N_QUBITS - 1
    )

    circuit.h(N_QUBITS - 1)

    circuit.x(range(N_QUBITS))

    return circuit


S0 = create_zero_reflection()


# --------------------------------------------------
# 6. ONE GROVER ITERATION
# --------------------------------------------------

Q = QuantumCircuit(N_QUBITS)

Q.compose(oracle, inplace=True)
Q.compose(A.inverse(), inplace=True)
Q.compose(S0, inplace=True)
Q.compose(A, inplace=True)


# --------------------------------------------------
# 7. TRANSPILATION AND RESOURCE COUNTING
# --------------------------------------------------

def compile_and_measure(circuit):

    compiled = transpile(
        circuit,
        basis_gates=BASIS_GATES,
        optimization_level=1
    )

    counts = compiled.count_ops()

    u_count = counts.get("u", 0)
    cx_count = counts.get("cx", 0)

    total_gates = sum(counts.values())

    return {
        "Depth": compiled.depth(),
        "U gates": u_count,
        "CX gates": cx_count,
        "Total gates": total_gates
    }


print("\nCOMPILING INDIVIDUAL COMPONENTS")
print("-------------------------------")

components = {
    "State preparation A": A,
    "Inverse preparation A†": A.inverse(),
    "Phase oracle": oracle,
    "Zero reflection S0": S0
}

component_results = []

for name, circuit in components.items():

    print(f"Compiling {name}...")

    resources = compile_and_measure(circuit)

    resources["Component"] = name

    component_results.append(resources)

    print(
        f"  Depth={resources['Depth']}, "
        f"CX={resources['CX gates']}, "
        f"Total={resources['Total gates']}"
    )


# --------------------------------------------------
# 8. COMPILE COMPLETE MLAE CIRCUITS
# --------------------------------------------------

print("\nCOMPILING COMPLETE MLAE CIRCUITS")
print("--------------------------------")

results = []

for k in GROVER_POWERS:

    print(f"Compiling circuit for k={k}...")

    circuit = QuantumCircuit(N_QUBITS)

    # Initial state preparation.
    circuit.compose(A, inplace=True)

    # Apply k Grover iterations.
    for _ in range(k):
        circuit.compose(Q, inplace=True)

    resources = compile_and_measure(circuit)

    # One initial A plus two A operations per Grover iteration.
    state_preparations = 1 + 2 * k

    # One phase oracle per Grover iteration.
    oracle_calls = k

    # Multiply the per-circuit cost by the number of shots.
    row = {
        "Grover iterations": k,
        "Shots": SHOTS,
        "State preparations per circuit": state_preparations,
        "Oracle calls per circuit": oracle_calls,
        "Circuit depth": resources["Depth"],
        "U gates per circuit": resources["U gates"],
        "CX gates per circuit": resources["CX gates"],
        "Total gates per circuit": resources["Total gates"],
        "Total CX gates across shots": resources["CX gates"] * SHOTS,
        "Total gates across shots": resources["Total gates"] * SHOTS,
        "Total oracle calls across shots": oracle_calls * SHOTS,
        "Total state preparations across shots": state_preparations * SHOTS
    }

    results.append(row)

    print(
        f"  Depth={resources['Depth']}, "
        f"U={resources['U gates']}, "
        f"CX={resources['CX gates']}, "
        f"Total gates={resources['Total gates']}"
    )


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

component_df = pd.DataFrame(component_results)
results_df = pd.DataFrame(results)

component_file = OUTPUT_DIR / "quantum_component_resources.csv"
results_file = OUTPUT_DIR / "quantum_resource_results.csv"

component_df.to_csv(component_file, index=False)
results_df.to_csv(results_file, index=False)


# --------------------------------------------------
# 10. TOTAL RESOURCE SUMMARY
# --------------------------------------------------

print("\nTOTAL RESOURCE ACCOUNTING")
print("-------------------------")

print(
    "Total measurement shots:",
    results_df["Shots"].sum()
)

print(
    "Total Grover oracle calls:",
    results_df["Total oracle calls across shots"].sum()
)

print(
    "Total state preparations:",
    results_df["Total state preparations across shots"].sum()
)

print(
    "Total CX gates across all shots:",
    results_df["Total CX gates across shots"].sum()
)

print(
    "Total elementary gates across all shots:",
    results_df["Total gates across shots"].sum()
)


# --------------------------------------------------
# 11. FINAL OUTPUT
# --------------------------------------------------

print("\nSAVED FILES")
print("-----------")
print(component_file)
print(results_file)

print("\nSUCCESS: Quantum resource analysis completed.")
