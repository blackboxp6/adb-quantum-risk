
from pathlib import Path
import csv
import sys

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "three_country_common_shock_term_distribution.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load the classical probability distribution
# ============================================================

states = []
probabilities = []
losses = []

with open(INPUT_FILE, "r", newline="") as file:

    reader = csv.DictReader(file)

    for row in reader:

        a = int(row["earthquake_A"])
        b = int(row["earthquake_B"])
        c = int(row["earthquake_C"])

        r1 = int(row["rain_A"])
        r2 = int(row["rain_B"])
        r3 = int(row["rain_C"])

        probability = float(row["probability"])

        # Calculate individual country losses.
        loss_a = 70_000_000 * a / 4 + 10_000_000 * r1
        loss_b = 70_000_000 * b / 4 + 10_000_000 * r2
        loss_c = 70_000_000 * c / 4 + 10_000_000 * r3

        portfolio_loss = loss_a + loss_b + loss_c

        states.append((a, b, c, r1, r2, r3))
        probabilities.append(probability)
        losses.append(portfolio_loss)


probabilities = np.asarray(probabilities, dtype=float)
losses = np.asarray(losses, dtype=float)

number_of_states = len(probabilities)

print("THREE-COUNTRY QUANTUM STATE PREPARATION")
print("=" * 60)

print("Number of classical outcomes:", number_of_states)
print("Probability sum:", probabilities.sum())


# ============================================================
# 3. Determine the number of qubits
# ============================================================

n_qubits = int(np.ceil(np.log2(number_of_states)))
dimension = 2 ** n_qubits

print("\nQuantum encoding")
print("Required qubits:", n_qubits)
print("Quantum state dimension:", dimension)
print("Unused basis states:", dimension - number_of_states)


# ============================================================
# 4. Pad the probability distribution
# ============================================================

# Quantum amplitudes are square roots of probabilities.
# The remaining unused states receive zero amplitude.

amplitudes = np.zeros(dimension, dtype=float)

amplitudes[:number_of_states] = np.sqrt(probabilities)

# Normalize to protect against small numerical errors.
amplitudes /= np.linalg.norm(amplitudes)

print("\nAmplitude vector")
print("Amplitude vector length:", len(amplitudes))
print("Amplitude normalization:", np.sum(amplitudes ** 2))


# ============================================================
# 5. Define the loss threshold and marked outcomes
# ============================================================

threshold = 120_000_000

marked = losses > threshold

exact_probability = probabilities[marked].sum()

print("\nOracle target")
print("Loss threshold: $120,000,000")
print("Marked outcomes:", np.count_nonzero(marked))
print("Exact exceedance probability:", exact_probability)
print("Exact exceedance percentage:", exact_probability * 100)


# ============================================================
# 6. Construct the quantum state-preparation circuit
# ============================================================

circuit = QuantumCircuit(n_qubits)

preparation = StatePreparation(amplitudes)

circuit.append(preparation, range(n_qubits))

print("\nHigh-level circuit")
print("Qubits:", circuit.num_qubits)
print("Operations:", circuit.count_ops())


# ============================================================
# 7. Verify the prepared state using a statevector
# ============================================================

statevector = Statevector.from_instruction(circuit)

prepared_amplitudes = statevector.data

# Fidelity checks whether the prepared state matches the
# intended probability-amplitude vector.
fidelity = abs(
    np.vdot(amplitudes, prepared_amplitudes)
) ** 2

# Compare measurement probabilities.
prepared_probabilities = np.abs(prepared_amplitudes) ** 2

max_probability_error = np.max(
    np.abs(prepared_probabilities - amplitudes ** 2)
)

print("\nStatevector verification")
print(f"State fidelity: {fidelity:.15f}")
print(f"Maximum probability error: {max_probability_error:.3e}")


# ============================================================
# 8. Transpile into elementary gates
# ============================================================

print("\nTranspiling circuit...")

transpiled = transpile(
    circuit,
    basis_gates=["u", "cx"],
    optimization_level=1
)

gate_counts = transpiled.count_ops()

u_count = gate_counts.get("u", 0)
cx_count = gate_counts.get("cx", 0)

print("\nTranspiled circuit resources")
print("Circuit depth:", transpiled.depth())
print("Total gates:", transpiled.size())
print("U gates:", u_count)
print("CX gates:", cx_count)


# ============================================================
# 9. Save circuit and quantum input data
# ============================================================

circuit_path = OUTPUT_DIR / "three_country_state_preparation.qpy"

from qiskit import qpy

with open(circuit_path, "wb") as file:
    qpy.dump(transpiled, file)


np.savez(
    OUTPUT_DIR / "three_country_quantum_input.npz",
    amplitudes=amplitudes,
    probabilities=probabilities,
    losses=losses,
    marked=marked,
    threshold=threshold
)


# Save a human-readable summary.
summary_path = OUTPUT_DIR / "three_country_quantum_resources.txt"

with open(summary_path, "w") as file:

    file.write("THREE-COUNTRY QUANTUM STATE PREPARATION\n")
    file.write("=" * 50 + "\n")
    file.write(f"Classical outcomes: {number_of_states}\n")
    file.write(f"Qubits: {n_qubits}\n")
    file.write(f"Quantum dimension: {dimension}\n")
    file.write(f"Unused states: {dimension - number_of_states}\n")
    file.write(f"Marked outcomes: {np.count_nonzero(marked)}\n")
    file.write(f"Exact probability: {exact_probability:.12f}\n")
    file.write(f"Fidelity: {fidelity:.15f}\n")
    file.write(f"Maximum probability error: {max_probability_error:.3e}\n")
    file.write(f"Circuit depth: {transpiled.depth()}\n")
    file.write(f"Total gates: {transpiled.size()}\n")
    file.write(f"U gates: {u_count}\n")
    file.write(f"CX gates: {cx_count}\n")


print("\nSaved files:")
print(circuit_path)
print(OUTPUT_DIR / "three_country_quantum_input.npz")
print(summary_path)

print("\nQUANTUM STATE PREPARATION COMPLETE.")
