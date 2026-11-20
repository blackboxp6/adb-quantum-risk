
from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.quantum_info import Statevector


# ============================================================
# 1. Load the three-country quantum input
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

data = np.load(
    OUTPUT_DIR / "three_country_quantum_input.npz"
)

amplitudes = data["amplitudes"]
probabilities = data["probabilities"]
losses = data["losses"]
marked = data["marked"]

n_qubits = 10
dimension = 2 ** n_qubits

print("THREE-COUNTRY LOSS ORACLE")
print("=" * 60)

print("Qubits:", n_qubits)
print("Basis-state dimension:", dimension)
print("Marked outcomes:", np.count_nonzero(marked))


# ============================================================
# 2. Construct the phase oracle
# ============================================================

# Unmarked states receive phase +1.
# Marked states receive phase -1.

# Extend the classical marking array to the full
# quantum register dimension (1024).

marked_padded = np.zeros(dimension, dtype=bool)
marked_padded[:len(marked)] = marked

# Unmarked states receive +1.
# Marked states receive -1.

diagonal = np.ones(dimension, dtype=complex)
diagonal[marked_padded] = -1.0

oracle = QuantumCircuit(n_qubits, name="LossOracle")

oracle.append(
    DiagonalGate(diagonal),
    range(n_qubits)
)

print("\nOracle constructed.")
print("Oracle operations:", oracle.count_ops())


# ============================================================
# 3. Verify the oracle using the prepared state
# ============================================================

# Prepare the original probability state.
from qiskit.circuit.library import StatePreparation

preparation = QuantumCircuit(n_qubits)
preparation.append(
    StatePreparation(amplitudes),
    range(n_qubits)
)

# Apply state preparation followed by the oracle.
test_circuit = preparation.compose(oracle)

before = Statevector.from_instruction(preparation).data
after = Statevector.from_instruction(test_circuit).data

# Marked amplitudes should change sign.
expected_after = before.copy()
expected_after[marked_padded] *= -1

oracle_error = np.max(
    np.abs(after - expected_after)
)

# Probabilities must remain unchanged.
probability_error = np.max(
    np.abs(np.abs(after) ** 2 - np.abs(before) ** 2)
)

# Calculate the probability represented by marked states.
marked_probability = np.sum(
    np.abs(after[marked_padded]) ** 2
)

print("\nOracle verification")
print("Maximum phase error:", f"{oracle_error:.3e}")
print("Maximum probability error:", f"{probability_error:.3e}")
print("Marked probability:", f"{marked_probability:.12f}")


# ============================================================
# 4. Transpile the oracle
# ============================================================

print("\nTranspiling oracle...")

transpiled_oracle = transpile(
    oracle,
    basis_gates=["u", "cx"],
    optimization_level=1
)

counts = transpiled_oracle.count_ops()

print("\nOracle resource estimates")
print("Depth:", transpiled_oracle.depth())
print("Total gates:", transpiled_oracle.size())
print("U gates:", counts.get("u", 0))
print("CX gates:", counts.get("cx", 0))


# ============================================================
# 5. Save the oracle
# ============================================================

from qiskit import qpy

oracle_path = OUTPUT_DIR / "three_country_loss_oracle.qpy"

with open(oracle_path, "wb") as file:
    qpy.dump(transpiled_oracle, file)

print("\nSaved oracle:")
print(oracle_path)

print("\nORACLE CONSTRUCTION COMPLETE.")
