
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


# --------------------------------------------------
# 1. LOAD THE PROBABILITY DISTRIBUTION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "outputs" / "quantum_input_distribution.csv"

data = pd.read_csv(INPUT_FILE)

amplitudes = data["Amplitude"].to_numpy(dtype=float)
classical_probabilities = data["Probability"].to_numpy(dtype=float)
marked = data["Marked"].to_numpy(dtype=bool)

NUMBER_OF_QUBITS = 7

assert len(amplitudes) == 128
assert np.isclose(np.sum(amplitudes ** 2), 1.0)


# --------------------------------------------------
# 2. CREATE THE QUANTUM CIRCUIT
# --------------------------------------------------

print("\nQISKIT STATE PREPARATION")
print("========================")

circuit = QuantumCircuit(NUMBER_OF_QUBITS)

# Prepare the desired quantum state.
state_preparation = StatePreparation(amplitudes)

circuit.append(
    state_preparation,
    range(NUMBER_OF_QUBITS)
)

print("Number of qubits:", circuit.num_qubits)
print("Circuit instructions:", len(circuit.data))


# --------------------------------------------------
# 3. SIMULATE THE QUANTUM STATE
# --------------------------------------------------

statevector = Statevector.from_instruction(circuit)

quantum_probabilities = statevector.probabilities()

print("\nSTATE NORMALIZATION")
print("-------------------")

print(
    "Sum of quantum probabilities:",
    quantum_probabilities.sum()
)

assert np.isclose(quantum_probabilities.sum(), 1.0)


# --------------------------------------------------
# 4. COMPARE QUANTUM AND CLASSICAL DISTRIBUTIONS
# --------------------------------------------------

classical_error = np.max(
    np.abs(classical_probabilities - quantum_probabilities)
)

print("\nDISTRIBUTION VALIDATION")
print("-----------------------")

print("Maximum absolute probability error:", classical_error)

assert classical_error < 1e-10


# --------------------------------------------------
# 5. ESTIMATE THE EXCEEDANCE PROBABILITY
# --------------------------------------------------

quantum_exceedance = np.sum(
    quantum_probabilities[marked]
)

classical_exceedance = np.sum(
    classical_probabilities[marked]
)

print("\nPORTFOLIO EXCEEDANCE")
print("--------------------")

print(f"Classical probability: {classical_exceedance:.10%}")
print(f"Quantum probability:   {quantum_exceedance:.10%}")

assert np.isclose(
    quantum_exceedance,
    classical_exceedance,
    atol=1e-10
)


# --------------------------------------------------
# 6. DECOMPOSE THE CIRCUIT AND COUNT RESOURCES
# --------------------------------------------------

print("\nCIRCUIT RESOURCE ESTIMATE")
print("-------------------------")

compiled = transpile(
    circuit,
    basis_gates=["u", "cx"],
    optimization_level=1
)

print("Transpiled circuit depth:", compiled.depth())
print("Transpiled gate counts:", compiled.count_ops())
print("Number of CX gates:", compiled.count_ops().get("cx", 0))


# --------------------------------------------------
# 7. SAVE CIRCUIT INFORMATION
# --------------------------------------------------

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "quantum_state_preparation.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    file.write(str(compiled.draw(output="text")))

print("\nSaved decomposed circuit:")
print(OUTPUT_FILE)


# --------------------------------------------------
# 8. FINAL VALIDATION
# --------------------------------------------------

print("\nSUCCESS")
print("=======")
print("Qiskit successfully prepared the seven-qubit state.")
print("The simulated distribution matches the classical model.")
