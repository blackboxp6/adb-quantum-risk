
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

BASE = "outputs/"

# Load state preparation
with open(BASE + "three_country_state_preparation.qpy", "rb") as f:
    A = qpy.load(f)[0]

# Load Grover operator
with open(BASE + "three_country_grover.qpy", "rb") as f:
    G = qpy.load(f)[0]

# Load the marking information
data = np.load(BASE + "three_country_quantum_input.npz")

marked = data["marked"]
dimension = 2 ** A.num_qubits

# Pad marking array to quantum dimension
marked_padded = np.zeros(dimension, dtype=bool)
marked_padded[:len(marked)] = marked

print("=" * 55)
print("THREE-COUNTRY GROVER VERIFICATION")
print("=" * 55)

# Prepare initial quantum state
initial = Statevector.from_instruction(A)

# Apply one Grover iteration
circuit = QuantumCircuit(A.num_qubits)
circuit.compose(A, inplace=True)
circuit.compose(G, inplace=True)

final_state = Statevector.from_instruction(circuit)

# Calculate marked probabilities
initial_probability = np.sum(
    np.abs(initial.data[marked_padded]) ** 2
)

final_probability = np.sum(
    np.abs(final_state.data[marked_padded]) ** 2
)

# Theoretical prediction
theta = np.arcsin(np.sqrt(initial_probability))
theoretical_probability = np.sin(3 * theta) ** 2

print(f"\nInitial marked probability: {initial_probability:.12%}")
print(f"Theoretical after Grover:  {theoretical_probability:.12%}")
print(f"Actual after Grover:       {final_probability:.12%}")

print("\nVerification errors")
print(
    "Absolute error:",
    abs(final_probability - theoretical_probability)
)

print(
    "Probability normalization error:",
    abs(np.sum(np.abs(final_state.data) ** 2) - 1.0)
)

print("\nGROVER VERIFICATION COMPLETE.")
