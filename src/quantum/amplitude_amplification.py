
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit.quantum_info import Statevector


# --------------------------------------------------
# 1. LOAD THE QUANTUM INPUT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "outputs" / "quantum_input_distribution.csv"

data = pd.read_csv(INPUT_FILE)

amplitudes = data["Amplitude"].to_numpy(dtype=float)
marked = data["Marked"].astype(bool).to_numpy()

NUMBER_OF_QUBITS = 7
DIMENSION = 2 ** NUMBER_OF_QUBITS

assert len(amplitudes) == DIMENSION
assert len(marked) == DIMENSION

initial_probability = np.sum(amplitudes[marked] ** 2)

print("\nAMPLITUDE AMPLIFICATION")
print("=======================")

print(f"Initial exceedance probability: {initial_probability:.8%}")
print("Number of marked states:", np.sum(marked))


# --------------------------------------------------
# 2. PREPARE THE INITIAL STATE
# --------------------------------------------------

A = QuantumCircuit(NUMBER_OF_QUBITS)

A.append(
    StatePreparation(amplitudes),
    range(NUMBER_OF_QUBITS)
)


# --------------------------------------------------
# 3. CONSTRUCT THE PHASE ORACLE
# --------------------------------------------------

# Marked states receive a phase of -1.
# Unmarked states receive a phase of +1.

diagonal = np.ones(DIMENSION)

diagonal[marked] = -1

oracle_gate = DiagonalGate(diagonal)

oracle = QuantumCircuit(NUMBER_OF_QUBITS)

oracle.append(
    oracle_gate,
    range(NUMBER_OF_QUBITS)
)


# --------------------------------------------------
# 4. CONSTRUCT REFLECTION ABOUT ZERO
# --------------------------------------------------

def create_zero_reflection():

    reflection = QuantumCircuit(NUMBER_OF_QUBITS)

    reflection.x(range(NUMBER_OF_QUBITS))

    reflection.h(NUMBER_OF_QUBITS - 1)

    reflection.mcx(
        list(range(NUMBER_OF_QUBITS - 1)),
        NUMBER_OF_QUBITS - 1
    )

    reflection.h(NUMBER_OF_QUBITS - 1)

    reflection.x(range(NUMBER_OF_QUBITS))

    return reflection


S0 = create_zero_reflection()


# --------------------------------------------------
# 5. CONSTRUCT ONE GROVER ITERATION
# --------------------------------------------------

# Q = A S0 A† S_good
#
# Circuit operations are applied from left to right
# in the order they are appended.

grover_iteration = QuantumCircuit(NUMBER_OF_QUBITS)

grover_iteration.compose(oracle, inplace=True)
grover_iteration.compose(A.inverse(), inplace=True)
grover_iteration.compose(S0, inplace=True)
grover_iteration.compose(A, inplace=True)


# --------------------------------------------------
# 6. CALCULATE THE OPTIMAL ITERATION COUNT
# --------------------------------------------------

theta = np.arcsin(np.sqrt(initial_probability))

optimal_iterations = round(
    np.pi / (4 * theta) - 0.5
)

print("\nAMPLIFICATION PARAMETERS")
print("------------------------")

print(f"Theta: {theta:.8f} radians")
print("Optimal Grover iterations:", optimal_iterations)


# --------------------------------------------------
# 7. BUILD THE COMPLETE AMPLIFICATION CIRCUIT
# --------------------------------------------------

amplified_circuit = QuantumCircuit(NUMBER_OF_QUBITS)

# Begin with the prepared probability distribution.
amplified_circuit.compose(A, inplace=True)

# Repeat the Grover iteration.
for _ in range(optimal_iterations):
    amplified_circuit.compose(
        grover_iteration,
        inplace=True
    )


# --------------------------------------------------
# 8. SIMULATE THE AMPLIFIED STATE
# --------------------------------------------------

final_state = Statevector.from_instruction(
    amplified_circuit
)

final_probabilities = final_state.probabilities()

amplified_probability = np.sum(
    final_probabilities[marked]
)

theoretical_probability = np.sin(
    (2 * optimal_iterations + 1) * theta
) ** 2


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

print("\nAMPLIFICATION RESULTS")
print("---------------------")

print(
    f"Initial probability:     "
    f"{initial_probability:.8%}"
)

print(
    f"Amplified probability:   "
    f"{amplified_probability:.8%}"
)

print(
    f"Theoretical probability: "
    f"{theoretical_probability:.8%}"
)

print(
    f"Amplification factor:    "
    f"{amplified_probability / initial_probability:.2f}x"
)

print("\nCIRCUIT INFORMATION")
print("-------------------")

print("Qubits:", amplified_circuit.num_qubits)
print("Circuit depth (high-level):", amplified_circuit.depth())
print("Circuit instructions:", len(amplified_circuit.data))


# --------------------------------------------------
# 10. VALIDATION
# --------------------------------------------------

assert np.isclose(
    amplified_probability,
    theoretical_probability,
    atol=1e-10
)

assert amplified_probability > initial_probability

print("\nSUCCESS: Amplitude amplification validated.")
