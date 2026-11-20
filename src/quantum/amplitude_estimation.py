
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit.quantum_info import Statevector


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "outputs" / "quantum_input_distribution.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

N_QUBITS = 7
DIMENSION = 2 ** N_QUBITS

SHOTS = 4096
GROVER_POWERS = [0, 1, 2, 4, 8]

RANDOM_SEED = 42


# --------------------------------------------------
# 2. LOAD QUANTUM DISTRIBUTION
# --------------------------------------------------

print("\nMAXIMUM-LIKELIHOOD AMPLITUDE ESTIMATION")
print("======================================")

data = pd.read_csv(INPUT_FILE)

amplitudes = data["Amplitude"].to_numpy(dtype=float)
marked = data["Marked"].astype(bool).to_numpy()

assert len(amplitudes) == DIMENSION
assert len(marked) == DIMENSION

exact_probability = np.sum(amplitudes[marked] ** 2)

print(f"Exact probability: {exact_probability:.8%}")
print("Shots per circuit:", SHOTS)
print("Grover powers:", GROVER_POWERS)


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
# 5. REFLECTION ABOUT ZERO
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
# 6. GROVER ITERATION
# --------------------------------------------------

Q = QuantumCircuit(N_QUBITS)

Q.compose(oracle, inplace=True)
Q.compose(A.inverse(), inplace=True)
Q.compose(S0, inplace=True)
Q.compose(A, inplace=True)


# --------------------------------------------------
# 7. BUILD CIRCUIT FOR EACH GROVER POWER
# --------------------------------------------------

def create_grover_circuit(k):

    circuit = QuantumCircuit(N_QUBITS)

    circuit.compose(A, inplace=True)

    for _ in range(k):
        circuit.compose(Q, inplace=True)

    return circuit


# --------------------------------------------------
# 8. STATEVECTOR SIMULATION AND SHOT SAMPLING
# --------------------------------------------------

print("\nRUNNING QUANTUM CIRCUITS")
print("------------------------")

rng = np.random.default_rng(RANDOM_SEED)

observations = []

for k in GROVER_POWERS:

    print(f"Simulating Grover power k={k}...")

    circuit = create_grover_circuit(k)

    # Calculate the final quantum state.
    state = Statevector.from_instruction(circuit)

    # Calculate the probability of measuring a marked state.
    probabilities = state.probabilities()

    theoretical_success_probability = np.sum(
        probabilities[marked]
    )

    # Simulate the number of successful measurements.
    successes = rng.binomial(
        SHOTS,
        theoretical_success_probability
    )

    observed_probability = successes / SHOTS

    observations.append({
        "Grover iterations": k,
        "Shots": SHOTS,
        "Successful measurements": successes,
        "Observed probability": observed_probability,
        "Theoretical probability": theoretical_success_probability
    })

    print(
        f"  Successes: {successes}/{SHOTS}"
    )

    print(
        f"  Observed probability: {observed_probability:.6f}"
    )


# --------------------------------------------------
# 9. MAXIMUM-LIKELIHOOD ESTIMATION
# --------------------------------------------------

print("\nCALCULATING MAXIMUM LIKELIHOOD")
print("------------------------------")

# Search over theta.
theta_grid = np.linspace(
    0.000001,
    np.pi / 2 - 0.000001,
    300_000
)

log_likelihood = np.zeros_like(theta_grid)

for observation in observations:

    k = observation["Grover iterations"]

    successes = observation["Successful measurements"]
    failures = SHOTS - successes

    predicted_probability = np.sin(
        (2 * k + 1) * theta_grid
    ) ** 2

    predicted_probability = np.clip(
        predicted_probability,
        1e-12,
        1 - 1e-12
    )

    log_likelihood += (
        successes * np.log(predicted_probability)
        + failures * np.log(1 - predicted_probability)
    )


best_index = np.argmax(log_likelihood)

estimated_theta = theta_grid[best_index]

estimated_probability = np.sin(
    estimated_theta
) ** 2


# --------------------------------------------------
# 10. RESULTS
# --------------------------------------------------

absolute_error = abs(
    estimated_probability - exact_probability
)

relative_error = (
    absolute_error / exact_probability * 100
)

print("\nAMPLITUDE ESTIMATION RESULTS")
print("----------------------------")

print(f"Exact probability:     {exact_probability:.8%}")
print(f"Estimated probability: {estimated_probability:.8%}")

print(f"Absolute error: {absolute_error:.10f}")
print(f"Relative error: {relative_error:.4f}%")


# --------------------------------------------------
# 11. QUERY ACCOUNTING
# --------------------------------------------------

total_shots = SHOTS * len(GROVER_POWERS)

total_oracle_queries = SHOTS * sum(GROVER_POWERS)

print("\nQUERY ACCOUNTING")
print("----------------")

print("Total measurement shots:", total_shots)
print("Total Grover oracle calls:", total_oracle_queries)


# --------------------------------------------------
# 12. SAVE RESULTS
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(observations).to_csv(
    OUTPUT_DIR / "amplitude_estimation_observations.csv",
    index=False
)

pd.DataFrame([{
    "Exact probability": exact_probability,
    "Estimated probability": estimated_probability,
    "Absolute error": absolute_error,
    "Relative error (%)": relative_error,
    "Total shots": total_shots,
    "Grover oracle calls": total_oracle_queries
}]).to_csv(
    OUTPUT_DIR / "amplitude_estimation_results.csv",
    index=False
)

print("\nSUCCESS")
print("=======")
print("Amplitude estimation completed.")
print("Results saved in the outputs folder.")
