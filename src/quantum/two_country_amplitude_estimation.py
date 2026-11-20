
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit.quantum_info import Statevector


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(MODEL_DIR))

from term_distribution import (
    calculate_term_distribution,
    calculate_losses,
)

THRESHOLD = 80_000_000

NUMBER_OF_STATES = 100
NUMBER_OF_QUBITS = 7
DIMENSION = 2 ** NUMBER_OF_QUBITS

SHOTS = 4096
GROVER_POWERS = [0, 1, 2, 4]
RANDOM_SEED = 42


# ==================================================
# 2. LOAD THE OFFICIAL TWO-COUNTRY ADB DISTRIBUTION
# ==================================================

print("\nTWO-COUNTRY ADB QMLAE EXPERIMENT")
print("=" * 55)

term_distribution = calculate_term_distribution(years=3)

sorted_states = sorted(term_distribution.keys())

probabilities = np.array(
    [term_distribution[state] for state in sorted_states],
    dtype=float,
)

assert len(sorted_states) == NUMBER_OF_STATES
assert np.isclose(probabilities.sum(), 1.0)

print("Countries: Kyrgyz Republic and Tajikistan")
print("Term length: 3 years")
print(f"Number of valid outcomes: {len(sorted_states)}")
print(f"Probability sum: {probabilities.sum():.12f}")


# ==================================================
# 3. PAD THE DISTRIBUTION TO 128 QUANTUM STATES
# ==================================================

padded_probabilities = np.zeros(DIMENSION)
padded_probabilities[:NUMBER_OF_STATES] = probabilities

amplitudes = np.sqrt(padded_probabilities)

assert np.isclose(np.sum(amplitudes ** 2), 1.0)


# ==================================================
# 4. MARK PORTFOLIO LOSSES ABOVE $80 MILLION
# ==================================================

marked = np.zeros(DIMENSION, dtype=bool)

for index, state in enumerate(sorted_states):
    _, _, total_loss = calculate_losses(state)

    if total_loss > THRESHOLD:
        marked[index] = True

exact_probability = np.sum(
    padded_probabilities[marked]
)

print("\nEXACT CLASSICAL REFERENCE")
print("-" * 55)
print(f"Loss threshold: ${THRESHOLD:,.0f}")
print(f"Marked outcomes: {np.sum(marked)}")
print(f"Exact exceedance probability: {exact_probability:.10%}")


# ==================================================
# 5. CREATE THE QUANTUM STATE PREPARATION CIRCUIT
# ==================================================

A = QuantumCircuit(NUMBER_OF_QUBITS)

A.append(
    StatePreparation(amplitudes),
    range(NUMBER_OF_QUBITS),
)


# ==================================================
# 6. CREATE THE PHASE ORACLE
# ==================================================

diagonal = np.ones(DIMENSION)
diagonal[marked] = -1

oracle = QuantumCircuit(NUMBER_OF_QUBITS)

oracle.append(
    DiagonalGate(diagonal),
    range(NUMBER_OF_QUBITS),
)


# ==================================================
# 7. CREATE REFLECTION ABOUT THE ZERO STATE
# ==================================================

def create_zero_reflection():

    circuit = QuantumCircuit(NUMBER_OF_QUBITS)

    circuit.x(range(NUMBER_OF_QUBITS))

    circuit.h(NUMBER_OF_QUBITS - 1)

    circuit.mcx(
        list(range(NUMBER_OF_QUBITS - 1)),
        NUMBER_OF_QUBITS - 1,
    )

    circuit.h(NUMBER_OF_QUBITS - 1)

    circuit.x(range(NUMBER_OF_QUBITS))

    return circuit


S0 = create_zero_reflection()


# ==================================================
# 8. CREATE ONE GROVER ITERATION
# ==================================================

G = QuantumCircuit(NUMBER_OF_QUBITS)

G.compose(oracle, inplace=True)
G.compose(A.inverse(), inplace=True)
G.compose(S0, inplace=True)
G.compose(A, inplace=True)


# ==================================================
# 9. RUN THE QMLAE EXPERIMENT
# ==================================================

print("\nRUNNING QMLAE EXPERIMENT")
print("-" * 55)

rng = np.random.default_rng(RANDOM_SEED)

observations = []

for k in GROVER_POWERS:

    circuit = QuantumCircuit(NUMBER_OF_QUBITS)

    circuit.compose(A, inplace=True)

    for _ in range(k):
        circuit.compose(G, inplace=True)

    statevector = Statevector.from_instruction(circuit)

    measurement_probabilities = statevector.probabilities()

    theoretical_probability = np.sum(
        measurement_probabilities[marked]
    )

    successes = rng.binomial(
        SHOTS,
        theoretical_probability,
    )

    observed_probability = successes / SHOTS

    observations.append({
        "Grover power": k,
        "Shots": SHOTS,
        "Successful measurements": successes,
        "Observed probability": observed_probability,
        "Theoretical probability": theoretical_probability,
    })

    print(
        f"k={k:2d} | "
        f"successes={successes:4d}/{SHOTS} | "
        f"observed={observed_probability:.6%} | "
        f"theoretical={theoretical_probability:.6%}"
    )


# ==================================================
# 10. MAXIMUM-LIKELIHOOD ESTIMATION
# ==================================================

print("\nMAXIMUM-LIKELIHOOD ESTIMATION")
print("-" * 55)

# p = sin(theta)^2
#
# After k Grover iterations:
# P(success) = sin^2((2k + 1) * theta)
#
# This bounded interval avoids phase ambiguity
# for the selected Grover powers.

k_max = max(GROVER_POWERS)

theta_max = np.pi / (2 * (2 * k_max + 1))

theta_grid = np.linspace(
    1e-8,
    theta_max,
    300_001,
)

log_likelihood = np.zeros_like(theta_grid)

for observation in observations:

    k = observation["Grover power"]
    successes = observation["Successful measurements"]
    failures = SHOTS - successes

    predicted = np.sin(
        (2 * k + 1) * theta_grid
    ) ** 2

    predicted = np.clip(
        predicted,
        1e-15,
        1 - 1e-15,
    )

    log_likelihood += (
        xlogy(successes, predicted)
        + xlogy(failures, 1 - predicted)
    )

best_index = np.argmax(log_likelihood)

estimated_theta = theta_grid[best_index]

estimated_probability = np.sin(
    estimated_theta
) ** 2


# ==================================================
# 11. CALCULATE ESTIMATION ERROR
# ==================================================

absolute_error = abs(
    estimated_probability - exact_probability
)

relative_error = (
    absolute_error / exact_probability * 100
)


# ==================================================
# 12. ACCOUNT FOR QUANTUM RESOURCES
# ==================================================

total_shots = SHOTS * len(GROVER_POWERS)

total_grover_applications = (
    SHOTS * sum(GROVER_POWERS)
)

total_state_preparations = total_shots


# ==================================================
# 13. DISPLAY RESULTS
# ==================================================

print("\nFINAL TWO-COUNTRY QMLAE RESULTS")
print("=" * 55)

print(f"Exact probability:      {exact_probability:.10%}")
print(f"Estimated probability:  {estimated_probability:.10%}")
print(f"Absolute error:         {absolute_error:.10%}")
print(f"Relative error:         {relative_error:.6f}%")

print("\nRESOURCE ACCOUNTING")
print("-" * 55)

print(f"Number of qubits: {NUMBER_OF_QUBITS}")
print(f"Shots per circuit: {SHOTS:,}")
print(f"Total measurement shots: {total_shots:,}")
print(f"Total Grover applications: {total_grover_applications:,}")
print(f"Total state preparations: {total_state_preparations:,}")


# ==================================================
# 14. SAVE EXPERIMENT RESULTS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(observations).to_csv(
    OUTPUT_DIR / "two_country_qmlae_observations.csv",
    index=False,
)

summary = pd.DataFrame([{
    "Exact probability": exact_probability,
    "Estimated probability": estimated_probability,
    "Absolute error": absolute_error,
    "Relative error (%)": relative_error,
    "Qubits": NUMBER_OF_QUBITS,
    "Shots per circuit": SHOTS,
    "Total shots": total_shots,
    "Grover powers": str(GROVER_POWERS),
    "Grover applications": total_grover_applications,
    "State preparations": total_state_preparations,
}])

summary.to_csv(
    OUTPUT_DIR / "two_country_qmlae_results.csv",
    index=False,
)

print("\nSUCCESS")
print("Results saved in the outputs folder.")
