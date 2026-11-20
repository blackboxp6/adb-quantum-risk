
import numpy as np
import pandas as pd

from scipy.special import xlogy
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

BASE = "outputs/"

# --------------------------------------------------
# 1. LOAD QUANTUM CIRCUITS
# --------------------------------------------------

with open(BASE + "three_country_state_preparation.qpy", "rb") as f:
    A = qpy.load(f)[0]

with open(BASE + "three_country_grover.qpy", "rb") as f:
    G = qpy.load(f)[0]

data = np.load(BASE + "three_country_quantum_input.npz")
marked = data["marked"]

n = A.num_qubits
dimension = 2 ** n

marked_padded = np.zeros(dimension, dtype=bool)
marked_padded[:len(marked)] = marked

# --------------------------------------------------
# 2. EXPERIMENT SETTINGS
# --------------------------------------------------

# Number of Grover iterations used in each experiment
grover_iterations = [0, 1, 2, 4]

# Number of measurement shots per experiment
shots = 4096

rng = np.random.default_rng(42)

# --------------------------------------------------
# 3. RUN IDEAL STATEVECTOR EXPERIMENTS
# --------------------------------------------------

print("=" * 60)
print("THREE-COUNTRY MAXIMUM-LIKELIHOOD AMPLITUDE ESTIMATION")
print("=" * 60)

results = []

for k in grover_iterations:

    circuit = QuantumCircuit(n)
    circuit.compose(A, inplace=True)

    for _ in range(k):
        circuit.compose(G, inplace=True)

    state = Statevector.from_instruction(circuit)

    # Probability of measuring a marked outcome
    probability = np.sum(
        np.abs(state.data[marked_padded]) ** 2
    )

    # Simulate finite measurement shots
    successes = rng.binomial(shots, probability)

    results.append({
        "grover_iterations": k,
        "shots": shots,
        "successes": successes,
        "observed_probability": successes / shots,
        "ideal_probability": probability
    })

    print(
        f"k={k:2d} | "
        f"successes={successes:4d}/{shots} | "
        f"observed={successes/shots:.6%} | "
        f"ideal={probability:.6%}"
    )

# --------------------------------------------------
# 4. MAXIMUM-LIKELIHOOD ESTIMATION
# --------------------------------------------------

# For amplitude a = sin(theta)^2:
#
# P(success | k, theta) = sin^2((2k+1)*theta)
#
# We search a bounded theta interval that avoids
# phase ambiguity for the largest tested k.

k_max = max(grover_iterations)

theta_max = np.pi / (2 * (2 * k_max + 1))

theta_grid = np.linspace(
    1e-8,
    theta_max,
    300001
)

log_likelihood = np.zeros_like(theta_grid)

for result in results:

    k = result["grover_iterations"]
    successes = result["successes"]
    failures = shots - successes

    predicted = np.sin((2 * k + 1) * theta_grid) ** 2

    predicted = np.clip(
        predicted,
        1e-15,
        1 - 1e-15
    )

    log_likelihood += (
        xlogy(successes, predicted)
        + xlogy(failures, 1 - predicted)
    )

best_index = np.argmax(log_likelihood)

estimated_theta = theta_grid[best_index]
estimated_probability = np.sin(estimated_theta) ** 2

# --------------------------------------------------
# 5. COMPARE WITH EXACT CLASSICAL ANSWER
# --------------------------------------------------

exact_probability = np.sum(
    data["probabilities"][marked]
)

absolute_error = abs(
    estimated_probability - exact_probability
)

relative_error = (
    absolute_error / exact_probability * 100
)

print("\n" + "=" * 60)
print("AMPLITUDE ESTIMATION RESULT")
print("=" * 60)

print(f"Exact probability:      {exact_probability:.10%}")
print(f"Estimated probability: {estimated_probability:.10%}")
print(f"Absolute error:        {absolute_error:.10%}")
print(f"Relative error:        {relative_error:.6f}%")

print(f"\nTotal measurement shots: {shots * len(results):,}")

# Count total Grover iterations across all shots.
total_grover_applications = sum(
    r["grover_iterations"] * shots
    for r in results
)

print(
    "Total Grover applications across shots:",
    f"{total_grover_applications:,}"
)

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

df = pd.DataFrame(results)
df.to_csv(
    BASE + "three_country_amplitude_estimation.csv",
    index=False
)

print("\nResults saved.")
print("THREE-COUNTRY AMPLITUDE ESTIMATION COMPLETE.")
