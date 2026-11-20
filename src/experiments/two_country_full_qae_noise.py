
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import xlogy

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "model"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(MODEL_DIR))

from term_distribution import (
    calculate_term_distribution,
    calculate_losses,
)

THRESHOLD = 80_000_000

N_STATES = 100
N_QUBITS = 7
DIMENSION = 2 ** N_QUBITS

SHOTS = 4096
GROVER_POWERS = [0, 1, 2, 4]

NOISE_LEVELS = [
    0.0,
    0.0001,
    0.001,
    0.005,
    0.01,
]

REPETITIONS = 10
RANDOM_SEED = 42

# ==================================================
# 2. LOAD ADB DISTRIBUTION
# ==================================================

print("\nTWO-COUNTRY REPEATED QAE NOISE EXPERIMENT")
print("=" * 65)

term_distribution = calculate_term_distribution(years=3)

states = sorted(term_distribution.keys())

probabilities = np.array(
    [term_distribution[state] for state in states],
    dtype=float,
)

assert len(states) == N_STATES
assert np.isclose(probabilities.sum(), 1.0)

padded_probabilities = np.zeros(DIMENSION)
padded_probabilities[:N_STATES] = probabilities

amplitudes = np.sqrt(padded_probabilities)

# ==================================================
# 3. MARK PORTFOLIO LOSSES
# ==================================================

marked = np.zeros(DIMENSION, dtype=bool)

for index, state in enumerate(states):
    _, _, total_loss = calculate_losses(state)

    if total_loss > THRESHOLD:
        marked[index] = True

valid = np.zeros(DIMENSION, dtype=bool)
valid[:N_STATES] = True

exact_probability = float(
    padded_probabilities[marked].sum()
)

print("Countries: Kyrgyz Republic and Tajikistan")
print("Qubits:", N_QUBITS)
print("Valid states:", N_STATES)
print("Unused states:", DIMENSION - N_STATES)
print(f"Exact probability: {exact_probability:.10%}")

# ==================================================
# 4. BUILD A
# ==================================================

A = QuantumCircuit(N_QUBITS)

A.append(
    StatePreparation(amplitudes),
    range(N_QUBITS),
)

# ==================================================
# 5. BUILD ORACLE
# ==================================================

diagonal = np.ones(DIMENSION)
diagonal[marked] = -1

oracle = QuantumCircuit(N_QUBITS)

oracle.append(
    DiagonalGate(diagonal),
    range(N_QUBITS),
)

# ==================================================
# 6. BUILD ZERO REFLECTION
# ==================================================

def create_zero_reflection():

    circuit = QuantumCircuit(N_QUBITS)

    circuit.x(range(N_QUBITS))
    circuit.h(N_QUBITS - 1)

    circuit.mcx(
        list(range(N_QUBITS - 1)),
        N_QUBITS - 1,
    )

    circuit.h(N_QUBITS - 1)
    circuit.x(range(N_QUBITS))

    return circuit


S0 = create_zero_reflection()

# ==================================================
# 7. BUILD GROVER OPERATOR
# ==================================================

G = QuantumCircuit(N_QUBITS)

G.compose(oracle, inplace=True)
G.compose(A.inverse(), inplace=True)
G.compose(S0, inplace=True)
G.compose(A, inplace=True)

# ==================================================
# 8. BUILD COMPILED QAE CIRCUITS
# ==================================================

circuits = {}

print("\nCOMPILING CIRCUITS")

for k in GROVER_POWERS:

    circuit = QuantumCircuit(N_QUBITS, N_QUBITS)

    circuit.compose(A, inplace=True)

    for _ in range(k):
        circuit.compose(G, inplace=True)

    circuit.measure(
        range(N_QUBITS),
        range(N_QUBITS),
    )

    circuits[k] = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=1,
        seed_transpiler=RANDOM_SEED,
    )

    print(
        f"k={k}: depth={circuits[k].depth():,}, "
        f"gates={circuits[k].size():,}"
    )

# ==================================================
# 9. NOISE MODEL
# ==================================================

def make_noise_model(error_rate):

    noise = NoiseModel()

    if error_rate > 0:

        noise.add_all_qubit_quantum_error(
            depolarizing_error(error_rate, 1),
            ["u"],
        )

        noise.add_all_qubit_quantum_error(
            depolarizing_error(error_rate, 2),
            ["cx"],
        )

        noise.add_all_qubit_readout_error(
            ReadoutError([
                [1 - error_rate, error_rate],
                [error_rate, 1 - error_rate],
            ])
        )

    return noise

# ==================================================
# 10. MAXIMUM-LIKELIHOOD ESTIMATOR
# ==================================================

K_MAX = max(GROVER_POWERS)

THETA_MAX = np.pi / (2 * (2 * K_MAX + 1))

THETA_GRID = np.linspace(
    1e-8,
    THETA_MAX,
    300_001,
)


def estimate_probability(observations):

    log_likelihood = np.zeros_like(THETA_GRID)

    for observation in observations:

        k = observation["Grover power"]
        successes = observation["Marked counts"]
        failures = SHOTS - successes

        predicted = np.sin(
            (2 * k + 1) * THETA_GRID
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

    theta = THETA_GRID[best_index]

    return float(np.sin(theta) ** 2)

# ==================================================
# 11. RUN REPEATED NOISE EXPERIMENT
# ==================================================

raw_results = []
estimate_results = []

total_runs = len(NOISE_LEVELS) * REPETITIONS

run_number = 0

for noise_index, error_rate in enumerate(NOISE_LEVELS):

    print("\n" + "=" * 65)
    print(f"NOISE RATE: {error_rate:.3%}")
    print("=" * 65)

    simulator = AerSimulator(
        method="matrix_product_state",
        noise_model=make_noise_model(error_rate),
    )

    for repetition in range(REPETITIONS):

        run_number += 1

        observations = []

        total_invalid = 0
        total_measurements = 0

        for k in GROVER_POWERS:

            seed = (
                RANDOM_SEED
                + noise_index * 100_000
                + repetition * 100
                + k
            )

            result = simulator.run(
                circuits[k],
                shots=SHOTS,
                seed_simulator=seed,
            ).result()

            counts = result.get_counts()

            marked_count = 0
            invalid_count = 0

            for bitstring, count in counts.items():

                outcome = int(
                    bitstring.replace(" ", ""),
                    2,
                )

                if not valid[outcome]:
                    invalid_count += count

                if marked[outcome]:
                    marked_count += count

            observed_probability = marked_count / SHOTS

            observations.append({
                "Grover power": k,
                "Marked counts": marked_count,
            })

            raw_results.append({
                "Noise rate": error_rate,
                "Repetition": repetition + 1,
                "Grover power": k,
                "Shots": SHOTS,
                "Marked counts": marked_count,
                "Invalid counts": invalid_count,
                "Observed probability": observed_probability,
            })

            total_invalid += invalid_count
            total_measurements += SHOTS

        # Estimate probability using all four Grover powers
        estimate = estimate_probability(observations)

        absolute_error = abs(
            estimate - exact_probability
        )

        relative_error = (
            absolute_error / exact_probability * 100
        )

        invalid_rate = (
            total_invalid / total_measurements
        )

        estimate_results.append({
            "Noise rate": error_rate,
            "Repetition": repetition + 1,
            "Exact probability": exact_probability,
            "Estimated probability": estimate,
            "Absolute error": absolute_error,
            "Absolute error (pp)": absolute_error * 100,
            "Relative error (%)": relative_error,
            "Invalid-state rate": invalid_rate,
            "Total shots": total_measurements,
        })

        print(
            f"Run {run_number}/{total_runs} | "
            f"Repetition {repetition + 1}/{REPETITIONS} | "
            f"Estimate={estimate:.6%} | "
            f"Error={relative_error:.3f}% | "
            f"Invalid={invalid_rate:.3%}"
        )

# ==================================================
# 12. SUMMARIZE REPEATED RESULTS
# ==================================================

raw_df = pd.DataFrame(raw_results)
estimates_df = pd.DataFrame(estimate_results)

summary_df = (
    estimates_df
    .groupby("Noise rate")
    .agg(
        Mean_estimate=("Estimated probability", "mean"),
        SD_estimate=("Estimated probability", "std"),
        Mean_absolute_error_pp=("Absolute error (pp)", "mean"),
        SD_absolute_error_pp=("Absolute error (pp)", "std"),
        Mean_relative_error=("Relative error (%)", "mean"),
        SD_relative_error=("Relative error (%)", "std"),
        Mean_invalid_state_rate=("Invalid-state rate", "mean"),
        SD_invalid_state_rate=("Invalid-state rate", "std"),
    )
    .reset_index()
)

summary_df = summary_df.rename(columns={
    "Mean_estimate": "Mean estimated probability",
    "SD_estimate": "SD estimated probability",
    "Mean_absolute_error_pp": "Mean absolute error (pp)",
    "SD_absolute_error_pp": "SD absolute error (pp)",
    "Mean_relative_error": "Mean relative error (%)",
    "SD_relative_error": "SD relative error (%)",
    "Mean_invalid_state_rate": "Mean invalid-state rate",
    "SD_invalid_state_rate": "SD invalid-state rate",
})

# ==================================================
# 13. SAVE RESULTS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

raw_df.to_csv(
    OUTPUT_DIR / "two_country_noise_repeated_raw.csv",
    index=False,
)

estimates_df.to_csv(
    OUTPUT_DIR / "two_country_noise_repeated_estimates.csv",
    index=False,
)

summary_df.to_csv(
    OUTPUT_DIR / "two_country_noise_repeated_summary.csv",
    index=False,
)

# ==================================================
# 14. PLOT MEAN ERROR WITH STANDARD DEVIATION
# ==================================================

plt.figure(figsize=(9, 6))

plt.errorbar(
    summary_df["Noise rate"] * 100,
    summary_df["Mean absolute error (pp)"],
    yerr=summary_df["SD absolute error (pp)"],
    marker="o",
    capsize=5,
)

plt.xlabel("Depolarizing / readout error rate (%)")
plt.ylabel("Mean absolute estimation error (percentage points)")
plt.title("Two-Country ADB QAE: Repeated Noise Experiment")
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "two_country_repeated_error_vs_noise.png",
    dpi=300,
)

plt.close()

# ==================================================
# 15. DISPLAY SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("REPEATED NOISE EXPERIMENT SUMMARY")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nSAVED OUTPUTS:")
print("two_country_noise_repeated_raw.csv")
print("two_country_noise_repeated_estimates.csv")
print("two_country_noise_repeated_summary.csv")
print("two_country_repeated_error_vs_noise.png")

print("\nEXPERIMENT COMPLETED")
