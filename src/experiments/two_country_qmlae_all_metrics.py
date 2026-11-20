
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import xlogy

from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(MODEL_DIR))

from term_distribution import (
    calculate_term_distribution,
    calculate_losses,
    calculate_metrics,
)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUMBER_OF_QUBITS = 8
INDEX_QUBITS = 7

DIMENSION = 2 ** NUMBER_OF_QUBITS

SHOTS = 4096
GROVER_POWERS = {
    "Expected principal loss": [0, 1, 2],
    "Both countries pay": [0, 1, 2],
    "Portfolio loss exceeds 80m": [0, 1, 2, 4],
    "Expected stop-loss": [0, 1, 2, 4],
    "Both bonds exhausted": [0, 1, 2, 4],
}

RANDOM_SEED = 42
GRID_POINTS = 300_001

THRESHOLD = 80_000_000

# --------------------------------------------------
# LOAD ADB DISTRIBUTION
# --------------------------------------------------

print("\nTWO-COUNTRY ADB QMLAE: ALL FIVE METRICS")
print("=" * 60)

term = calculate_term_distribution(years=3)

states = sorted(term.keys())

probabilities = np.array(
    [term[state] for state in states],
    dtype=float,
)

probabilities /= probabilities.sum()

assert len(states) == 100
assert np.isclose(probabilities.sum(), 1.0)

# --------------------------------------------------
# CALCULATE LOSS AND EVENT VALUES
# --------------------------------------------------

loss_1_values = []
loss_2_values = []
total_loss_values = []

for state in states:

    loss_1, loss_2, total_loss = calculate_losses(state)

    loss_1_values.append(loss_1)
    loss_2_values.append(loss_2)
    total_loss_values.append(total_loss)

loss_1_values = np.array(loss_1_values, dtype=float)
loss_2_values = np.array(loss_2_values, dtype=float)
total_loss_values = np.array(total_loss_values, dtype=float)

stop_loss_values = np.maximum(
    total_loss_values - THRESHOLD,
    0,
)

# --------------------------------------------------
# DEFINE FIVE NORMALIZED METRICS
# --------------------------------------------------

metric_values = {

    "Expected principal loss": (
        total_loss_values / 160_000_000
    ),

    "Both countries pay": (
        (loss_1_values > 0) & (loss_2_values > 0)
    ).astype(float),

    "Portfolio loss exceeds 80m": (
        total_loss_values > THRESHOLD
    ).astype(float),

    "Expected stop-loss": (
        stop_loss_values / 80_000_000
    ),

    "Both bonds exhausted": (
        (loss_1_values == 80_000_000)
        & (loss_2_values == 80_000_000)
    ).astype(float),
}

scales = {
    "Expected principal loss": 160_000_000,
    "Both countries pay": 1.0,
    "Portfolio loss exceeds 80m": 1.0,
    "Expected stop-loss": 80_000_000,
    "Both bonds exhausted": 1.0,
}

exact_metrics = calculate_metrics(term)

# --------------------------------------------------
# PREPARE QMLAE STATE
# --------------------------------------------------

def prepare_metric_state(values):

    # For each original outcome i:
    #
    # ancilla=0 amplitude:
    # sqrt(p_i * (1 - x_i))
    #
    # ancilla=1 amplitude:
    # sqrt(p_i * x_i)
    #
    # x_i is the normalized metric value.

    amplitudes = np.zeros(DIMENSION, dtype=float)

    for index, (probability, value) in enumerate(
        zip(probabilities, values)
    ):

        amplitudes[index] = np.sqrt(
            probability * (1 - value)
        )

        amplitudes[index + 128] = np.sqrt(
            probability * value
        )

    assert np.isclose(np.sum(amplitudes ** 2), 1.0)

    return amplitudes


# --------------------------------------------------
# CREATE ZERO-STATE REFLECTION
# --------------------------------------------------

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


# --------------------------------------------------
# RUN QMLAE FOR ONE METRIC
# --------------------------------------------------

def estimate_metric(metric_name, values, rng):

    print(f"\nMETRIC: {metric_name}")
    print("-" * 60)

    amplitudes = prepare_metric_state(values)

    A = QuantumCircuit(NUMBER_OF_QUBITS)

    A.append(
        StatePreparation(amplitudes),
        range(NUMBER_OF_QUBITS),
    )

    # The last qubit is the event/expectation ancilla.
    # Z marks every state where that ancilla is 1.

    oracle = QuantumCircuit(NUMBER_OF_QUBITS)

    oracle.z(NUMBER_OF_QUBITS - 1)

    # Grover operator:
    # G = A S0 A† O
    #
    # Circuit composition applies O first.

    G = QuantumCircuit(NUMBER_OF_QUBITS)

    G.compose(oracle, inplace=True)
    G.compose(A.inverse(), inplace=True)
    G.compose(S0, inplace=True)
    G.compose(A, inplace=True)

    observations = []
    metric_powers = GROVER_POWERS[metric_name]

    for k in metric_powers:

        circuit = QuantumCircuit(NUMBER_OF_QUBITS)

        circuit.compose(A, inplace=True)

        for _ in range(k):
            circuit.compose(G, inplace=True)

        statevector = Statevector.from_instruction(circuit)

        measurement_probabilities = statevector.probabilities()

        # Ancilla=1 corresponds to basis indices 128...255.

        theoretical_probability = np.sum(
            measurement_probabilities[128:]
        )

        successes = rng.binomial(
            SHOTS,
            theoretical_probability,
        )

        observations.append({
            "Grover power": k,
            "Shots": SHOTS,
            "Successes": successes,
            "Theoretical probability": theoretical_probability,
            "Observed probability": successes / SHOTS,
        })

        print(
            f"k={k:2d} | "
            f"successes={successes:4d}/{SHOTS} | "
            f"observed={successes / SHOTS:.6%} | "
            f"theoretical={theoretical_probability:.6%}"
        )

    # --------------------------------------------------
    # MAXIMUM-LIKELIHOOD ESTIMATION
    # --------------------------------------------------

    k_max = max(metric_powers)

    theta_max = np.pi / (2 * (2 * k_max + 1))

    theta_grid = np.linspace(
        1e-8,
        theta_max,
        GRID_POINTS,
    )

    log_likelihood = np.zeros_like(theta_grid)

    for observation in observations:

        k = observation["Grover power"]

        successes = observation["Successes"]
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

    estimated_normalized_value = np.sin(
        estimated_theta
    ) ** 2

    estimated_value = (
        estimated_normalized_value * scales[metric_name]
    )

    # --------------------------------------------------
    # EXACT VALUE AND ERRORS
    # --------------------------------------------------

    exact_value = exact_metrics[metric_name]

    absolute_error = abs(
        estimated_value - exact_value
    )

    relative_error = (
        absolute_error / abs(exact_value) * 100
        if exact_value != 0
        else np.nan
    )

    print("\nQMLAE RESULT")
    print(f"Exact value:      {exact_value:,.10f}")
    print(f"Estimated value:  {estimated_value:,.10f}")
    print(f"Absolute error:   {absolute_error:,.10f}")
    print(f"Relative error:   {relative_error:.6f}%")

    total_shots = SHOTS * len(metric_powers)

    total_grover = (
        SHOTS * sum(metric_powers)
    )

    return {
        "Metric": metric_name,
        "Exact value": exact_value,
        "QMLAE estimate": estimated_value,
        "Absolute error": absolute_error,
        "Relative error (%)": relative_error,
        "Qubits": NUMBER_OF_QUBITS,
        "Shots per power": SHOTS,
        "Total shots": total_shots,
        "Grover applications": total_grover,
        "State preparations": total_shots,
        "Observations": observations,
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    rng = np.random.default_rng(RANDOM_SEED)

    results = []
    observation_rows = []

    for metric_name, values in metric_values.items():

        result = estimate_metric(
            metric_name,
            values,
            rng,
        )

        observation_rows.append({
            "Metric": metric_name,
            "Observations": result["Observations"],
        })

        results.append({
            key: value
            for key, value in result.items()
            if key != "Observations"
        })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = (
        OUTPUT_DIR / "two_country_qmlae_all_metrics.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    observation_rows_flat = []

    for item in observation_rows:

        for observation in item["Observations"]:

            row = {
                "Metric": item["Metric"],
                **observation,
            }

            observation_rows_flat.append(row)

    observations_df = pd.DataFrame(
        observation_rows_flat
    )

    observations_path = (
        OUTPUT_DIR / "two_country_qmlae_all_observations.csv"
    )

    observations_df.to_csv(
        observations_path,
        index=False,
    )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\nALL FIVE METRICS — QMLAE SUMMARY")
    print("=" * 60)

    print(
        results_df[
            [
                "Metric",
                "Exact value",
                "QMLAE estimate",
                "Absolute error",
                "Relative error (%)",
            ]
        ].to_string(index=False)
    )

    print("\nResource accounting per metric:")
    print(f"Qubits: {NUMBER_OF_QUBITS}")
    print(f"Shots: {SHOTS * len(GROVER_POWERS):,}")
    total_grover_applications = sum(
    result["Grover applications"]
    for result in results
    )

    print(
    f"Total Grover applications across five metrics: "
    f"{total_grover_applications:,}"
    )
    print("\nSaved:", results_path)
    print("Saved:", observations_path)

    print("\nEXPERIMENT COMPLETED")


if __name__ == "__main__":
    main()
