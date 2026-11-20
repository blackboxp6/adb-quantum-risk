
from pathlib import Path
import sys

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, DiagonalGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)

# ==================================================
# 1. PROJECT CONFIGURATION
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

NUMBER_OF_STATES = 100
NUMBER_OF_QUBITS = 7
DIMENSION = 2 ** NUMBER_OF_QUBITS

SHOTS = 256
NOISE_LEVELS = [0.0, 0.001]

RANDOM_SEED = 42

# ==================================================
# 2. LOAD THE OFFICIAL ADB DISTRIBUTION
# ==================================================

print("\nTWO-COUNTRY ADB NOISE EXPERIMENT")
print("=" * 55)

term_distribution = calculate_term_distribution(years=3)

sorted_states = sorted(term_distribution.keys())

probabilities = np.array(
    [term_distribution[state] for state in sorted_states],
    dtype=float,
)

assert len(sorted_states) == NUMBER_OF_STATES
assert np.isclose(probabilities.sum(), 1.0)

# ==================================================
# 3. PAD TO 128 QUANTUM STATES
# ==================================================

padded_probabilities = np.zeros(DIMENSION)
padded_probabilities[:NUMBER_OF_STATES] = probabilities

amplitudes = np.sqrt(padded_probabilities)

# ==================================================
# 4. CREATE THE LOSS MARKING FUNCTION
# ==================================================

marked = np.zeros(DIMENSION, dtype=bool)

for index, state in enumerate(sorted_states):
    _, _, total_loss = calculate_losses(state)

    if total_loss > THRESHOLD:
        marked[index] = True

exact_probability = float(
    padded_probabilities[marked].sum()
)

print("Countries: Kyrgyz Republic and Tajikistan")
print("Qubits:", NUMBER_OF_QUBITS)
print("Valid outcomes:", NUMBER_OF_STATES)
print(f"Threshold: ${THRESHOLD:,.0f}")
print(f"Exact probability: {exact_probability:.10%}")

# ==================================================
# 5. BUILD STATE PREPARATION
# ==================================================

A = QuantumCircuit(NUMBER_OF_QUBITS)

A.append(
    StatePreparation(amplitudes),
    range(NUMBER_OF_QUBITS),
)

# ==================================================
# 6. BUILD THE MEASUREMENT CIRCUIT
# ==================================================

qc = QuantumCircuit(NUMBER_OF_QUBITS, NUMBER_OF_QUBITS)

qc.compose(A, inplace=True)

qc.measure(
    range(NUMBER_OF_QUBITS),
    range(NUMBER_OF_QUBITS),
)

# Compile high-level gates into the noise model's
# supported basis gates.

compiled_qc = transpile(
    qc,
    basis_gates=["u", "cx"],
    optimization_level=1,
    seed_transpiler=RANDOM_SEED,
)

print("\nCOMPILED CIRCUIT")
print("-" * 55)
print("Depth:", compiled_qc.depth())
print("Operations:", compiled_qc.count_ops())

# ==================================================
# 7. CREATE THE NOISE MODEL
# ==================================================

def make_noise_model(error_rate):

    noise = NoiseModel()

    if error_rate > 0:

        single_error = depolarizing_error(
            error_rate, 1
        )

        two_error = depolarizing_error(
            error_rate, 2
        )

        readout_error = ReadoutError([
            [1 - error_rate, error_rate],
            [error_rate, 1 - error_rate],
        ])

        noise.add_all_qubit_quantum_error(
            single_error, ["u"]
        )

        noise.add_all_qubit_quantum_error(
            two_error, ["cx"]
        )

        noise.add_all_qubit_readout_error(
            readout_error
        )

    return noise

# ==================================================
# 8. RUN IDEAL AND NOISY SIMULATIONS
# ==================================================

results = []

for noise_index, error_rate in enumerate(NOISE_LEVELS):

    print("\n" + "-" * 55)
    print(f"NOISE LEVEL: {error_rate:.3%}")
    print("-" * 55)

    noise_model = make_noise_model(error_rate)

    simulator = AerSimulator(
        method="matrix_product_state",
        noise_model=noise_model,
    )

    result = simulator.run(
        compiled_qc,
        shots=SHOTS,
        seed_simulator=RANDOM_SEED + noise_index,
    ).result()

    counts = result.get_counts()

    marked_count = 0

    for bitstring, count in counts.items():

        outcome = int(
            bitstring.replace(" ", ""),
            2,
        )

        if outcome < NUMBER_OF_STATES:
            if marked[outcome]:
                marked_count += count

    estimated_probability = marked_count / SHOTS

    absolute_error = abs(
        estimated_probability - exact_probability
    )

    results.append({
        "Noise rate": error_rate,
        "Shots": SHOTS,
        "Marked counts": marked_count,
        "Estimated probability": estimated_probability,
        "Exact probability": exact_probability,
        "Absolute error": absolute_error,
    })

    print("Marked counts:", marked_count)
    print(
        f"Estimated probability: "
        f"{estimated_probability:.10%}"
    )
    print(
        f"Exact probability: "
        f"{exact_probability:.10%}"
    )
    print(
        f"Absolute error: "
        f"{absolute_error:.10%}"
    )

# ==================================================
# 9. SAVE RESULTS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import pandas as pd

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "two_country_noise_test.csv",
    index=False,
)

print("\nSUCCESS")
print("Saved: outputs/two_country_noise_test.csv")
