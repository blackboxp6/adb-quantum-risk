
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "src" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(MODEL_DIR))

from term_distribution import (
    calculate_term_distribution,
    calculate_losses
)


# --------------------------------------------------
# 2. QUANTUM REGISTER CONFIGURATION
# --------------------------------------------------

NUMBER_OF_STATES = 100
NUMBER_OF_QUBITS = 7
QUANTUM_DIMENSION = 2 ** NUMBER_OF_QUBITS

THRESHOLD = 80_000_000


# --------------------------------------------------
# 3. LOAD EXACT CLASSICAL DISTRIBUTION
# --------------------------------------------------

def prepare_probability_vector():

    print("\nLOADING ADB THREE-YEAR DISTRIBUTION")
    print("-----------------------------------")

    term_distribution = calculate_term_distribution(years=3)

    # Sort states to guarantee consistent indexing.
    sorted_states = sorted(term_distribution.keys())

    probabilities = np.array(
        [term_distribution[state] for state in sorted_states],
        dtype=float
    )

    print("Number of classical states:", len(sorted_states))
    print("Total classical probability:", probabilities.sum())

    assert len(sorted_states) == NUMBER_OF_STATES
    assert np.isclose(probabilities.sum(), 1.0)

    return sorted_states, probabilities


# --------------------------------------------------
# 4. PAD DISTRIBUTION TO 128 STATES
# --------------------------------------------------

def pad_probability_vector(probabilities):

    padded = np.zeros(QUANTUM_DIMENSION)

    padded[:NUMBER_OF_STATES] = probabilities

    print("\nQUANTUM REGISTER")
    print("----------------")
    print("Number of qubits:", NUMBER_OF_QUBITS)
    print("Quantum basis states:", QUANTUM_DIMENSION)
    print("Valid ADB states:", NUMBER_OF_STATES)
    print("Unused basis states:", QUANTUM_DIMENSION - NUMBER_OF_STATES)

    assert len(padded) == QUANTUM_DIMENSION
    assert np.isclose(padded.sum(), 1.0)

    return padded


# --------------------------------------------------
# 5. CONVERT PROBABILITIES INTO AMPLITUDES
# --------------------------------------------------

def calculate_amplitudes(probabilities):

    amplitudes = np.sqrt(probabilities)

    # Quantum normalization condition:
    # Sum of squared amplitudes must equal one.

    normalization = np.sum(amplitudes ** 2)

    print("\nQUANTUM STATE NORMALIZATION")
    print("--------------------------")
    print("Sum of squared amplitudes:", normalization)

    assert np.isclose(normalization, 1.0)

    return amplitudes


# --------------------------------------------------
# 6. MARK STATES EXCEEDING THE LOSS THRESHOLD
# --------------------------------------------------

def create_marked_states(sorted_states):

    marked = np.zeros(QUANTUM_DIMENSION, dtype=bool)

    for index, state in enumerate(sorted_states):

        _, _, total_loss = calculate_losses(state)

        if total_loss > THRESHOLD:
            marked[index] = True

    return marked


# --------------------------------------------------
# 7. CALCULATE THE QUANTUM AMPLITUDE
# --------------------------------------------------

def calculate_exceedance_probability(amplitudes, marked):

    # Probability of measuring a marked state:
    # Sum of squared amplitudes for marked states.

    amplitude = np.sum(amplitudes[marked] ** 2)

    print("\nPORTFOLIO EXCEEDANCE")
    print("--------------------")
    print("Marked states:", np.sum(marked))
    print(f"Estimated probability: {amplitude:.8%}")

    return amplitude


# --------------------------------------------------
# 8. SAVE QUANTUM INPUT DISTRIBUTION
# --------------------------------------------------

def save_distribution(states, probabilities, amplitudes, marked):

    rows = []

    for index in range(QUANTUM_DIMENSION):

        if index < NUMBER_OF_STATES:
            state = str(states[index])
        else:
            state = "Unused padding state"

        rows.append({
            "Basis index": index,
            "ADB state": state,
            "Probability": probabilities[index],
            "Amplitude": amplitudes[index],
            "Marked": bool(marked[index])
        })

    dataframe = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "quantum_input_distribution.csv"

    dataframe.to_csv(output_file, index=False)

    print("\nSaved quantum input:")
    print(output_file)


# --------------------------------------------------
# 9. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    states, classical_probabilities = prepare_probability_vector()

    padded_probabilities = pad_probability_vector(
        classical_probabilities
    )

    amplitudes = calculate_amplitudes(
        padded_probabilities
    )

    marked = create_marked_states(states)

    quantum_probability = calculate_exceedance_probability(
        amplitudes,
        marked
    )

    # Compare against the exact classical result.
    exact_probability = sum(
        classical_probabilities[i]
        for i in range(NUMBER_OF_STATES)
        if marked[i]
    )

    print("\nVALIDATION")
    print("----------")
    print(f"Exact classical probability: {exact_probability:.8%}")
    print(f"Quantum input probability:   {quantum_probability:.8%}")

    assert np.isclose(
        quantum_probability,
        exact_probability,
        atol=1e-12
    )

    print("\nSUCCESS: Quantum input distribution is valid.")

    save_distribution(
        states,
        padded_probabilities,
        amplitudes,
        marked
    )
