
from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

COUNTRY_COUNTS = [2, 3, 4, 5]

# Each country has:
# 5 earthquake levels x 2 precipitation states = 10 states.
STATES_PER_COUNTRY = 10

# Each probability is stored as an 8-byte float64.
BYTES_PER_PROBABILITY = 8


# --------------------------------------------------
# 2. CALCULATE SCALING
# --------------------------------------------------

def calculate_scaling():

    rows = []

    for countries in COUNTRY_COUNTS:

        # Number of possible joint states.
        classical_states = STATES_PER_COUNTRY ** countries

        # Minimum number of qubits.
        qubits = math.ceil(math.log2(classical_states))

        # Number of basis states available on the quantum register.
        quantum_dimension = 2 ** qubits

        # Unused basis states.
        unused_states = quantum_dimension - classical_states

        # Memory for a dense classical probability vector.
        classical_memory = (
            classical_states * BYTES_PER_PROBABILITY
        )

        # Memory for a dense quantum amplitude vector.
        quantum_vector_memory = (
            quantum_dimension * BYTES_PER_PROBABILITY
        )

        # Percentage of unused quantum basis states.
        padding_percentage = (
            unused_states / quantum_dimension * 100
        )

        rows.append({
            "Countries": countries,
            "Classical joint states": classical_states,
            "Minimum qubits": qubits,
            "Quantum basis states": quantum_dimension,
            "Unused basis states": unused_states,
            "Padding (%)": padding_percentage,
            "Classical vector memory (KB)": classical_memory / 1024,
            "Quantum vector memory (KB)": quantum_vector_memory / 1024
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 3. DISPLAY RESULTS
# --------------------------------------------------

def display_results(df):

    print("\nADB MULTI-COUNTRY SCALING ANALYSIS")
    print("=================================")

    print("\nState-space growth:")
    print("-------------------")

    for _, row in df.iterrows():

        print(
            f"{int(row['Countries'])} countries | "
            f"States: {int(row['Classical joint states']):,} | "
            f"Qubits: {int(row['Minimum qubits'])} | "
            f"Quantum dimension: "
            f"{int(row['Quantum basis states']):,} | "
            f"Unused: {int(row['Unused basis states']):,}"
        )

    print("\nMEMORY REQUIREMENTS")
    print("-------------------")

    for _, row in df.iterrows():

        print(
            f"{int(row['Countries'])} countries | "
            f"Classical vector: "
            f"{row['Classical vector memory (KB)']:.2f} KB | "
            f"Quantum amplitude vector: "
            f"{row['Quantum vector memory (KB)']:.2f} KB"
        )

    print("\nIMPORTANT")
    print("---------")
    print(
        "These are vector-storage estimates, "
        "not complete quantum circuit resource estimates."
    )


# --------------------------------------------------
# 4. SAVE RESULTS
# --------------------------------------------------

def save_results(df):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "scaling_analysis.csv"

    df.to_csv(output_file, index=False)

    print("\nSaved scaling table:")
    print(output_file)


# --------------------------------------------------
# 5. PLOT STATE-SPACE GROWTH
# --------------------------------------------------

def plot_scaling(df):

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["Countries"],
        df["Classical joint states"],
        marker="o",
        label="Classical joint states"
    )

    plt.plot(
        df["Countries"],
        df["Quantum basis states"],
        marker="s",
        label="Quantum basis states"
    )

    plt.yscale("log")

    plt.xlabel("Number of countries")
    plt.ylabel("Number of states (logarithmic scale)")

    plt.title("ADB Joint State-Space Scaling")

    plt.xticks(COUNTRY_COUNTS)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    output_file = OUTPUT_DIR / "scaling_analysis.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print("\nSaved scaling graph:")
    print(output_file)


# --------------------------------------------------
# 6. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    df = calculate_scaling()

    display_results(df)

    save_results(df)

    plot_scaling(df)

    print("\nSUCCESS: Scaling analysis completed.")
