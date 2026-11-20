
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"

MC_FILE = OUTPUT / "three_country_monte_carlo.csv"
MLAE_FILE = OUTPUT / "three_country_amplitude_estimation.csv"

# --------------------------------------------------
# 2. LOAD EXPERIMENT RESULTS
# --------------------------------------------------

mc = pd.read_csv(MC_FILE)
mlae = pd.read_csv(MLAE_FILE)

# Exact probability from the classical benchmark
exact_probability = float(mc["exact_probability"].iloc[0])

# --------------------------------------------------
# 3. CLASSICAL MONTE CARLO RESOURCES
# --------------------------------------------------

mc_final = mc.iloc[-1]

mc_samples = int(mc_final["samples"])
mc_estimate = float(mc_final["estimate"])

mc_absolute_error = abs(mc_estimate - exact_probability)
mc_relative_error = (
    mc_absolute_error / exact_probability * 100
)

# --------------------------------------------------
# 4. QUANTUM MLAE RESOURCES
# --------------------------------------------------

quantum_estimate = 0.007213366830

quantum_absolute_error = abs(
    quantum_estimate - exact_probability
)

quantum_relative_error = (
    quantum_absolute_error / exact_probability * 100
)

measurement_shots = int(mlae["shots"].sum())

grover_applications = int(
    (
        mlae["grover_iterations"] * mlae["shots"]
    ).sum()
)

state_preparations = measurement_shots

# Transpiled circuit resource estimates
grover_cx = 3338
state_preparation_cx = 1013

grover_u = 2996
state_preparation_u = 1023

total_cx = (
    grover_applications * grover_cx
    + state_preparations * state_preparation_cx
)

total_u = (
    grover_applications * grover_u
    + state_preparations * state_preparation_u
)

total_elementary_gates = total_cx + total_u

# --------------------------------------------------
# 5. EXACT CLASSICAL ENUMERATION
# --------------------------------------------------

number_of_outcomes = 1000

# Exact enumeration evaluates all outcome probabilities
# and sums the probabilities of marked outcomes.

# --------------------------------------------------
# 6. DISPLAY COMPARISON
# --------------------------------------------------

print("=" * 75)
print("THREE-COUNTRY CLASSICAL VS QUANTUM RESOURCE COMPARISON")
print("=" * 75)

print(f"\nExact reference probability: {exact_probability:.10%}")

print("\nCLASSICAL MONTE CARLO")
print("-" * 45)
print(f"Samples: {mc_samples:,}")
print(f"Estimated probability: {mc_estimate:.10%}")
print(f"Relative error: {mc_relative_error:.6f}%")

print("\nQUANTUM MLAE")
print("-" * 45)
print(f"Measurement shots: {measurement_shots:,}")
print(f"Grover applications: {grover_applications:,}")
print(f"State preparations: {state_preparations:,}")
print(f"Estimated probability: {quantum_estimate:.10%}")
print(f"Relative error: {quantum_relative_error:.6f}%")

print("\nQUANTUM GATE RESOURCE ESTIMATES")
print("-" * 45)
print(f"Total CX gates: {total_cx:,}")
print(f"Total U gates: {total_u:,}")
print(f"Total elementary gates: {total_elementary_gates:,}")

print("\nEXACT CLASSICAL ENUMERATION")
print("-" * 45)
print(f"Outcomes evaluated: {number_of_outcomes:,}")
print(f"Exact probability: {exact_probability:.10%}")
print("Relative error: 0%")

# --------------------------------------------------
# 7. BUILD COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame([
    {
        "method": "Exact classical enumeration",
        "estimated_probability": exact_probability,
        "relative_error_percent": 0.0,
        "classical_samples_or_outcomes": number_of_outcomes,
        "quantum_measurement_shots": 0,
        "grover_applications": 0,
        "state_preparations": 0,
        "CX_gates": 0,
        "U_gates": 0
    },
    {
        "method": "Classical Monte Carlo",
        "estimated_probability": mc_estimate,
        "relative_error_percent": mc_relative_error,
        "classical_samples_or_outcomes": mc_samples,
        "quantum_measurement_shots": 0,
        "grover_applications": 0,
        "state_preparations": 0,
        "CX_gates": 0,
        "U_gates": 0
    },
    {
        "method": "Quantum MLAE",
        "estimated_probability": quantum_estimate,
        "relative_error_percent": quantum_relative_error,
        "classical_samples_or_outcomes": 0,
        "quantum_measurement_shots": measurement_shots,
        "grover_applications": grover_applications,
        "state_preparations": state_preparations,
        "CX_gates": total_cx,
        "U_gates": total_u
    }
])

comparison.to_csv(
    OUTPUT / "three_country_resource_comparison.csv",
    index=False
)

print("\nCOMPARISON TABLE")
print("-" * 75)

print(
    comparison[
        [
            "method",
            "estimated_probability",
            "relative_error_percent"
        ]
    ].to_string(index=False)
)

# --------------------------------------------------
# 8. SAVE A RESOURCE SUMMARY
# --------------------------------------------------

summary = f"""
THREE-COUNTRY RESOURCE SUMMARY

Exact probability: {exact_probability:.12%}

CLASSICAL MONTE CARLO
Samples: {mc_samples:,}
Relative error: {mc_relative_error:.6f}%

QUANTUM MLAE
Measurement shots: {measurement_shots:,}
Grover applications: {grover_applications:,}
State preparations: {state_preparations:,}
Relative error: {quantum_relative_error:.6f}%

QUANTUM GATES
CX gates: {total_cx:,}
U gates: {total_u:,}
Total elementary gates: {total_elementary_gates:,}

EXACT CLASSICAL ENUMERATION
Outcomes: {number_of_outcomes:,}
Relative error: 0%

IMPORTANT:
Classical samples, quantum shots, Grover applications,
and quantum gates are different resource units.
They must not be treated as directly equivalent.
"""

with open(
    OUTPUT / "three_country_resource_summary.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(summary)

print("\nSaved:")
print("outputs/three_country_resource_comparison.csv")
print("outputs/three_country_resource_summary.txt")

print("\nRESOURCE COMPARISON COMPLETE.")
