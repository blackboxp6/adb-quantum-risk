
from pathlib import Path
import pandas as pd

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# 2. EXPERIMENT RESULTS
# Values taken from the completed experiment runs.
# Absolute error is expressed in percentage points.
# ==================================================

results = [
    {
        "countries": 2,
        "valid_outcomes": 100,
        "qubits": 7,
        "exact_probability_percent": 1.2046495708,
        "estimated_probability_percent": 1.2064728506,
        "absolute_error_percentage_points": 0.0018232798,
        "relative_error_percent": 0.151354,
        "total_shots": 16384,
        "grover_applications": 28672,
    },
    {
        "countries": 3,
        "valid_outcomes": 1000,
        "qubits": 10,
        "exact_probability_percent": 0.7263678323,
        "estimated_probability_percent": 0.7240406907,
        "absolute_error_percentage_points": 0.0023271416,
        "relative_error_percent": 0.320381,
        "total_shots": 16384,
        "grover_applications": 28672,
    },
]

df = pd.DataFrame(results)

# ==================================================
# 3. CALCULATE SCALING FACTORS
# ==================================================

df["outcome_growth_vs_previous"] = (
    df["valid_outcomes"] /
    df["valid_outcomes"].shift(1)
)

df["qubit_growth_vs_previous"] = (
    df["qubits"] /
    df["qubits"].shift(1)
)

# ==================================================
# 4. DISPLAY COMPARISON
# ==================================================

print("=" * 75)
print("TWO-COUNTRY VS THREE-COUNTRY QMLAE COMPARISON")
print("=" * 75)

print(df.to_string(index=False))

# ==================================================
# 5. SAVE RESULTS
# ==================================================

output_file = OUTPUT_DIR / "country_scaling_comparison.csv"

df.to_csv(output_file, index=False)

print("\nComparison saved to:")
print(output_file)

print("\nCOMPARISON COMPLETE.")
