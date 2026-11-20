
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FOLDER = PROJECT_ROOT / "src" / "model"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

sys.path.insert(0, str(MODEL_FOLDER))

from term_distribution import (
    calculate_term_distribution,
    calculate_metrics,
)

# --------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------

NUMBER_OF_YEARS = 3
THRESHOLD = 80_000_000

# --------------------------------------------------
# EXACT REFERENCE DISTRIBUTION
# --------------------------------------------------

def prepare_exact_distribution():

    term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    states = np.array(
        list(term.keys()),
        dtype=int
    )

    probabilities = np.array(
        list(term.values()),
        dtype=float
    )

    probabilities /= probabilities.sum()

    return states, probabilities


# --------------------------------------------------
# CALCULATE ALL FIVE RISK METRICS
# --------------------------------------------------

def calculate_metrics_from_states(term_states):

    a = term_states[:, 0]
    b = term_states[:, 1]
    r1 = term_states[:, 2]
    r2 = term_states[:, 3]

    loss_1 = (
        70_000_000 * a / 4
        + 10_000_000 * r1
    )

    loss_2 = (
        70_000_000 * b / 4
        + 10_000_000 * r2
    )

    total_loss = loss_1 + loss_2

    return {
        "Expected principal loss": np.mean(total_loss),

        "Both countries pay": np.mean(
            (loss_1 > 0) & (loss_2 > 0)
        ),

        "Portfolio loss exceeds 80m": np.mean(
            total_loss > THRESHOLD
        ),

        "Expected stop-loss": np.mean(
            np.maximum(total_loss - THRESHOLD, 0)
        ),

        "Both bonds exhausted": np.mean(
            (loss_1 == 80_000_000)
            & (loss_2 == 80_000_000)
        ),
    }


# --------------------------------------------------
# CALCULATE EXACT VALUES
# --------------------------------------------------

def calculate_exact_reference():

    states, probabilities = prepare_exact_distribution()

    # Existing exact ADB calculation
    exact_term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    exact_values = calculate_metrics(exact_term)

    # Independent check using the shared metric function
    weighted_values = {}

    a = states[:, 0]
    b = states[:, 1]
    r1 = states[:, 2]
    r2 = states[:, 3]

    loss_1 = 70_000_000 * a / 4 + 10_000_000 * r1
    loss_2 = 70_000_000 * b / 4 + 10_000_000 * r2
    total_loss = loss_1 + loss_2

    weighted_values["Expected principal loss"] = np.sum(
        probabilities * total_loss
    )

    weighted_values["Both countries pay"] = np.sum(
        probabilities * ((loss_1 > 0) & (loss_2 > 0))
    )

    weighted_values["Portfolio loss exceeds 80m"] = np.sum(
        probabilities * (total_loss > THRESHOLD)
    )

    weighted_values["Expected stop-loss"] = np.sum(
        probabilities * np.maximum(total_loss - THRESHOLD, 0)
    )

    weighted_values["Both bonds exhausted"] = np.sum(
        probabilities
        * (
            (loss_1 == 80_000_000)
            & (loss_2 == 80_000_000)
        )
    )

    return exact_values, weighted_values


# --------------------------------------------------
# DISPLAY AND VALIDATE
# --------------------------------------------------

def main():

    print("\nADB FAIR COMPARISON — FOUNDATION")
    print("================================")

    states, probabilities = prepare_exact_distribution()

    print(f"\nNumber of term outcomes: {len(states):,}")
    print(f"Probability sum: {probabilities.sum():.12f}")

    exact_values, independent_values = calculate_exact_reference()

    print("\nEXACT ADB REFERENCE VALUES")
    print("--------------------------")

    rows = []

    for metric, exact_value in exact_values.items():

        independent_value = independent_values[metric]

        difference = abs(exact_value - independent_value)

        print(f"\n{metric}")
        print(f"  Exact reference:       {exact_value:,.10f}")
        print(f"  Independent check:     {independent_value:,.10f}")
        print(f"  Difference:            {difference:.3e}")

        rows.append({
            "Metric": metric,
            "Exact reference": exact_value,
            "Independent check": independent_value,
            "Absolute difference": difference,
        })

    dataframe = pd.DataFrame(rows)

    output_path = OUTPUT_FOLDER / "fair_comparison_exact_reference.csv"

    dataframe.to_csv(output_path, index=False)

    print("\nSaved:", output_path)

    # Check that the two calculations agree
    for row in rows:
        assert row["Absolute difference"] < 1e-6, (
            f"Reference mismatch for {row['Metric']}"
        )

    print("\nVALIDATION PASSED")
    print("The five exact reference metrics agree.")


if __name__ == "__main__":
    main()
