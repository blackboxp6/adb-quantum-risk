
import sys
import time
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

from probability_model import annual_probability_distribution
from term_distribution import calculate_term_distribution, calculate_metrics


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

NUMBER_OF_YEARS = 3
SAMPLES_PER_STRATUM = 100_000
RANDOM_SEED = 42

THRESHOLD = 80_000_000


# --------------------------------------------------
# DRAW ANNUAL STATES
# --------------------------------------------------

def draw_annual_states(states, probabilities, number, rng):

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0

    random_numbers = rng.random(number)

    indices = np.searchsorted(
        cumulative,
        random_numbers,
        side="right"
    )

    return states[indices]


# --------------------------------------------------
# SIMULATE ONE STRATUM
# --------------------------------------------------

def simulate_stratum(annual, first_payout_year, number, rng):

    states = np.array(list(annual.keys()), dtype=int)
    probabilities = np.array(list(annual.values()))

    # Identify the no-payout annual state

    no_payout = np.array([0, 0, 0, 0])

    no_payout_index = np.where(
        np.all(states == no_payout, axis=1)
    )[0][0]

    p0 = probabilities[no_payout_index]

    # Conditional distribution:
    # Annual states excluding the no-payout state

    nonzero_mask = np.any(states != 0, axis=1)

    conditional_states = states[nonzero_mask]
    conditional_probabilities = probabilities[nonzero_mask]

    conditional_probabilities = (
        conditional_probabilities
        / conditional_probabilities.sum()
    )

    term_states = np.zeros((number, 4), dtype=int)

    # Generate the three annual states

    for year in range(NUMBER_OF_YEARS):

        if year < first_payout_year:

            # Before first payout: force no payout

            annual_draws = np.zeros((number, 4), dtype=int)

        elif year == first_payout_year:

            # First payout year: force a nonzero state

            annual_draws = draw_annual_states(
                conditional_states,
                conditional_probabilities,
                number,
                rng
            )

        else:

            # Later years: use the ordinary distribution

            annual_draws = draw_annual_states(
                states,
                probabilities,
                number,
                rng
            )

        term_states += annual_draws

    # Apply three-year caps

    term_states[:, 0] = np.minimum(term_states[:, 0], 4)
    term_states[:, 1] = np.minimum(term_states[:, 1], 4)
    term_states[:, 2] = np.minimum(term_states[:, 2], 1)
    term_states[:, 3] = np.minimum(term_states[:, 3], 1)

    return term_states


# --------------------------------------------------
# CALCULATE RISK METRICS
# --------------------------------------------------

def calculate_sample_metrics(term_states):

    a = term_states[:, 0]
    b = term_states[:, 1]
    r1 = term_states[:, 2]
    r2 = term_states[:, 3]

    loss_1 = 70_000_000 * a / 4 + 10_000_000 * r1
    loss_2 = 70_000_000 * b / 4 + 10_000_000 * r2

    total_loss = loss_1 + loss_2

    return {
        "Expected principal loss": total_loss,

        "Both countries pay": (
            (loss_1 > 0) & (loss_2 > 0)
        ).astype(float),

        "Portfolio loss exceeds 80m": (
            total_loss > THRESHOLD
        ).astype(float),

        "Expected stop-loss": np.maximum(
            total_loss - THRESHOLD, 0
        ),

        "Both bonds exhausted": (
            (loss_1 == 80_000_000)
            & (loss_2 == 80_000_000)
        ).astype(float)
    }


# --------------------------------------------------
# EXACT REFERENCE
# --------------------------------------------------

def get_exact_values():

    term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    return calculate_metrics(term)


# --------------------------------------------------
# MAIN STRATIFIED MONTE CARLO
# --------------------------------------------------

def main():

    start_time = time.time()

    print("\nADB STRATIFIED MONTE CARLO")
    print("==========================")

    annual = annual_probability_distribution()

    p0 = annual[(0, 0, 0, 0)]

    print(f"\nAnnual no-payout probability: {p0:.6%}")
    print(f"Samples per stratum: {SAMPLES_PER_STRATUM:,}")

    exact = get_exact_values()

    # Store weighted estimates and variances

    weighted_estimates = {}
    weighted_variances = {}

    rng = np.random.default_rng(RANDOM_SEED)

    # --------------------------------------------------
    # SIMULATE EACH FIRST-PAYOUT-YEAR STRATUM
    # --------------------------------------------------

    for first_payout_year in range(NUMBER_OF_YEARS):

        year_number = first_payout_year + 1

        # Probability of this stratum

        weight = (p0 ** first_payout_year) * (1 - p0)

        print(f"\nSTRATUM {year_number}")
        print("-------------------")

        print(f"First payout year: {year_number}")
        print(f"Stratum probability: {weight:.8%}")

        term_states = simulate_stratum(
            annual,
            first_payout_year,
            SAMPLES_PER_STRATUM,
            rng
        )

        sample_metrics = calculate_sample_metrics(term_states)

        for metric, observations in sample_metrics.items():

            estimate = np.mean(observations)

            sample_variance = np.var(
                observations,
                ddof=1
            )

            if metric not in weighted_estimates:

                weighted_estimates[metric] = 0.0
                weighted_variances[metric] = 0.0

            weighted_estimates[metric] += weight * estimate

            weighted_variances[metric] += (
                weight ** 2
                * sample_variance
                / SAMPLES_PER_STRATUM
            )

    # --------------------------------------------------
    # COMBINE THE STRATA
    # --------------------------------------------------

    print("\nFINAL STRATIFIED RESULTS")
    print("========================")

    rows = []

    for metric in weighted_estimates:

        estimate = weighted_estimates[metric]

        standard_error = np.sqrt(
            weighted_variances[metric]
        )

        exact_value = exact[metric]

        relative_error = (
            abs(estimate - exact_value)
            / abs(exact_value)
            * 100
            if exact_value != 0
            else 0
        )

        if metric in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:

            print(f"\n{metric}")
            print(f"  Estimate:       ${estimate:,.2f}")
            print(f"  Exact:          ${exact_value:,.2f}")
            print(f"  Standard error: ${standard_error:,.2f}")

        else:

            print(f"\n{metric}")
            print(f"  Estimate:       {estimate * 100:.6f}%")
            print(f"  Exact:          {exact_value * 100:.6f}%")
            print(f"  Standard error: {standard_error * 100:.6f} percentage points")

        print(f"  Relative error: {relative_error:.3f}%")

        rows.append({
            "Metric": metric,
            "Stratified Estimate": estimate,
            "Exact Value": exact_value,
            "Standard Error": standard_error,
            "Relative Error (%)": relative_error
        })

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    dataframe = pd.DataFrame(rows)

    output_path = OUTPUT_FOLDER / "stratified_results.csv"

    dataframe.to_csv(
        output_path,
        index=False
    )

    print("\nSaved results:", output_path)

    print("\nEXPERIMENT COMPLETED")
    print("--------------------")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
