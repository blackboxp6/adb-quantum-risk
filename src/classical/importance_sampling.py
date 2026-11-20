
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
# EXPERIMENT SETTINGS
# --------------------------------------------------

NUMBER_OF_YEARS = 3

# Match the total path budget of our stratified experiment
NUMBER_OF_PATHS = 300_000

RANDOM_SEED = 42
THRESHOLD = 80_000_000


# --------------------------------------------------
# PREPARE TRUE AND TILTED DISTRIBUTIONS
# --------------------------------------------------

def prepare_distributions():

    annual = annual_probability_distribution()

    states = np.array(
        list(annual.keys()),
        dtype=int
    )

    true_probabilities = np.array(
        list(annual.values()),
        dtype=float
    )

    # Calculate annual losses for both countries

    a = states[:, 0]
    b = states[:, 1]
    r1 = states[:, 2]
    r2 = states[:, 3]

    loss_1 = 70_000_000 * a / 4 + 10_000_000 * r1
    loss_2 = 70_000_000 * b / 4 + 10_000_000 * r2

    annual_principal_loss = loss_1 + loss_2

    # Tilt the true distribution toward high-loss states:
    #
    # p_tilt(s) = p(s) * L(s) / E_p[L]

    tilted_probabilities = (
        true_probabilities * annual_principal_loss
    )

    tilted_probabilities /= tilted_probabilities.sum()

    # ADB proposal:
    #
    # q(s) = 0.5 * p(s) + 0.5 * p_tilt(s)

    proposal_probabilities = (
        0.5 * true_probabilities
        + 0.5 * tilted_probabilities
    )

    # Likelihood ratio for each annual state

    likelihood_ratios = (
        true_probabilities / proposal_probabilities
    )

    print("\nDISTRIBUTION CHECKS")
    print("-------------------")

    print("True probability sum:",
          true_probabilities.sum())

    print("Tilted probability sum:",
          tilted_probabilities.sum())

    print("Proposal probability sum:",
          proposal_probabilities.sum())

    print("Minimum likelihood ratio:",
          likelihood_ratios.min())

    print("Maximum likelihood ratio:",
          likelihood_ratios.max())

    return (
        states,
        true_probabilities,
        proposal_probabilities,
        likelihood_ratios
    )


# --------------------------------------------------
# SIMULATE IMPORTANCE-SAMPLED TERMS
# --------------------------------------------------

def generate_importance_samples(
    states,
    proposal_probabilities,
    likelihood_ratios
):

    rng = np.random.default_rng(RANDOM_SEED)

    cumulative_proposal = np.cumsum(
        proposal_probabilities
    )

    cumulative_proposal[-1] = 1.0

    term_states = np.zeros(
        (NUMBER_OF_PATHS, 4),
        dtype=int
    )

    # One importance weight for each three-year path

    path_weights = np.ones(NUMBER_OF_PATHS)

    print("\nGenerating importance-sampled paths...")

    for year in range(NUMBER_OF_YEARS):

        random_numbers = rng.random(NUMBER_OF_PATHS)

        indices = np.searchsorted(
            cumulative_proposal,
            random_numbers,
            side="right"
        )

        annual_draws = states[indices]

        # Add annual states to the three-year term

        term_states += annual_draws

        # Multiply by the annual likelihood ratio

        path_weights *= likelihood_ratios[indices]

        print(f"Year {year + 1} completed")

    # Apply ADB three-year accumulation caps

    term_states[:, 0] = np.minimum(term_states[:, 0], 4)
    term_states[:, 1] = np.minimum(term_states[:, 1], 4)
    term_states[:, 2] = np.minimum(term_states[:, 2], 1)
    term_states[:, 3] = np.minimum(term_states[:, 3], 1)

    return term_states, path_weights


# --------------------------------------------------
# CALCULATE METRIC VALUES FOR EACH PATH
# --------------------------------------------------

def calculate_path_metrics(term_states):

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
        "Expected principal loss": total_loss,

        "Both countries pay": (
            (loss_1 > 0) & (loss_2 > 0)
        ).astype(float),

        "Portfolio loss exceeds 80m": (
            total_loss > THRESHOLD
        ).astype(float),

        "Expected stop-loss": np.maximum(
            total_loss - THRESHOLD,
            0
        ),

        "Both bonds exhausted": (
            (loss_1 == 80_000_000)
            & (loss_2 == 80_000_000)
        ).astype(float)
    }


# --------------------------------------------------
# EXACT REFERENCE VALUES
# --------------------------------------------------

def get_exact_values():

    term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    return calculate_metrics(term)


# --------------------------------------------------
# IMPORTANCE-SAMPLING ESTIMATES
# --------------------------------------------------

def calculate_importance_estimates(
    path_metrics,
    path_weights,
    exact_values
):

    rows = []

    print("\nIMPORTANCE SAMPLING RESULTS")
    print("===========================")

    for metric, observations in path_metrics.items():

        # Likelihood-ratio weighted observations

        weighted_observations = (
            path_weights * observations
        )

        # Unbiased importance-sampling estimator

        estimate = np.mean(weighted_observations)

        # Estimated standard error

        standard_error = (
            np.std(weighted_observations, ddof=1)
            / np.sqrt(NUMBER_OF_PATHS)
        )

        exact = exact_values[metric]

        relative_error = (
            abs(estimate - exact)
            / abs(exact)
            * 100
            if exact != 0
            else 0
        )

        if metric in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:

            print(f"\n{metric}")
            print(f"  Estimate:       ${estimate:,.2f}")
            print(f"  Exact:          ${exact:,.2f}")
            print(f"  Standard error: ${standard_error:,.2f}")

        else:

            print(f"\n{metric}")
            print(f"  Estimate:       {estimate * 100:.6f}%")
            print(f"  Exact:          {exact * 100:.6f}%")

            print(
                f"  Standard error: "
                f"{standard_error * 100:.6f} percentage points"
            )

        print(f"  Relative error: {relative_error:.3f}%")

        rows.append({
            "Metric": metric,
            "Importance Sampling Estimate": estimate,
            "Exact Value": exact,
            "Standard Error": standard_error,
            "Relative Error (%)": relative_error
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# WEIGHT DIAGNOSTICS
# --------------------------------------------------

def calculate_weight_diagnostics(path_weights):

    sum_weights = np.sum(path_weights)
    sum_squared_weights = np.sum(path_weights ** 2)

    effective_sample_size = (
        sum_weights ** 2 / sum_squared_weights
    )

    print("\nWEIGHT DIAGNOSTICS")
    print("------------------")

    print(f"Mean path weight: {np.mean(path_weights):.6f}")
    print(f"Minimum path weight: {np.min(path_weights):.6f}")
    print(f"Maximum path weight: {np.max(path_weights):.6f}")

    print(
        f"Effective sample size: "
        f"{effective_sample_size:,.0f}"
    )

    print(
        f"Effective sample size ratio: "
        f"{effective_sample_size / len(path_weights):.2%}"
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    start_time = time.time()

    print("\nADB IMPORTANCE SAMPLING EXPERIMENT")
    print("==================================")

    # 1. Exact reference

    print("\nCalculating exact reference values...")

    exact_values = get_exact_values()

    # 2. Prepare proposal distribution

    (
        states,
        true_probabilities,
        proposal_probabilities,
        likelihood_ratios
    ) = prepare_distributions()

    # 3. Generate weighted paths

    term_states, path_weights = generate_importance_samples(
        states,
        proposal_probabilities,
        likelihood_ratios
    )

    # 4. Calculate metrics

    path_metrics = calculate_path_metrics(term_states)

    # 5. Calculate estimates

    results = calculate_importance_estimates(
        path_metrics,
        path_weights,
        exact_values
    )

    # 6. Weight diagnostics

    calculate_weight_diagnostics(path_weights)

    # 7. Save results

    output_path = (
        OUTPUT_FOLDER / "importance_sampling_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nSaved results:", output_path)

    print("\nEXPERIMENT COMPLETED")
    print("--------------------")

    print(f"Number of paths: {NUMBER_OF_PATHS:,}")

    print(f"Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
