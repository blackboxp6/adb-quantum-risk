
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_FOLDER = PROJECT_ROOT / "src" / "model"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

sys.path.insert(0, str(MODEL_FOLDER))

from probability_model import annual_probability_distribution
from term_distribution import (
    calculate_term_distribution,
    calculate_metrics,
)

# --------------------------------------------------
# EXPERIMENT SETTINGS
# --------------------------------------------------

NUMBER_OF_YEARS = 3

BUDGETS = [1024, 4096, 8192, 16384]

REPETITIONS = 100

RANDOM_SEED = 42

THRESHOLD = 80_000_000

METRICS = [
    "Expected principal loss",
    "Both countries pay",
    "Portfolio loss exceeds 80m",
    "Expected stop-loss",
    "Both bonds exhausted",
]

METHODS = [
    "Monte Carlo",
    "Stratified Monte Carlo",
    "Sobol",
    "Importance Sampling",
]

# --------------------------------------------------
# PREPARE ANNUAL DISTRIBUTION
# --------------------------------------------------

def prepare_annual_distribution():

    annual = annual_probability_distribution()

    states = np.array(
        list(annual.keys()),
        dtype=int
    )

    probabilities = np.array(
        list(annual.values()),
        dtype=float
    )

    probabilities /= probabilities.sum()

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0

    return states, probabilities, cumulative


# --------------------------------------------------
# PREPARE EXACT REFERENCE
# --------------------------------------------------

def prepare_exact_values():

    exact_term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    return calculate_metrics(exact_term)


# --------------------------------------------------
# CONVERT ANNUAL STATES INTO THREE-YEAR TERMS
# --------------------------------------------------

def accumulate_three_years(annual_draws):

    term_states = np.sum(
        annual_draws,
        axis=1
    )

    term_states[:, 0] = np.minimum(term_states[:, 0], 4)
    term_states[:, 1] = np.minimum(term_states[:, 1], 4)
    term_states[:, 2] = np.minimum(term_states[:, 2], 1)
    term_states[:, 3] = np.minimum(term_states[:, 3], 1)

    return term_states


# --------------------------------------------------
# CALCULATE METRIC OBSERVATIONS FOR EACH PATH
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
        ).astype(float),
    }


# --------------------------------------------------
# DRAW FROM A DISCRETE DISTRIBUTION
# --------------------------------------------------

def draw_states(states, cumulative, number, rng):

    random_numbers = rng.random(number)

    indices = np.searchsorted(
        cumulative,
        random_numbers,
        side="right"
    )

    return states[indices]


# --------------------------------------------------
# METHOD 1: ORDINARY MONTE CARLO
# --------------------------------------------------

def run_monte_carlo(
    states,
    cumulative,
    budget,
    rng
):

    annual_draws = np.zeros(
        (budget, NUMBER_OF_YEARS, 4),
        dtype=int
    )

    for year in range(NUMBER_OF_YEARS):

        annual_draws[:, year, :] = draw_states(
            states,
            cumulative,
            budget,
            rng
        )

    term_states = accumulate_three_years(
        annual_draws
    )

    observations = calculate_path_metrics(
        term_states
    )

    return {
        metric: np.mean(values)
        for metric, values in observations.items()
    }


# --------------------------------------------------
# METHOD 2: STRATIFIED MONTE CARLO
# --------------------------------------------------

def run_stratified_monte_carlo(
    states,
    probabilities,
    budget,
    rng
):

    no_payout = np.all(states == 0, axis=1)

    p0 = probabilities[no_payout][0]

    nonzero_mask = ~no_payout

    conditional_states = states[nonzero_mask]

    conditional_probabilities = probabilities[
        nonzero_mask
    ]

    conditional_cumulative = np.cumsum(
        conditional_probabilities
        / conditional_probabilities.sum()
    )

    conditional_cumulative[-1] = 1.0

    ordinary_cumulative = np.cumsum(probabilities)
    ordinary_cumulative[-1] = 1.0

    # Divide total budget approximately equally
    # across the three first-payout-year strata.

    allocations = [
        budget // NUMBER_OF_YEARS
        for _ in range(NUMBER_OF_YEARS)
    ]

    for i in range(budget % NUMBER_OF_YEARS):
        allocations[i] += 1

    estimates = {
        metric: 0.0
        for metric in METRICS
    }

    for first_payout_year, number in enumerate(allocations):

        if number == 0:
            continue

        # Probability of this first-payout-year stratum
        weight = (
            p0 ** first_payout_year
            * (1 - p0)
        )

        annual_draws = np.zeros(
            (number, NUMBER_OF_YEARS, 4),
            dtype=int
        )

        for year in range(NUMBER_OF_YEARS):

            if year < first_payout_year:

                # Before first payout:
                # force zero loss.

                continue

            elif year == first_payout_year:

                # First payout year:
                # draw from nonzero annual states.

                annual_draws[:, year, :] = draw_states(
                    conditional_states,
                    conditional_cumulative,
                    number,
                    rng
                )

            else:

                # Later years:
                # draw from the ordinary distribution.

                annual_draws[:, year, :] = draw_states(
                    states,
                    ordinary_cumulative,
                    number,
                    rng
                )

        term_states = accumulate_three_years(
            annual_draws
        )

        observations = calculate_path_metrics(
            term_states
        )

        for metric in METRICS:

            estimates[metric] += (
                weight * np.mean(observations[metric])
            )

    # The all-three-years-no-payout stratum
    # contributes zero to all five metrics.

    return estimates


# --------------------------------------------------
# METHOD 3: SCRAMBLED SOBOL
# --------------------------------------------------

def run_sobol(
    states,
    probabilities,
    budget,
    seed
):

    if budget & (budget - 1):
        raise ValueError(
            "Sobol budget must be a power of two."
        )

    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0

    sampler = qmc.Sobol(
        d=NUMBER_OF_YEARS,
        scramble=True,
        seed=seed
    )

    points = sampler.random_base2(
        m=int(np.log2(budget))
    )

    annual_draws = np.zeros(
        (budget, NUMBER_OF_YEARS, 4),
        dtype=int
    )

    for year in range(NUMBER_OF_YEARS):

        indices = np.searchsorted(
            cumulative,
            points[:, year],
            side="right"
        )

        annual_draws[:, year, :] = states[indices]

    term_states = accumulate_three_years(
        annual_draws
    )

    observations = calculate_path_metrics(
        term_states
    )

    return {
        metric: np.mean(values)
        for metric, values in observations.items()
    }


# --------------------------------------------------
# METHOD 4: IMPORTANCE SAMPLING
# --------------------------------------------------

def prepare_importance_distribution(
    states,
    probabilities
):

    a = states[:, 0]
    b = states[:, 1]
    r1 = states[:, 2]
    r2 = states[:, 3]

    loss_1 = (
        70_000_000 * a / 4
        + 10_000_000 * r1
    )

    loss_2 = (
        70_000_000 * b / 4
        + 10_000_000 * r2
    )

    annual_loss = loss_1 + loss_2

    tilted = probabilities * annual_loss

    tilted /= tilted.sum()

    proposal = (
        0.5 * probabilities
        + 0.5 * tilted
    )

    likelihood_ratios = probabilities / proposal

    proposal_cumulative = np.cumsum(proposal)
    proposal_cumulative[-1] = 1.0

    return proposal_cumulative, likelihood_ratios


def run_importance_sampling(
    states,
    proposal_cumulative,
    likelihood_ratios,
    budget,
    rng
):

    annual_draws = np.zeros(
        (budget, NUMBER_OF_YEARS, 4),
        dtype=int
    )

    path_weights = np.ones(budget)

    for year in range(NUMBER_OF_YEARS):

        random_numbers = rng.random(budget)

        indices = np.searchsorted(
            proposal_cumulative,
            random_numbers,
            side="right"
        )

        annual_draws[:, year, :] = states[indices]

        path_weights *= likelihood_ratios[indices]

    term_states = accumulate_three_years(
        annual_draws
    )

    observations = calculate_path_metrics(
        term_states
    )

    return {
        metric: np.mean(
            path_weights * values
        )
        for metric, values in observations.items()
    }


# --------------------------------------------------
# RUN ALL CLASSICAL METHODS
# --------------------------------------------------

def run_experiment():

    print("\nADB CLASSICAL BENCHMARK")
    print("======================")

    states, probabilities, cumulative = (
        prepare_annual_distribution()
    )

    exact_values = prepare_exact_values()

    (
        proposal_cumulative,
        likelihood_ratios
    ) = prepare_importance_distribution(
        states,
        probabilities
    )

    print(f"\nAnnual states: {len(states)}")
    print(f"Repetitions per budget: {REPETITIONS}")
    print(f"Budgets: {BUDGETS}")

    rows = []

    for budget in BUDGETS:

        print(f"\nBUDGET: {budget:,}")
        print("--------------------")

        for repetition in range(REPETITIONS):

            seed = RANDOM_SEED + repetition + budget * 100

            # Independent random streams per method
            mc_rng = np.random.default_rng(seed + 1)
            strat_rng = np.random.default_rng(seed + 2)
            is_rng = np.random.default_rng(seed + 3)

            # --------------------------------------
            # Ordinary Monte Carlo
            # --------------------------------------

            start = time.perf_counter()

            estimates = run_monte_carlo(
                states,
                cumulative,
                budget,
                mc_rng
            )

            runtime = time.perf_counter() - start

            for metric, estimate in estimates.items():

                rows.append({
                    "Method": "Monte Carlo",
                    "Budget": budget,
                    "Repetition": repetition + 1,
                    "Metric": metric,
                    "Estimate": estimate,
                    "Exact": exact_values[metric],
                    "Runtime (seconds)": runtime,
                })

            # --------------------------------------
            # Stratified Monte Carlo
            # --------------------------------------

            start = time.perf_counter()

            estimates = run_stratified_monte_carlo(
                states,
                probabilities,
                budget,
                strat_rng
            )

            runtime = time.perf_counter() - start

            for metric, estimate in estimates.items():

                rows.append({
                    "Method": "Stratified Monte Carlo",
                    "Budget": budget,
                    "Repetition": repetition + 1,
                    "Metric": metric,
                    "Estimate": estimate,
                    "Exact": exact_values[metric],
                    "Runtime (seconds)": runtime,
                })

            # --------------------------------------
            # Scrambled Sobol
            # --------------------------------------

            start = time.perf_counter()

            estimates = run_sobol(
                states,
                probabilities,
                budget,
                seed + 4
            )

            runtime = time.perf_counter() - start

            for metric, estimate in estimates.items():

                rows.append({
                    "Method": "Sobol",
                    "Budget": budget,
                    "Repetition": repetition + 1,
                    "Metric": metric,
                    "Estimate": estimate,
                    "Exact": exact_values[metric],
                    "Runtime (seconds)": runtime,
                })

            # --------------------------------------
            # Importance Sampling
            # --------------------------------------

            start = time.perf_counter()

            estimates = run_importance_sampling(
                states,
                proposal_cumulative,
                likelihood_ratios,
                budget,
                is_rng
            )

            runtime = time.perf_counter() - start

            for metric, estimate in estimates.items():

                rows.append({
                    "Method": "Importance Sampling",
                    "Budget": budget,
                    "Repetition": repetition + 1,
                    "Metric": metric,
                    "Estimate": estimate,
                    "Exact": exact_values[metric],
                    "Runtime (seconds)": runtime,
                })

        print(
            f"Completed {REPETITIONS} repetitions."
        )

    return pd.DataFrame(rows), exact_values


# --------------------------------------------------
# SUMMARIZE ERRORS
# --------------------------------------------------

def summarize_results(raw_results):

    rows = []

    grouped = raw_results.groupby(
        ["Method", "Budget", "Metric"]
    )

    for (method, budget, metric), group in grouped:

        errors = (
            group["Estimate"].to_numpy()
            - group["Exact"].to_numpy()
        )

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)
        standard_deviation = np.std(errors, ddof=1)

        rows.append({
            "Method": method,
            "Budget": budget,
            "Metric": metric,
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "Standard deviation": standard_deviation,
            "Mean runtime (seconds)": group[
                "Runtime (seconds)"
            ].mean(),
            "Repetitions": len(group),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# PLOT MAE FOR ALL FIVE METRICS
# --------------------------------------------------

def plot_mae(summary):

    plot_settings = {
        "Expected principal loss": 1_000_000,
        "Both countries pay": 100,
        "Portfolio loss exceeds 80m": 100,
        "Expected stop-loss": 1_000_000,
        "Both bonds exhausted": 100,
    }

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(17, 10)
    )

    axes = axes.flatten()

    for index, metric in enumerate(METRICS):

        ax = axes[index]

        subset = summary[
            summary["Metric"] == metric
        ]

        for method in METHODS:

            method_data = subset[
                subset["Method"] == method
            ].sort_values("Budget")

            ax.plot(
                method_data["Budget"],
                method_data["MAE"] * plot_settings[metric],
                marker="o",
                label=method
            )

        ax.set_title(metric)
        ax.set_xlabel("Total sample budget")

        if metric in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:
            ax.set_ylabel("MAE (USD millions)")
        else:
            ax.set_ylabel("MAE (percentage points)")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.delaxes(axes[5])

    fig.suptitle(
        "ADB Classical Methods: MAE vs Sample Budget",
        fontsize=16
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER
        / "fair_classical_mae_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("\nSaved graph:", output_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    start = time.time()

    raw_results, exact_values = run_experiment()

    raw_path = (
        OUTPUT_FOLDER
        / "fair_classical_raw_results.csv"
    )

    raw_results.to_csv(
        raw_path,
        index=False
    )

    summary = summarize_results(raw_results)

    summary_path = (
        OUTPUT_FOLDER
        / "fair_classical_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    plot_mae(summary)

    print("\nCLASSICAL BENCHMARK COMPLETED")
    print("=============================")

    print("\nSummary:")
    print(
        summary[
            [
                "Method",
                "Budget",
                "Metric",
                "MAE",
                "RMSE",
            ]
        ].to_string(index=False)
    )

    print("\nSaved raw results:", raw_path)
    print("Saved summary:", summary_path)

    print(
        f"\nTotal runtime: {time.time() - start:.2f} seconds"
    )


if __name__ == "__main__":
    main()
