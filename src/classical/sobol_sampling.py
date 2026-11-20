
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import qmc, t

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

# ADB methodology: 8 independent scrambles
NUMBER_OF_SCRAMBLES = 8

# Each sample size is a power of 2
SAMPLE_EXPONENTS = [10, 12, 14, 16]

RANDOM_SEED = 42
THRESHOLD = 80_000_000

CONFIDENCE_LEVEL = 0.95


# --------------------------------------------------
# PREPARE ANNUAL DISTRIBUTION
# --------------------------------------------------

def prepare_distribution():

    annual = annual_probability_distribution()

    states = np.array(
        list(annual.keys()),
        dtype=int
    )

    probabilities = np.array(
        list(annual.values())
    )

    cumulative_probabilities = np.cumsum(probabilities)

    cumulative_probabilities[-1] = 1.0

    return states, cumulative_probabilities


# --------------------------------------------------
# CONVERT SOBOL POINTS INTO THREE-YEAR TERMS
# --------------------------------------------------

def generate_term_states(sobol_points, states, cumulative):

    number_of_paths = len(sobol_points)

    term_states = np.zeros(
        (number_of_paths, 4),
        dtype=int
    )

    # Each Sobol dimension represents one year

    for year in range(NUMBER_OF_YEARS):

        indices = np.searchsorted(
            cumulative,
            sobol_points[:, year],
            side="right"
        )

        annual_draws = states[indices]

        term_states += annual_draws

    # Apply ADB three-year accumulation caps

    term_states[:, 0] = np.minimum(term_states[:, 0], 4)
    term_states[:, 1] = np.minimum(term_states[:, 1], 4)
    term_states[:, 2] = np.minimum(term_states[:, 2], 1)
    term_states[:, 3] = np.minimum(term_states[:, 3], 1)

    return term_states


# --------------------------------------------------
# CALCULATE FIVE RISK METRICS
# --------------------------------------------------

def calculate_sample_metrics(term_states):

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
        )
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
# RUN EIGHT INDEPENDENT SCRAMBLES
# --------------------------------------------------

def run_sobol_experiment(states, cumulative, exponent):

    number_of_points = 2 ** exponent

    print(f"\nSOBOL SAMPLE SIZE: {number_of_points:,}")
    print("-----------------------------------")

    replicate_results = []

    for scramble in range(NUMBER_OF_SCRAMBLES):

        seed = RANDOM_SEED + scramble

        # Three-dimensional scrambled Sobol sequence

        sampler = qmc.Sobol(
            d=NUMBER_OF_YEARS,
            scramble=True,
            seed=seed
        )

        sobol_points = sampler.random_base2(
            m=exponent
        )

        # Convert points into ADB annual states

        term_states = generate_term_states(
            sobol_points,
            states,
            cumulative
        )

        metrics = calculate_sample_metrics(
            term_states
        )

        replicate_results.append(metrics)

        print(
            f"Scramble {scramble + 1}/{NUMBER_OF_SCRAMBLES} completed"
        )

    return replicate_results


# --------------------------------------------------
# CALCULATE MEAN AND T CONFIDENCE INTERVAL
# --------------------------------------------------

def summarize_scrambles(replicate_results, exact_values, exponent):

    rows = []

    for metric in exact_values:

        estimates = np.array([
            result[metric]
            for result in replicate_results
        ])

        mean_estimate = np.mean(estimates)

        standard_deviation = np.std(
            estimates,
            ddof=1
        )

        standard_error = (
            standard_deviation
            / np.sqrt(NUMBER_OF_SCRAMBLES)
        )

        # Student-t critical value, df = 8 - 1 = 7

        critical_value = t.ppf(
            (1 + CONFIDENCE_LEVEL) / 2,
            df=NUMBER_OF_SCRAMBLES - 1
        )

        margin_of_error = critical_value * standard_error

        lower_bound = mean_estimate - margin_of_error
        upper_bound = mean_estimate + margin_of_error

        exact = exact_values[metric]

        relative_error = (
            abs(mean_estimate - exact)
            / abs(exact)
            * 100
            if exact != 0
            else 0
        )

        rows.append({
            "Metric": metric,
            "Exponent": exponent,
            "Points per scramble": 2 ** exponent,
            "Total points across scrambles": (
                NUMBER_OF_SCRAMBLES * 2 ** exponent
            ),
            "Sobol estimate": mean_estimate,
            "Exact value": exact,
            "Standard error": standard_error,
            "95% CI lower": lower_bound,
            "95% CI upper": upper_bound,
            "Relative error (%)": relative_error
        })

        if metric in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:

            print(f"\n{metric}")
            print(f"  Sobol estimate: ${mean_estimate:,.2f}")
            print(f"  Exact value:    ${exact:,.2f}")
            print(f"  Standard error: ${standard_error:,.2f}")

        else:

            print(f"\n{metric}")
            print(f"  Sobol estimate: {mean_estimate * 100:.6f}%")
            print(f"  Exact value:    {exact * 100:.6f}%")
            print(
                f"  Standard error: "
                f"{standard_error * 100:.6f} percentage points"
            )

        print(
            f"  95% CI: [{lower_bound:.8f}, {upper_bound:.8f}]"
        )

        print(f"  Relative error: {relative_error:.3f}%")

    return rows


# --------------------------------------------------
# PLOT ALL FIVE METRICS IN ONE FIGURE
# --------------------------------------------------

def plot_sobol_convergence(dataframe):

    plot_settings = {
        "Expected principal loss": {
            "title": "Expected Principal Loss",
            "scale": 1_000_000,
            "ylabel": "USD millions"
        },

        "Both countries pay": {
            "title": "Both Countries Pay",
            "scale": 100,
            "ylabel": "Probability (%)"
        },

        "Portfolio loss exceeds 80m": {
            "title": "Loss Exceeds $80m",
            "scale": 100,
            "ylabel": "Probability (%)"
        },

        "Expected stop-loss": {
            "title": "Expected Stop-Loss",
            "scale": 1_000_000,
            "ylabel": "USD millions"
        },

        "Both bonds exhausted": {
            "title": "Both Bonds Exhausted",
            "scale": 100,
            "ylabel": "Probability (%)"
        }
    }

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 9)
    )

    axes = axes.flatten()

    for i, (metric, settings) in enumerate(plot_settings.items()):

        ax = axes[i]

        subset = dataframe[
            dataframe["Metric"] == metric
        ].sort_values("Points per scramble")

        x = subset["Points per scramble"]

        scale = settings["scale"]

        estimate = subset["Sobol estimate"] * scale
        lower = subset["95% CI lower"] * scale
        upper = subset["95% CI upper"] * scale
        exact = subset["Exact value"].iloc[0] * scale

        ax.plot(
            x,
            estimate,
            marker="o",
            linewidth=2,
            label="Mean of 8 scrambles"
        )

        ax.fill_between(
            x,
            lower,
            upper,
            alpha=0.2,
            label="95% t confidence interval"
        )

        ax.axhline(
            exact,
            linestyle="--",
            linewidth=2,
            label="Exact result"
        )

        ax.set_xscale("log")

        ax.set_title(settings["title"])
        ax.set_xlabel("Points per scramble")
        ax.set_ylabel(settings["ylabel"])

        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.delaxes(axes[5])

    fig.suptitle(
        "ADB Scrambled Sobol Convergence",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_FOLDER / "sobol_all_metrics.png"
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
# MAIN PROGRAM
# --------------------------------------------------

def main():

    start_time = time.time()

    print("\nADB SCRAMBLED SOBOL EXPERIMENT")
    print("==============================")

    print("\nPreparing annual probability distribution...")

    states, cumulative = prepare_distribution()

    print("\nCalculating exact reference values...")

    exact_values = get_exact_values()

    all_rows = []

    # Run all sample sizes

    for exponent in SAMPLE_EXPONENTS:

        replicate_results = run_sobol_experiment(
            states,
            cumulative,
            exponent
        )

        rows = summarize_scrambles(
            replicate_results,
            exact_values,
            exponent
        )

        all_rows.extend(rows)

    # Save results

    dataframe = pd.DataFrame(all_rows)

    output_path = (
        OUTPUT_FOLDER / "sobol_results.csv"
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print("\nSaved numerical results:", output_path)

    # Plot convergence

    print("\nGenerating combined convergence graph...")

    plot_sobol_convergence(dataframe)

    print("\nEXPERIMENT COMPLETED")
    print("--------------------")
    print(f"Runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
