
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# SIMULATION SETTINGS
# --------------------------------------------------

NUMBER_OF_YEARS = 3
MAXIMUM_SIMULATIONS = 1_000_000

SAMPLE_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000
]

RANDOM_SEED = 42
THRESHOLD = 80_000_000


# --------------------------------------------------
# MONTE CARLO SIMULATION
# --------------------------------------------------

def generate_simulations(number_of_paths, annual):

    rng = np.random.default_rng(RANDOM_SEED)

    states = list(annual.keys())
    probabilities = np.array(list(annual.values()))

    cumulative_probabilities = np.cumsum(probabilities)
    cumulative_probabilities[-1] = 1.0

    # Generate random numbers:
    # 3 years x 1 random number per year

    random_numbers = rng.random(
        (number_of_paths, NUMBER_OF_YEARS)
    )

    term_states = np.zeros(
        (number_of_paths, 4),
        dtype=int
    )

    for year in range(NUMBER_OF_YEARS):

        indices = np.searchsorted(
            cumulative_probabilities,
            random_numbers[:, year],
            side="right"
        )

        annual_draws = np.array(states)[indices]

        term_states += annual_draws

    # Apply the ADB term caps

    term_states[:, 0] = np.minimum(term_states[:, 0], 4)
    term_states[:, 1] = np.minimum(term_states[:, 1], 4)
    term_states[:, 2] = np.minimum(term_states[:, 2], 1)
    term_states[:, 3] = np.minimum(term_states[:, 3], 1)

    # Calculate country losses

    loss_1 = (
        70_000_000 * term_states[:, 0] / 4
        + 10_000_000 * term_states[:, 2]
    )

    loss_2 = (
        70_000_000 * term_states[:, 1] / 4
        + 10_000_000 * term_states[:, 3]
    )

    total_loss = loss_1 + loss_2

    # Calculate five risk metrics for every simulated term

    results = {
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

    return results


# --------------------------------------------------
# EXACT REFERENCE VALUES
# --------------------------------------------------

def get_exact_values():

    term = calculate_term_distribution(
        years=NUMBER_OF_YEARS
    )

    return calculate_metrics(term)


# --------------------------------------------------
# PLOT EACH RISK METRIC
# --------------------------------------------------


def plot_convergence(results, exact_values):

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

        estimates = np.array(results[metric])
        exact = exact_values[metric]

        scale = settings["scale"]

        ax.plot(
            SAMPLE_SIZES,
            estimates * scale,
            marker="o",
            linewidth=2,
            label="Monte Carlo"
        )

        ax.axhline(
            y=exact * scale,
            linestyle="--",
            linewidth=2,
            label="Exact result"
        )

        ax.set_xscale("log")

        ax.set_title(settings["title"])
        ax.set_xlabel("Number of simulations")
        ax.set_ylabel(settings["ylabel"])

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    # Remove the unused sixth subplot

    fig.delaxes(axes[5])

    fig.suptitle(
        "ADB Monte Carlo Convergence Analysis",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = OUTPUT_FOLDER / "monte_carlo_all_metrics.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("Saved combined graph:", output_path)



# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():

    start_time = time.time()

    print("\nADB MONTE CARLO CONVERGENCE EXPERIMENT")
    print("======================================")

    # 1. Calculate exact values

    print("\nCalculating exact reference values...")

    exact_values = get_exact_values()

    # 2. Generate one million simulations

    print("\nGenerating one million simulated terms...")

    annual = annual_probability_distribution()

    simulations = generate_simulations(
        MAXIMUM_SIMULATIONS,
        annual
    )

    # 3. Calculate estimates at different sample sizes

    results = {
        metric: []
        for metric in simulations
    }

    print("\nComparing sample sizes...")

    for sample_size in SAMPLE_SIZES:

        print(f"\nSample size: {sample_size:,}")

        for metric, observations in simulations.items():

            estimate = np.mean(
                observations[:sample_size]
            )

            exact = exact_values[metric]

            results[metric].append(estimate)

            relative_error = (
                abs(estimate - exact) / abs(exact) * 100
                if exact != 0
                else 0
            )

            if metric in [
                "Expected principal loss",
                "Expected stop-loss"
            ]:

                print(
                    f"{metric}: "
                    f"${estimate:,.2f} "
                    f"(Exact: ${exact:,.2f}) "
                    f"(Error: {relative_error:.3f}%)"
                )

            else:

                print(
                    f"{metric}: "
                    f"{estimate * 100:.6f}% "
                    f"(Exact: {exact * 100:.6f}%) "
                    f"(Error: {relative_error:.3f}%)"
                )

    # 4. Save numerical results

    rows = []

    for metric in results:

        for i, sample_size in enumerate(SAMPLE_SIZES):

            estimate = results[metric][i]
            exact = exact_values[metric]

            relative_error = (
                abs(estimate - exact) / abs(exact) * 100
                if exact != 0
                else 0
            )

            rows.append({
                "Metric": metric,
                "Sample Size": sample_size,
                "Monte Carlo Estimate": estimate,
                "Exact Value": exact,
                "Relative Error (%)": relative_error
            })

    dataframe = pd.DataFrame(rows)

    csv_path = OUTPUT_FOLDER / "convergence_results.csv"

    dataframe.to_csv(
        csv_path,
        index=False
    )

    print("\nSaved numerical results:", csv_path)

    # 5. Generate five individual graphs

    print("\nGenerating convergence graphs...")

    plot_convergence(
        results,
        exact_values
    )

    # 6. Finish

    runtime = time.time() - start_time

    print("\nEXPERIMENT COMPLETED")
    print("--------------------")
    print(f"Runtime: {runtime:.2f} seconds")
    print("Five convergence graphs generated.")


if __name__ == "__main__":
    main()
