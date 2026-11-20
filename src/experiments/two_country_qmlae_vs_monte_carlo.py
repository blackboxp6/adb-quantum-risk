
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import xlogy


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.insert(0, str(MODEL_DIR))

from term_distribution import calculate_term_distribution, calculate_losses


THRESHOLD = 80_000_000

# QMLAE uses these Grover powers.
GROVER_POWERS = [0, 1, 2, 4]

# Total measurements per experiment, across all four powers.
BUDGETS = [1024, 4096, 8192, 16384]

# Independent repetitions at each budget.
REPETITIONS = 100

# Number of points used in the likelihood search.
GRID_POINTS = 30_001

RANDOM_SEED = 42


# ==================================================
# 2. LOAD THE EXACT ADB DISTRIBUTION
# ==================================================

print("\nTWO-COUNTRY QMLAE VS MONTE CARLO")
print("=" * 60)

term_distribution = calculate_term_distribution(years=3)
states = sorted(term_distribution.keys())

probabilities = np.array(
    [term_distribution[state] for state in states],
    dtype=float,
)

assert np.isclose(probabilities.sum(), 1.0)

marked = np.array([
    calculate_losses(state)[2] > THRESHOLD
    for state in states
])

exact_probability = probabilities[marked].sum()

print(f"Number of outcomes: {len(states)}")
print(f"Exact exceedance probability: {exact_probability:.10%}")
print(f"Grover powers: {GROVER_POWERS}")
print(f"Independent repetitions: {REPETITIONS}")


# ==================================================
# 3. QMLAE LIKELIHOOD ESTIMATOR
# ==================================================

def estimate_qmlae(success_counts, shots_per_power):
    """
    Estimate the original marked probability from
    measurements after different Grover powers.
    """

    k_max = max(GROVER_POWERS)

    theta_max = np.pi / (2 * (2 * k_max + 1))

    theta_grid = np.linspace(
        1e-8,
        theta_max,
        GRID_POINTS,
    )

    log_likelihood = np.zeros_like(theta_grid)

    for k, successes in zip(GROVER_POWERS, success_counts):

        failures = shots_per_power - successes

        predicted = np.sin(
            (2 * k + 1) * theta_grid
        ) ** 2

        predicted = np.clip(
            predicted,
            1e-15,
            1 - 1e-15,
        )

        log_likelihood += (
            xlogy(successes, predicted)
            + xlogy(failures, 1 - predicted)
        )

    best_index = np.argmax(log_likelihood)

    estimated_theta = theta_grid[best_index]

    return np.sin(estimated_theta) ** 2


# ==================================================
# 4. RUN REPEATED EXPERIMENTS
# ==================================================

rng = np.random.default_rng(RANDOM_SEED)

theta_exact = np.arcsin(np.sqrt(exact_probability))

qmlae_theoretical_probabilities = {
    k: np.sin((2 * k + 1) * theta_exact) ** 2
    for k in GROVER_POWERS
}

all_results = []

print("\nRUNNING EXPERIMENTS")
print("-" * 60)

for budget in BUDGETS:

    # Equal allocation to each Grover power.
    shots_per_power = budget // len(GROVER_POWERS)
    actual_budget = shots_per_power * len(GROVER_POWERS)

    mc_estimates = []
    qmlae_estimates = []

    for repetition in range(REPETITIONS):

        # ------------------------------------------
        # CLASSICAL MONTE CARLO
        # ------------------------------------------

        sampled_indices = rng.choice(
            len(states),
            size=actual_budget,
            p=probabilities,
        )

        mc_estimate = np.mean(marked[sampled_indices])

        mc_estimates.append(mc_estimate)

        # ------------------------------------------
        # QMLAE MEASUREMENT SIMULATION
        # ------------------------------------------

        success_counts = []

        for k in GROVER_POWERS:

            theoretical_probability = (
                qmlae_theoretical_probabilities[k]
            )

            successes = rng.binomial(
                shots_per_power,
                theoretical_probability,
            )

            success_counts.append(successes)

        qmlae_estimate = estimate_qmlae(
            success_counts,
            shots_per_power,
        )

        qmlae_estimates.append(qmlae_estimate)

    # ----------------------------------------------
    # CALCULATE ERROR METRICS
    # ----------------------------------------------

    for method, estimates in [
        ("Monte Carlo", mc_estimates),
        ("QMLAE", qmlae_estimates),
    ]:

        estimates = np.asarray(estimates)

        errors = estimates - exact_probability

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        bias = np.mean(errors)
        standard_deviation = np.std(estimates, ddof=1)

        all_results.append({
            "Method": method,
            "Measurement budget": actual_budget,
            "Repetitions": REPETITIONS,
            "Mean estimate": np.mean(estimates),
            "MAE": mae,
            "RMSE": rmse,
            "Bias": bias,
            "Standard deviation": standard_deviation,
        })

    print(
        f"Budget {actual_budget:>6,} | "
        f"MC MAE: {np.mean(np.abs(np.array(mc_estimates) - exact_probability)):.8f} | "
        f"QMLAE MAE: {np.mean(np.abs(np.array(qmlae_estimates) - exact_probability)):.8f}"
    )


# ==================================================
# 5. SAVE NUMERICAL RESULTS
# ==================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(all_results)

results_file = OUTPUT_DIR / "two_country_qmlae_vs_monte_carlo.csv"

results_df.to_csv(results_file, index=False)


# ==================================================
# 6. PLOT MEAN ABSOLUTE ERROR
# ==================================================

plt.figure(figsize=(9, 6))

for method in ["Monte Carlo", "QMLAE"]:

    subset = results_df[results_df["Method"] == method]

    plt.plot(
        subset["Measurement budget"],
        subset["MAE"],
        marker="o",
        linewidth=2,
        label=method,
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Total measurement budget")
plt.ylabel("Mean absolute estimation error")

plt.title(
    "Two-Country ADB Benchmark: "
    "QMLAE vs Monte Carlo"
)

plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "two_country_qmlae_vs_monte_carlo_mae.png",
    dpi=300,
)

plt.close()


# ==================================================
# 7. PLOT ROOT MEAN SQUARED ERROR
# ==================================================

plt.figure(figsize=(9, 6))

for method in ["Monte Carlo", "QMLAE"]:

    subset = results_df[results_df["Method"] == method]

    plt.plot(
        subset["Measurement budget"],
        subset["RMSE"],
        marker="o",
        linewidth=2,
        label=method,
    )

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Total measurement budget")
plt.ylabel("Root mean squared estimation error")

plt.title(
    "Two-Country ADB Benchmark: "
    "QMLAE vs Monte Carlo (RMSE)"
)

plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "two_country_qmlae_vs_monte_carlo_rmse.png",
    dpi=300,
)

plt.close()


# ==================================================
# 8. REPORT RESOURCE ACCOUNTING
# ==================================================

print("\nRESOURCE ACCOUNTING")
print("-" * 60)

print("QMLAE Grover powers:", GROVER_POWERS)

for budget in BUDGETS:

    shots_per_power = budget // len(GROVER_POWERS)

    total_shots = shots_per_power * len(GROVER_POWERS)

    grover_applications_per_trial = (
        shots_per_power * sum(GROVER_POWERS)
    )

    print(
        f"Budget {total_shots:>6,} | "
        f"QMLAE Grover applications: "
        f"{grover_applications_per_trial:,} per trial"
    )


# ==================================================
# 9. DISPLAY FINAL SUMMARY
# ==================================================

print("\nFINAL COMPARISON")
print("=" * 60)

print(
    results_df[
        [
            "Method",
            "Measurement budget",
            "Mean estimate",
            "MAE",
            "RMSE",
        ]
    ].to_string(index=False)
)

print("\nFILES SAVED")
print("-" * 60)
print(results_file)
print(OUTPUT_DIR / "two_country_qmlae_vs_monte_carlo_mae.png")
print(OUTPUT_DIR / "two_country_qmlae_vs_monte_carlo_rmse.png")

print("\nEXPERIMENT COMPLETE")
