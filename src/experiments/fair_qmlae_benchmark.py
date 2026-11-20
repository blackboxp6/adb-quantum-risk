
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import xlogy

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(MODEL_DIR))

from term_distribution import (
    calculate_term_distribution,
    calculate_losses,
)

BUDGETS = [1024, 4096, 8192, 16384]
REPETITIONS = 100

GROVER_POWERS = {
    "Expected principal loss": [0, 1, 2],
    "Both countries pay": [0, 1, 2],
    "Portfolio loss exceeds 80m": [0, 1, 2, 4],
    "Expected stop-loss": [0, 1, 2, 4],
    "Both bonds exhausted": [0, 1, 2, 4],
}

THRESHOLD = 80_000_000

# Number of points used for maximum-likelihood estimation.
GRID_POINTS = 30_001

RANDOM_SEED = 2026


# ============================================================
# 2. LOAD THE EXACT ADB TERM DISTRIBUTION
# ============================================================

def load_reference_distribution():

    print("\nLOADING TWO-COUNTRY ADB DISTRIBUTION")
    print("=" * 65)

    term_distribution = calculate_term_distribution(years=3)

    states = sorted(term_distribution.keys())

    probabilities = np.array(
        [term_distribution[state] for state in states],
        dtype=float,
    )

    probabilities /= probabilities.sum()

    assert len(states) == 100
    assert np.isclose(probabilities.sum(), 1.0)

    print(f"Term outcomes: {len(states)}")
    print(f"Probability sum: {probabilities.sum():.12f}")

    return states, probabilities


# ============================================================
# 3. CALCULATE THE FIVE METRICS
# ============================================================

def prepare_metrics(states, probabilities):

    loss_1 = []
    loss_2 = []
    total_loss = []

    for state in states:

        first, second, total = calculate_losses(state)

        loss_1.append(first)
        loss_2.append(second)
        total_loss.append(total)

    loss_1 = np.asarray(loss_1, dtype=float)
    loss_2 = np.asarray(loss_2, dtype=float)
    total_loss = np.asarray(total_loss, dtype=float)

    stop_loss = np.maximum(
        total_loss - THRESHOLD,
        0.0,
    )

    # Each metric is normalized to the interval [0, 1].
    # This is required for the ancilla probability encoding.

    metrics = {

        "Expected principal loss": (
            total_loss / 160_000_000
        ),

        "Both countries pay": (
            (loss_1 > 0) & (loss_2 > 0)
        ).astype(float),

        "Portfolio loss exceeds 80m": (
            total_loss > THRESHOLD
        ).astype(float),

        "Expected stop-loss": (
            stop_loss / 80_000_000
        ),

        "Both bonds exhausted": (
            (loss_1 == 80_000_000)
            & (loss_2 == 80_000_000)
        ).astype(float),
    }

    scales = {
        "Expected principal loss": 160_000_000,
        "Both countries pay": 1.0,
        "Portfolio loss exceeds 80m": 1.0,
        "Expected stop-loss": 80_000_000,
        "Both bonds exhausted": 1.0,
    }

    exact_values = {}

    for name, values in metrics.items():

        normalized_expectation = np.dot(
            probabilities,
            values,
        )

        exact_values[name] = (
            normalized_expectation * scales[name]
        )

    print("\nEXACT REFERENCE VALUES")
    print("-" * 65)

    for name, value in exact_values.items():
        print(f"{name:35s}: {value:,.10f}")

    return metrics, scales, exact_values


# ============================================================
# 4. ALLOCATE A SHOT BUDGET ACROSS GROVER POWERS
# ============================================================

def allocate_shots(total_budget, powers):

    number_of_powers = len(powers)

    base = total_budget // number_of_powers
    remainder = total_budget % number_of_powers

    allocation = np.full(
        number_of_powers,
        base,
        dtype=int,
    )

    allocation[:remainder] += 1

    assert allocation.sum() == total_budget

    return allocation


# ============================================================
# 5. PREPARE THE MLE GRID
# ============================================================

def prepare_likelihood_grid(powers):

    maximum_power = max(powers)

    # Restricted interval to avoid phase ambiguity
    # for the selected Grover powers.

    theta_max = np.pi / (
        2 * (2 * maximum_power + 1)
    )

    theta_grid = np.linspace(
        1e-9,
        theta_max,
        GRID_POINTS,
    )

    log_probabilities = {}

    log_failure_probabilities = {}

    for k in powers:

        predicted = np.sin(
            (2 * k + 1) * theta_grid
        ) ** 2

        predicted = np.clip(
            predicted,
            1e-15,
            1 - 1e-15,
        )

        log_probabilities[k] = np.log(predicted)

        log_failure_probabilities[k] = np.log1p(
            -predicted
        )

    return (
        theta_grid,
        log_probabilities,
        log_failure_probabilities,
    )


# ============================================================
# 6. ESTIMATE ONE METRIC USING QMLAE
# ============================================================

def estimate_one_metric(
    exact_normalized_probability,
    scale,
    powers,
    total_budget,
    rng,
    likelihood_grid,
):

    (
        theta_grid,
        log_probabilities,
        log_failure_probabilities,
    ) = likelihood_grid

    shot_allocation = allocate_shots(
        total_budget,
        powers,
    )

    theta = np.arcsin(
        np.sqrt(exact_normalized_probability)
    )

    log_likelihood = np.zeros(
        GRID_POINTS,
        dtype=float,
    )

    total_grover_applications = 0

    # Simulate measurements for each Grover power.

    for k, shots in zip(powers, shot_allocation):

        theoretical_probability = np.sin(
            (2 * k + 1) * theta
        ) ** 2

        successes = rng.binomial(
            shots,
            theoretical_probability,
        )

        failures = shots - successes

        log_likelihood += (
            successes * log_probabilities[k]
            + failures * log_failure_probabilities[k]
        )

        total_grover_applications += (
            shots * k
        )

    # Maximum-likelihood estimate.

    best_index = np.argmax(log_likelihood)

    estimated_theta = theta_grid[best_index]

    estimated_normalized_probability = np.sin(
        estimated_theta
    ) ** 2

    estimate = (
        estimated_normalized_probability * scale
    )

    return (
        estimate,
        total_grover_applications,
        total_budget,
    )


# ============================================================
# 7. RUN THE REPEATED QMLAE BENCHMARK
# ============================================================

def run_benchmark(metrics, scales, exact_values):

    rng = np.random.default_rng(RANDOM_SEED)

    raw_results = []
    summary_results = []

    print("\nSTARTING REPEATED QMLAE BENCHMARK")
    print("=" * 65)

    start_time = time.perf_counter()

    for metric_name, values in metrics.items():

        powers = GROVER_POWERS[metric_name]
        scale = scales[metric_name]
        exact_value = exact_values[metric_name]

        exact_normalized_probability = (
            exact_value / scale
        )

        likelihood_grid = prepare_likelihood_grid(
            powers
        )

        print(f"\nMetric: {metric_name}")
        print(f"Grover powers: {powers}")

        for budget in BUDGETS:

            estimates = []
            grover_counts = []

            for repetition in range(1, REPETITIONS + 1):

                estimate, grover_calls, shots = (
                    estimate_one_metric(
                        exact_normalized_probability,
                        scale,
                        powers,
                        budget,
                        rng,
                        likelihood_grid,
                    )
                )

                estimates.append(estimate)
                grover_counts.append(grover_calls)

                absolute_error = abs(
                    estimate - exact_value
                )

                raw_results.append({
                    "Metric": metric_name,
                    "Budget": budget,
                    "Repetition": repetition,
                    "Exact value": exact_value,
                    "Estimate": estimate,
                    "Absolute error": absolute_error,
                    "Grover applications": grover_calls,
                    "Measurement shots": shots,
                    "Grover powers": str(powers),
                })

            estimates = np.asarray(estimates)

            errors = estimates - exact_value

            mae = np.mean(np.abs(errors))

            rmse = np.sqrt(
                np.mean(errors ** 2)
            )

            relative_mae = (
                mae / abs(exact_value) * 100
                if exact_value != 0
                else np.nan
            )

            relative_rmse = (
                rmse / abs(exact_value) * 100
                if exact_value != 0
                else np.nan
            )

            average_grover_calls = np.mean(
                grover_counts
            )

            summary_results.append({
                "Metric": metric_name,
                "Budget": budget,
                "Repetitions": REPETITIONS,
                "Exact value": exact_value,
                "MAE": mae,
                "RMSE": rmse,
                "Relative MAE (%)": relative_mae,
                "Relative RMSE (%)": relative_rmse,
                "Average shots": budget,
                "Average Grover applications": (
                    average_grover_calls
                ),
                "Grover powers": str(powers),
                "Qubits": 7,
            })

            print(
                f"Budget={budget:6,d} | "
                f"MAE={mae:,.8f} | "
                f"RMSE={rmse:,.8f} | "
                f"Relative MAE={relative_mae:.4f}%"
            )

    elapsed_time = time.perf_counter() - start_time

    print("\nBENCHMARK RUNTIME")
    print(f"{elapsed_time:.2f} seconds")

    return (
        pd.DataFrame(raw_results),
        pd.DataFrame(summary_results),
        elapsed_time,
    )


# ============================================================
# 8. SAVE RESULTS
# ============================================================

def save_results(raw_df, summary_df, elapsed_time):

    raw_path = (
        OUTPUT_DIR / "fair_qmlae_raw_results.csv"
    )

    summary_path = (
        OUTPUT_DIR / "fair_qmlae_summary.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    runtime_path = (
        OUTPUT_DIR / "fair_qmlae_runtime.csv"
    )

    pd.DataFrame([{
        "Runtime seconds": elapsed_time,
        "Repetitions": REPETITIONS,
        "Budgets": str(BUDGETS),
        "Metrics": len(GROVER_POWERS),
    }]).to_csv(
        runtime_path,
        index=False,
    )

    print("\nSAVED FILES")
    print(raw_path)
    print(summary_path)
    print(runtime_path)


# ============================================================
# 9. PLOT MAE FOR EACH METRIC
# ============================================================

def plot_mae(summary_df):

    for metric_name in summary_df["Metric"].unique():

        subset = summary_df[
            summary_df["Metric"] == metric_name
        ].sort_values("Budget")

        plt.figure(figsize=(8, 5))

        plt.plot(
            subset["Budget"],
            subset["MAE"],
            marker="o",
            linewidth=2,
        )

        plt.xscale("log", base=2)

        plt.xlabel("Measurement-shot budget")
        plt.ylabel("Mean absolute error")

        plt.title(
            f"QMLAE MAE — {metric_name}"
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_name = (
            metric_name.lower()
            .replace(" ", "_")
            .replace("$", "")
        )

        output_path = (
            OUTPUT_DIR
            / f"fair_qmlae_mae_{safe_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
        )

        plt.close()

        print("Saved plot:", output_path)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("\nFAIR QMLAE BENCHMARK — TWO-COUNTRY ADB")
    print("=" * 65)

    states, probabilities = (
        load_reference_distribution()
    )

    metrics, scales, exact_values = prepare_metrics(
        states,
        probabilities,
    )

    raw_df, summary_df, elapsed_time = run_benchmark(
        metrics,
        scales,
        exact_values,
    )

    save_results(
        raw_df,
        summary_df,
        elapsed_time,
    )

    plot_mae(summary_df)

    print("\nFINAL QMLAE SUMMARY")
    print("=" * 65)

    print(
        summary_df[
            [
                "Metric",
                "Budget",
                "MAE",
                "RMSE",
                "Relative MAE (%)",
                "Average Grover applications",
            ]
        ].to_string(index=False)
    )

    print("\nTOTAL BENCHMARK CONFIGURATION")
    print(f"Metrics: {len(metrics)}")
    print(f"Budgets: {BUDGETS}")
    print(f"Repetitions per metric and budget: {REPETITIONS}")
    print(f"Total independent estimates: {len(raw_df):,}")

    print("\nIMPORTANT:")
    print("Measurements are simulated using ideal amplitude")
    print("amplification probabilities and binomial sampling.")
    print("This is not a real quantum-hardware experiment.")

    print("\nBENCHMARK COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
