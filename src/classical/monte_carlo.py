
import sys
import time
from pathlib import Path

import numpy as np

# Allow this file to import the model when run as a script or from an IDE.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_FOLDER = PROJECT_ROOT / "src"
MODEL_FOLDER = SRC_FOLDER / "model"

for path in (str(PROJECT_ROOT), str(SRC_FOLDER), str(MODEL_FOLDER)):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from src.model.probability_model import annual_probability_distribution
except ModuleNotFoundError:  # pragma: no cover - fallback for some editor setups
    from probability_model import annual_probability_distribution


# --------------------------------------------------
# 1. SIMULATION SETTINGS
# --------------------------------------------------

NUMBER_OF_PATHS = 100_000
NUMBER_OF_YEARS = 3
RANDOM_SEED = 42

THRESHOLD = 80_000_000


# --------------------------------------------------
# 2. PREPARE THE ANNUAL PROBABILITY DISTRIBUTION
# --------------------------------------------------

def prepare_annual_distribution():

    annual = annual_probability_distribution()

    # Convert the dictionary into arrays.
    states = np.array(list(annual.keys()), dtype=int)

    probabilities = np.array(
        list(annual.values()),
        dtype=float
    )

    # Cumulative distribution for inverse-CDF sampling.
    cumulative_probabilities = np.cumsum(probabilities)

    # Protect against tiny floating-point rounding errors.
    cumulative_probabilities[-1] = 1.0

    return states, probabilities, cumulative_probabilities


# --------------------------------------------------
# 3. GENERATE RANDOM THREE-YEAR SCENARIOS
# --------------------------------------------------

def simulate_term_states(number_of_paths, rng):

    states, probabilities, cdf = prepare_annual_distribution()

    # Generate three independent uniform random numbers per path.
    uniforms = rng.random(
        (number_of_paths, NUMBER_OF_YEARS)
    )

    # Convert each uniform number into an annual state.
    indices = np.searchsorted(
        cdf,
        uniforms,
        side="right"
    )

    annual_draws = states[indices]

    # Accumulate earthquake payouts and apply the contract caps.
    a = np.minimum(
        annual_draws[:, :, 0].sum(axis=1),
        4
    )

    b = np.minimum(
        annual_draws[:, :, 1].sum(axis=1),
        4
    )

    # Precipitation triggers are capped at one.
    r1 = np.minimum(
        annual_draws[:, :, 2].sum(axis=1),
        1
    )

    r2 = np.minimum(
        annual_draws[:, :, 3].sum(axis=1),
        1
    )

    term_states = np.column_stack((a, b, r1, r2))

    return term_states


# --------------------------------------------------
# 4. CALCULATE LOSSES FOR EVERY SIMULATED TERM
# --------------------------------------------------

def calculate_path_losses(term_states):

    a = term_states[:, 0]
    b = term_states[:, 1]
    r1 = term_states[:, 2]
    r2 = term_states[:, 3]

    loss_1 = 70_000_000 * a / 4 + 10_000_000 * r1

    loss_2 = 70_000_000 * b / 4 + 10_000_000 * r2

    portfolio_loss = loss_1 + loss_2

    return loss_1, loss_2, portfolio_loss


# --------------------------------------------------
# 5. ESTIMATE THE FIVE ADB RISK METRICS
# --------------------------------------------------

def estimate_metrics(term_states):

    loss_1, loss_2, total_loss = calculate_path_losses(
        term_states
    )

    observations = {
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

    results = {}

    for name, values in observations.items():

        estimate = np.mean(values)

        standard_error = (
            np.std(values, ddof=1)
            / np.sqrt(len(values))
        )

        results[name] = {
            "estimate": estimate,
            "standard_error": standard_error
        }

    return results


# --------------------------------------------------
# 6. RUN THE MONTE CARLO EXPERIMENT
# --------------------------------------------------

if __name__ == "__main__":

    print("\nADB MONTE CARLO SIMULATION")
    print("==========================")

    print("Number of simulated terms:", NUMBER_OF_PATHS)
    print("Years per term:", NUMBER_OF_YEARS)
    print("Random seed:", RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)

    start_time = time.perf_counter()

    term_states = simulate_term_states(
        NUMBER_OF_PATHS,
        rng
    )

    results = estimate_metrics(term_states)

    elapsed_time = time.perf_counter() - start_time

    print("\nMONTE CARLO RESULTS")
    print("-------------------")

    for name, result in results.items():

        estimate = result["estimate"]
        standard_error = result["standard_error"]

        if name in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:

            print(f"\n{name}:")
            print(f"  Estimate:       ${estimate:,.2f}")
            print(f"  Standard error: ${standard_error:,.2f}")

        else:

            print(f"\n{name}:")
            print(f"  Estimate:       {estimate * 100:.6f}%")
            print(f"  Standard error: {standard_error * 100:.6f} percentage points")

    print("\nSIMULATION INFORMATION")
    print("----------------------")
    print(f"Runtime: {elapsed_time:.4f} seconds")

    exhausted_count = np.sum(
        (term_states[:, 0] == 4)
        & (term_states[:, 1] == 4)
        & (term_states[:, 2] == 1)
        & (term_states[:, 3] == 1)
    )

    print(
        "Both-bonds-exhausted observations:",
        exhausted_count,
        "out of",
        NUMBER_OF_PATHS
    )

    print("\nSUCCESS: Monte Carlo simulation completed.")
