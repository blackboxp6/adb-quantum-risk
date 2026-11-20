
from pathlib import Path
import sys
import csv
from collections import defaultdict
from itertools import product

import numpy as np

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.multicountry_probability_model import (
    earthquake_distribution,
    precipitation_distribution,
    annual_joint_distribution,
)


# ============================================================
# 1. Country parameters
# ============================================================

# Countries A and B use the ADB earthquake rates.
# Country C uses illustrative rates, not official ADB data.

country_rates = [
    [0.016147, 0.005498, 0.006215, 0.011743],
    [0.022711, 0.004020, 0.002915, 0.010395],
    [0.018000, 0.004500, 0.003500, 0.009000],
]

n = len(country_rates)

marginal_eq = np.array(
    [sum(rates) for rates in country_rates],
    dtype=float
)

# Each country has a 2% annual precipitation trigger probability.
mu = -np.log(0.98)
marginal_rain = np.array([mu] * n)

# ADB-calibrated A-B shared earthquake rate.
c_ab = 0.5 * (
    0.075 * marginal_eq[0]
    + 0.074 * marginal_eq[1]
)

# Shared precipitation assumptions.
c_rain_ab = 0.25 * mu
c_rain_abc = 0.10 * mu

# Illustrative triple-country earthquake common-shock rate.
c_abc = 0.0005


# ============================================================
# 2. Build the three dependence scenarios
# ============================================================

def build_scenario(name):
    """
    Construct event-scope rates while preserving each
    country's marginal earthquake and rain rates.
    """

    earthquake_scopes = {}
    rain_scopes = {}

    if name == "Independent":

        # Each country has its own earthquake process.
        for i in range(n):
            earthquake_scopes[(i,)] = marginal_eq[i]
            rain_scopes[(i,)] = marginal_rain[i]

    elif name == "ADB A-B dependence":

        # A and B share earthquake events.
        earthquake_scopes[(0, 1)] = c_ab

        for i in range(n):
            if i in (0, 1):
                earthquake_scopes[(i,)] = (
                    marginal_eq[i] - c_ab
                )
            else:
                earthquake_scopes[(i,)] = marginal_eq[i]

        # A and B share precipitation events.
        rain_scopes[(0, 1)] = c_rain_ab

        for i in range(n):
            if i in (0, 1):
                rain_scopes[(i,)] = (
                    marginal_rain[i] - c_rain_ab
                )
            else:
                rain_scopes[(i,)] = marginal_rain[i]

    elif name == "Three-country common shock":

        # A-B shared earthquake process.
        earthquake_scopes[(0, 1)] = c_ab

        # A-B-C common earthquake process.
        earthquake_scopes[(0, 1, 2)] = c_abc

        for i in range(n):
            deduction = c_ab if i in (0, 1) else 0.0
            earthquake_scopes[(i,)] = (
                marginal_eq[i] - deduction - c_abc
            )

        # Shared precipitation processes.
        rain_scopes[(0, 1)] = c_rain_ab
        rain_scopes[(0, 1, 2)] = c_rain_abc

        for i in range(n):
            deduction = (
                c_rain_ab if i in (0, 1) else 0.0
            )
            rain_scopes[(i,)] = (
                marginal_rain[i]
                - deduction
                - c_rain_abc
            )

    else:
        raise ValueError(f"Unknown scenario: {name}")

    return earthquake_scopes, rain_scopes


# ============================================================
# 3. Calculate the annual probability distribution
# ============================================================

def calculate_annual_distribution(name):

    eq_scopes, rain_scopes = build_scenario(name)

    eq_states, eq_probabilities = earthquake_distribution(
        country_rates,
        eq_scopes
    )

    rain_states, rain_probabilities = precipitation_distribution(
        marginal_rain,
        rain_scopes
    )

    states, probabilities = annual_joint_distribution(
        eq_states,
        eq_probabilities,
        rain_states,
        rain_probabilities
    )

    return states, probabilities


# ============================================================
# 4. Accumulate losses over three years
# ============================================================

def three_year_distribution(annual_states, annual_probabilities):

    # State format:
    # (eq_A, eq_B, eq_C, rain_A, rain_B, rain_C)

    zero_state = (0, 0, 0, 0, 0, 0)
    term_distribution = {zero_state: 1.0}

    for year in range(3):

        updated = defaultdict(float)

        for old_state, old_probability in term_distribution.items():

            for new_state, new_probability in zip(
                annual_states,
                annual_probabilities
            ):

                combined = []

                # Earthquake payout steps: add and cap at 4.
                for i in range(n):
                    combined.append(
                        min(old_state[i] + new_state[i], 4)
                    )

                # Rain triggers: logical OR.
                for i in range(n):
                    combined.append(
                        max(old_state[n + i], new_state[n + i])
                    )

                combined = tuple(combined)

                updated[combined] += (
                    old_probability * new_probability
                )

        term_distribution = dict(updated)

        print(
            f"  Completed year {year + 1}: "
            f"{len(term_distribution)} accumulated states"
        )

    return term_distribution


# ============================================================
# 5. Calculate portfolio loss and risk metrics
# ============================================================

def calculate_metrics(term_distribution):

    threshold = 120_000_000
    expected_loss = 0.0
    probability_all_three = 0.0
    probability_exceedance = 0.0
    expected_stop_loss = 0.0

    for state, probability in term_distribution.items():

        earthquake_steps = state[:n]
        rain_triggers = state[n:]

        country_losses = []

        for i in range(n):

            earthquake_loss = (
                70_000_000 * earthquake_steps[i] / 4
            )

            precipitation_loss = (
                10_000_000 * rain_triggers[i]
            )

            country_losses.append(
                earthquake_loss + precipitation_loss
            )

        portfolio_loss = sum(country_losses)

        expected_loss += probability * portfolio_loss

        # All three countries experience a positive loss.
        if all(loss > 0 for loss in country_losses):
            probability_all_three += probability

        # Portfolio loss exceeds $120 million.
        if portfolio_loss > threshold:
            probability_exceedance += probability

        # Expected amount above the threshold.
        expected_stop_loss += (
            probability
            * max(portfolio_loss - threshold, 0)
        )

    return {
        "expected_loss": expected_loss,
        "all_three": probability_all_three,
        "exceedance": probability_exceedance,
        "stop_loss": expected_stop_loss,
    }


# ============================================================
# 6. Run all scenarios
# ============================================================

scenarios = [
    "Independent",
    "ADB A-B dependence",
    "Three-country common shock",
]

results = {}

print("\nTHREE-YEAR JOINT LOSS ANALYSIS")
print("=" * 65)

for scenario in scenarios:

    print(f"\nScenario: {scenario}")

    annual_states, annual_probabilities = (
        calculate_annual_distribution(scenario)
    )

    print(
        "Annual probability sum:",
        f"{annual_probabilities.sum():.12f}"
    )

    term_distribution = three_year_distribution(
        annual_states,
        annual_probabilities
    )

    term_probability_sum = sum(term_distribution.values())

    print(
        "Three-year probability sum:",
        f"{term_probability_sum:.12f}"
    )

    results[scenario] = calculate_metrics(term_distribution)

# ============================================================
# 7. Export three-country common-shock distribution
# ============================================================

output_path = (
    PROJECT_ROOT
    / "outputs"
    / "three_country_common_shock_term_distribution.csv"
)

output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "earthquake_A",
        "earthquake_B",
        "earthquake_C",
        "rain_A",
        "rain_B",
        "rain_C",
        "probability"
    ])

    for state, probability in term_distribution.items():

        writer.writerow([
            *state,
            probability
        ])

print(f"\nSaved quantum input distribution to: {output_path}")


# ============================================================
# 8. Display comparison
# ============================================================

print("\n\nFINAL COMPARISON")
print("=" * 90)

print(
    f"{'Scenario':32s}"
    f"{'Expected loss':>18s}"
    f"{'All three':>14s}"
    f"{'Exceedance':>14s}"
    f"{'Stop-loss':>16s}"
)

print("-" * 90)

for scenario, metrics in results.items():

    print(
        f"{scenario:32s}"
        f"${metrics['expected_loss']/1e6:>15.4f}M"
        f"{metrics['all_three']*100:>13.5f}%"
        f"{metrics['exceedance']*100:>13.5f}%"
        f"${metrics['stop_loss']/1e6:>13.5f}M"
    )

print("\nANALYSIS COMPLETE.")
