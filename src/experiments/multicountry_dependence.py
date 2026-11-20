
import sys
from pathlib import Path

import numpy as np

# Allow this script to import the model from src/model.
MODEL_PATH = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_PATH))

import probability_model as adb
import multicountry_probability_model as multi


# ============================================================
# 1. COUNTRY PARAMETERS
# ============================================================

# Countries A and B use the original ADB marginal rates.
# Country C is illustrative, not calibrated ADB data.

country_rates = [
    adb.LAMBDA_1,
    adb.LAMBDA_2,
    np.array([0.018000, 0.004500, 0.003500, 0.009000]),
]

n = len(country_rates)

marginal_earthquake_rates = [
    sum(rates) for rates in country_rates
]

# Annual precipitation Poisson rate for a 2% trigger probability.
mu = -np.log(1.0 - 0.02)
rain_marginal_rates = [mu] * n

# Original ADB shared-event rates for countries A and B.
adb_shared_earthquake = adb.SHARED_EARTHQUAKE_RATE
adb_shared_rain = 0.25 * mu

# Illustrative additional three-country common shocks.
triple_earthquake = 0.0005
triple_rain = 0.10 * mu


# ============================================================
# 2. BUILD THE THREE DEPENDENCE SCENARIOS
# ============================================================

def build_scenario(name):

    earthquake_scopes = {}
    rain_scopes = {}

    if name == "Independent":

        # No shared events.
        for i in range(n):
            earthquake_scopes[(i,)] = (
                marginal_earthquake_rates[i]
            )
            rain_scopes[(i,)] = mu

    elif name == "ADB A-B dependence":

        # A and B share the original ADB event rates.
        # Country C remains independent.

        earthquake_scopes = {
            (0,): marginal_earthquake_rates[0]
                  - adb_shared_earthquake,

            (1,): marginal_earthquake_rates[1]
                  - adb_shared_earthquake,

            (2,): marginal_earthquake_rates[2],

            (0, 1): adb_shared_earthquake,
        }

        rain_scopes = {
            (0,): mu - adb_shared_rain,
            (1,): mu - adb_shared_rain,
            (2,): mu,
            (0, 1): adb_shared_rain,
        }

    elif name == "Three-country common shock":

        # A and B retain their ADB shared events.
        # All three countries can also experience a common event.
        #
        # The private rates are reduced so that every country's
        # original marginal event rate remains unchanged.

        earthquake_scopes = {
            (0,): marginal_earthquake_rates[0]
                  - adb_shared_earthquake
                  - triple_earthquake,

            (1,): marginal_earthquake_rates[1]
                  - adb_shared_earthquake
                  - triple_earthquake,

            (2,): marginal_earthquake_rates[2]
                  - triple_earthquake,

            (0, 1): adb_shared_earthquake,

            (0, 1, 2): triple_earthquake,
        }

        rain_scopes = {
            (0,): mu - adb_shared_rain - triple_rain,
            (1,): mu - adb_shared_rain - triple_rain,
            (2,): mu - triple_rain,
            (0, 1): adb_shared_rain,
            (0, 1, 2): triple_rain,
        }

    else:
        raise ValueError(f"Unknown scenario: {name}")

    return earthquake_scopes, rain_scopes


# ============================================================
# 3. CALCULATE EXACT ANNUAL DISTRIBUTION
# ============================================================

def calculate_scenario(name):

    earthquake_scopes, rain_scopes = build_scenario(name)

    e_states, e_probs = multi.earthquake_distribution(
        country_rates,
        earthquake_scopes
    )

    r_states, r_probs = multi.precipitation_distribution(
        rain_marginal_rates,
        rain_scopes
    )

    annual_states, annual_probs = multi.annual_joint_distribution(
        e_states,
        e_probs,
        r_states,
        r_probs
    )

    return annual_states, annual_probs


# ============================================================
# 4. CALCULATE JOINT-LOSS INDICATORS
# ============================================================

def calculate_metrics(states, probabilities):

    # First three columns: earthquake payout steps.
    # Last three columns: precipitation trigger bits.

    earthquake = states[:, :n]
    rain = states[:, n:]

    # A country experiences a payout if either its earthquake
    # payout step is positive or its rain trigger is active.
    country_pays = (earthquake > 0) | (rain > 0)

    number_paying = country_pays.sum(axis=1)

    all_three_earthquake = np.all(earthquake > 0, axis=1)

    return {
        "No country pays": probabilities[number_paying == 0].sum(),

        "At least one pays": probabilities[number_paying >= 1].sum(),

        "At least two pay": probabilities[number_paying >= 2].sum(),

        "All three pay": probabilities[number_paying == 3].sum(),

        "All three have earthquake payouts": (
            probabilities[all_three_earthquake].sum()
        ),
    }


# ============================================================
# 5. RUN ALL SCENARIOS
# ============================================================

scenarios = [
    "Independent",
    "ADB A-B dependence",
    "Three-country common shock",
]

results = {}

print("\nTHREE-COUNTRY DEPENDENCE EXPERIMENT")
print("===================================")

for scenario in scenarios:

    states, probabilities = calculate_scenario(scenario)

    metrics = calculate_metrics(states, probabilities)

    results[scenario] = metrics

    print(f"\nSCENARIO: {scenario}")
    print("-" * 45)

    print("Number of annual states:", len(states))
    print(f"Probability sum: {probabilities.sum():.12f}")

    for metric, value in metrics.items():
        print(f"{metric:40s}: {value * 100:.6f}%")


# ============================================================
# 6. COMPARE THE EFFECT OF DEPENDENCE
# ============================================================

print("\n\nEFFECT OF SHARED EVENTS")
print("=======================")

baseline = results["Independent"]

for scenario in scenarios[1:]:

    print(f"\n{scenario}")

    for metric in [
        "At least two pay",
        "All three pay",
        "All three have earthquake payouts",
    ]:

        difference = (
            results[scenario][metric] - baseline[metric]
        ) * 100

        print(
            f"{metric:40s}: "
            f"{difference:+.6f} percentage points"
        )


# ============================================================
# 7. VERIFY MARGINAL RATES AND PROBABILITY NORMALIZATION
# ============================================================

for scenario in scenarios:

    earthquake_scopes, rain_scopes = build_scenario(scenario)

    multi.validate_event_scopes(
        n,
        earthquake_scopes,
        marginal_earthquake_rates,
        "Earthquake"
    )

    multi.validate_event_scopes(
        n,
        rain_scopes,
        rain_marginal_rates,
        "Precipitation"
    )

    states, probabilities = calculate_scenario(scenario)

    assert np.isclose(probabilities.sum(), 1.0)
    assert len(states) == 1000

print("\nSUCCESS: All scenarios preserve marginal rates.")
print("SUCCESS: All three annual distributions are normalized.")
