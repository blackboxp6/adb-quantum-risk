
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(MODEL_PATH))

import probability_model as adb
import multicountry_probability_model as multi


# ------------------------------------------------------------
# 1. PARAMETERS
# ------------------------------------------------------------

country_rates = [
    adb.LAMBDA_1,
    adb.LAMBDA_2,
    np.array([0.018000, 0.004500, 0.003500, 0.009000]),
]

n = 3

marginal_rates = [
    sum(rates) for rates in country_rates
]

mu = -np.log(0.98)
rain_marginal_rates = [mu] * n

adb_shared = adb.SHARED_EARTHQUAKE_RATE

# Keep the A-B shared earthquake rate fixed.
# Vary only the three-country common earthquake rate.
common_shock_rates = [
    0.0000,
    0.0005,
    0.0010,
    0.0020,
    0.0030,
    0.0050,
]


# ------------------------------------------------------------
# 2. FIXED RAIN DEPENDENCE
# ------------------------------------------------------------

shared_rain_ab = 0.25 * mu

rain_scopes = {
    (0,): mu - shared_rain_ab,
    (1,): mu - shared_rain_ab,
    (2,): mu,
    (0, 1): shared_rain_ab,
}

r_states, r_probs = multi.precipitation_distribution(
    rain_marginal_rates,
    rain_scopes
)


# ------------------------------------------------------------
# 3. RUN EACH COMMON-SHOCK SCENARIO
# ------------------------------------------------------------

all_three_earthquake_results = []
all_three_any_results = []

for common_rate in common_shock_rates:

    # Preserve every country's marginal earthquake rate.
    earthquake_scopes = {
        (0,): marginal_rates[0] - adb_shared - common_rate,
        (1,): marginal_rates[1] - adb_shared - common_rate,
        (2,): marginal_rates[2] - common_rate,
        (0, 1): adb_shared,
        (0, 1, 2): common_rate,
    }

    # Check that all rates remain valid and marginals are preserved.
    multi.validate_event_scopes(
        n,
        earthquake_scopes,
        marginal_rates,
        "Earthquake"
    )

    e_states, e_probs = multi.earthquake_distribution(
        country_rates,
        earthquake_scopes
    )

    annual_states, annual_probs = multi.annual_joint_distribution(
        e_states,
        e_probs,
        r_states,
        r_probs
    )

    earthquake = annual_states[:, :n]
    rain = annual_states[:, n:]

    all_three_earthquake = np.all(
        earthquake > 0,
        axis=1
    )

    country_pays = (earthquake > 0) | (rain > 0)

    all_three_any = np.all(
        country_pays,
        axis=1
    )

    p_earthquake = annual_probs[
        all_three_earthquake
    ].sum()

    p_any = annual_probs[
        all_three_any
    ].sum()

    all_three_earthquake_results.append(p_earthquake)
    all_three_any_results.append(p_any)

    print(
        f"Common rate: {common_rate:.4f} | "
        f"All-three earthquake: {p_earthquake * 100:.6f}% | "
        f"All-three any payout: {p_any * 100:.6f}%"
    )


# ------------------------------------------------------------
# 4. PLOT THE RESULTS
# ------------------------------------------------------------

rates = np.array(common_shock_rates)

earthquake_results = np.array(
    all_three_earthquake_results
)

any_results = np.array(
    all_three_any_results
)

plt.figure(figsize=(9, 5))

plt.plot(
    rates,
    earthquake_results * 100,
    marker="o",
    label="All three: earthquake payouts"
)

plt.plot(
    rates,
    any_results * 100,
    marker="s",
    label="All three: any payout"
)

plt.xlabel("Three-country shared earthquake rate (per year)")
plt.ylabel("Probability (%)")
plt.title("Sensitivity of Three-Country Joint Payout Risk")
plt.grid(True)
plt.legend()
plt.tight_layout()

output_dir = Path(__file__).resolve().parents[2] / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(
    output_dir / "common_shock_sensitivity.png",
    dpi=300
)

plt.show()

print("\nSaved graph to outputs/common_shock_sensitivity.png")
