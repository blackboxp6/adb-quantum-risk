
import numpy as np

import probability_model as adb
import multicountry_probability_model as multi


# --------------------------------------------------
# 1. REUSE THE ORIGINAL ADB PARAMETERS
# --------------------------------------------------

country_rates = [
    adb.LAMBDA_1,
    adb.LAMBDA_2,
]

lambda_1 = adb.TOTAL_RATE_1
lambda_2 = adb.TOTAL_RATE_2
c = adb.SHARED_EARTHQUAKE_RATE

print("ADB SHARED EARTHQUAKE RATE:", c)


# --------------------------------------------------
# 2. BUILD IDENTICAL EARTHQUAKE EVENT SCOPES
# --------------------------------------------------

earthquake_scope_rates = {
    (0,): lambda_1 - c,
    (1,): lambda_2 - c,
    (0, 1): c,
}

# Calculate the generalized model's distribution.
e_states, e_probs = multi.earthquake_distribution(
    country_rates,
    earthquake_scope_rates
)

# Calculate the original ADB distribution.
adb_e = adb.earthquake_distribution()

# Convert both to dictionaries indexed by state.
multi_e = {
    tuple(state): probability
    for state, probability in zip(e_states, e_probs)
}

original_e = {
    (a, b): adb_e[a, b]
    for a in range(5)
    for b in range(5)
}

earthquake_error = max(
    abs(multi_e[state] - original_e[state])
    for state in original_e
)


# --------------------------------------------------
# 3. BUILD IDENTICAL PRECIPITATION EVENT SCOPES
# --------------------------------------------------

q1 = 0.02
q2 = 0.02

mu1 = -np.log(1 - q1)
mu2 = -np.log(1 - q2)

shared_rain_rate = 0.25 * min(mu1, mu2)

rain_scope_rates = {
    (0,): mu1 - shared_rain_rate,
    (1,): mu2 - shared_rain_rate,
    (0, 1): shared_rain_rate,
}

r_states, r_probs = multi.precipitation_distribution(
    [mu1, mu2],
    rain_scope_rates
)

# Original ADB precipitation distribution.
adb_r = adb.precipitation_distribution()

multi_r = {
    tuple(state): probability
    for state, probability in zip(r_states, r_probs)
}

original_r = {
    (r1, r2): adb_r[r1, r2]
    for r1 in range(2)
    for r2 in range(2)
}

rain_error = max(
    abs(multi_r[state] - original_r[state])
    for state in original_r
)


# --------------------------------------------------
# 4. COMPARE COMPLETE ANNUAL DISTRIBUTIONS
# --------------------------------------------------

# Original ADB annual distribution.
adb_annual = adb.annual_probability_distribution()

# Generalized annual distribution.
annual_states, annual_probs = multi.annual_joint_distribution(
    e_states,
    e_probs,
    r_states,
    r_probs
)

multi_annual = {
    tuple(state): probability
    for state, probability in zip(
        annual_states,
        annual_probs
    )
}

annual_error = max(
    abs(multi_annual[state] - adb_annual[state])
    for state in adb_annual
)


# --------------------------------------------------
# 5. PRINT VALIDATION RESULTS
# --------------------------------------------------

print("\nTWO-COUNTRY VALIDATION")
print("=====================")

print("\nEarthquake states:", len(e_states))
print("Rain states:", len(r_states))
print("Annual states:", len(annual_states))

print("\nMaximum absolute probability differences:")
print(f"Earthquake:    {earthquake_error:.12e}")
print(f"Precipitation: {rain_error:.12e}")
print(f"Annual:        {annual_error:.12e}")

# Probability of no earthquake payout in either country.
no_earthquake = multi_e[(0, 0)]

print("\nProbability of no earthquake payout:")
print(f"{no_earthquake * 100:.4f}%")

# Probability of no payout of any kind.
no_payout = multi_annual[(0, 0, 0, 0)]

print("\nProbability of no payout of any kind:")
print(f"{no_payout * 100:.4f}%")

print("\nProbability sum:")
print(f"{sum(multi_annual.values()):.12f}")


# --------------------------------------------------
# 6. AUTOMATIC PASS / FAIL TESTS
# --------------------------------------------------

tolerance = 1e-10

assert earthquake_error < tolerance
assert rain_error < tolerance
assert annual_error < tolerance

assert np.isclose(
    sum(multi_annual.values()),
    1.0,
    atol=tolerance
)

print("\nSUCCESS: Generalized model reproduces the ADB model.")
