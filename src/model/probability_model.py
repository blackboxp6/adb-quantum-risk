
import numpy as np
from scipy.linalg import expm

from annual_states import generate_annual_states


# --------------------------------------------------
# 1. ADB BASE-CASE EARTHQUAKE PARAMETERS
# --------------------------------------------------

LAMBDA_1 = np.array([
    0.016147,
    0.005498,
    0.006215,
    0.011743
])

LAMBDA_2 = np.array([
    0.022711,
    0.004020,
    0.002915,
    0.010395
])

TOTAL_RATE_1 = LAMBDA_1.sum()
TOTAL_RATE_2 = LAMBDA_2.sum()

SHARED_EARTHQUAKE_RATE = (
    0.5 * (
        0.075 * TOTAL_RATE_1
        + 0.074 * TOTAL_RATE_2
    )
)


# --------------------------------------------------
# 2. BUILD THE 25-STATE EARTHQUAKE GENERATOR
# --------------------------------------------------

def build_earthquake_generator():

    G = np.zeros((25, 25))

    weights_1 = LAMBDA_1 / TOTAL_RATE_1
    weights_2 = LAMBDA_2 / TOTAL_RATE_2

    c = SHARED_EARTHQUAKE_RATE

    for a in range(5):
        for b in range(5):

            current = a * 5 + b

            # Country 1 only
            for k in range(1, 5):

                new_a = min(a + k, 4)
                destination = new_a * 5 + b

                rate = (
                    (TOTAL_RATE_1 - c)
                    * weights_1[k - 1]
                )

                if destination != current:
                    G[current, destination] += rate

            # Country 2 only
            for j in range(1, 5):

                new_b = min(b + j, 4)
                destination = a * 5 + new_b

                rate = (
                    (TOTAL_RATE_2 - c)
                    * weights_2[j - 1]
                )

                if destination != current:
                    G[current, destination] += rate

            # Shared earthquake
            for k in range(1, 5):
                for j in range(1, 5):

                    new_a = min(a + k, 4)
                    new_b = min(b + j, 4)

                    destination = new_a * 5 + new_b

                    rate = (
                        c
                        * weights_1[k - 1]
                        * weights_2[j - 1]
                    )

                    if destination != current:
                        G[current, destination] += rate

    # Markov generator:
    # Each row must sum to zero.
    for i in range(25):
        G[i, i] = -np.sum(G[i, :])

    return G


# --------------------------------------------------
# 3. CALCULATE ANNUAL EARTHQUAKE DISTRIBUTION
# --------------------------------------------------

def earthquake_distribution():

    G = build_earthquake_generator()

    transition_matrix = expm(G)

    # Start from state (0, 0)
    probabilities = transition_matrix[0, :]

    return probabilities.reshape(5, 5)


# --------------------------------------------------
# 4. CALCULATE PRECIPITATION DISTRIBUTION
# --------------------------------------------------

def precipitation_distribution():

    q1 = 0.02
    q2 = 0.02

    mu1 = -np.log(1 - q1)
    mu2 = -np.log(1 - q2)

    shared_rate = 0.25 * min(mu1, mu2)

    p00 = np.exp(-(mu1 + mu2 - shared_rate))

    p10 = np.exp(-mu2) - p00

    p01 = np.exp(-mu1) - p00

    p11 = 1 - p00 - p10 - p01

    return np.array([
        [p00, p01],
        [p10, p11]
    ])


# --------------------------------------------------
# 5. COMBINE EARTHQUAKE AND PRECIPITATION
# --------------------------------------------------

def annual_probability_distribution():

    earthquake = earthquake_distribution()

    precipitation = precipitation_distribution()

    states = generate_annual_states()

    probabilities = {}

    for a, b, r1, r2 in states:

        p_earthquake = earthquake[a, b]

        p_rain = precipitation[r1, r2]

        probabilities[(a, b, r1, r2)] = (
            p_earthquake * p_rain
        )

    return probabilities


# --------------------------------------------------
# 6. TEST THE MODEL
# --------------------------------------------------

if __name__ == "__main__":

    earthquake = earthquake_distribution()

    precipitation = precipitation_distribution()

    annual = annual_probability_distribution()

    print("\nADB ANNUAL PROBABILITY MODEL")
    print("---------------------------")

    print("\nEarthquake probabilities sum:")
    print(earthquake.sum())

    print("\nPrecipitation probabilities:")
    print(precipitation)

    print("\nPrecipitation probabilities sum:")
    print(precipitation.sum())

    print("\nNumber of annual states:")
    print(len(annual))

    print("\nTotal annual probability:")
    print(sum(annual.values()))

    no_payout = annual[(0, 0, 0, 0)]

    print("\nProbability of no payout:")
    print(f"{no_payout * 100:.4f}%")

    assert np.isclose(earthquake.sum(), 1.0)
    assert np.isclose(precipitation.sum(), 1.0)
    assert np.isclose(sum(annual.values()), 1.0)

    print("\nSUCCESS: Probability model is normalized.")
