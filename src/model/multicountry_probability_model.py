
from itertools import product
from math import exp
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply


# ------------------------------------------------------------
# 1. Basic validation
# ------------------------------------------------------------

def validate_event_scopes(n_countries, event_rates, marginal_rates, label):
    """
    event_rates:
        Dictionary mapping a tuple of affected country indices
        to an event rate.

        Example for 3 countries:
        {
            (0,): 0.01,       # country 1 only
            (1,): 0.01,       # country 2 only
            (2,): 0.01,       # country 3 only
            (0, 1): 0.002,    # countries 1 and 2
            (0, 1, 2): 0.001  # all three countries
        }

    marginal_rates:
        Total annual event rate for each country.
    """
    if len(marginal_rates) != n_countries:
        raise ValueError("One marginal rate is required per country.")

    for scope, rate in event_rates.items():
        if not scope:
            raise ValueError(f"{label}: event scope cannot be empty.")

        if len(set(scope)) != len(scope):
            raise ValueError(f"{label}: duplicate country in scope {scope}.")

        if any(i < 0 or i >= n_countries for i in scope):
            raise ValueError(f"{label}: invalid country index in {scope}.")

        if rate < 0:
            raise ValueError(f"{label}: event rates cannot be negative.")

    implied_rates = np.zeros(n_countries)

    for scope, rate in event_rates.items():
        for country in scope:
            implied_rates[country] += rate

    if not np.allclose(implied_rates, marginal_rates, atol=1e-10):
        raise ValueError(
            f"{label}: event scopes do not preserve country marginal rates.\n"
            f"Expected: {marginal_rates}\n"
            f"Implied:  {implied_rates}"
        )

    return implied_rates


# ------------------------------------------------------------
# 2. Severity distributions
# ------------------------------------------------------------

def severity_weights(country_rates):
    """
    Convert compound-Poisson severity rates into conditional
    probabilities of severity k = 1, 2, 3, 4.

    country_rates[i] = [lambda_1, lambda_2, lambda_3, lambda_4]
    """
    weights = []

    for rates in country_rates:
        rates = np.asarray(rates, dtype=float)

        if rates.shape != (4,):
            raise ValueError("Each country needs four severity rates.")

        total = rates.sum()

        if total <= 0:
            raise ValueError("Country severity rates must sum to > 0.")

        weights.append(rates / total)

    return weights


# ------------------------------------------------------------
# 3. Multi-country earthquake CTMC
# ------------------------------------------------------------

def earthquake_distribution(
    country_rates,
    event_scope_rates,
    year_length=1.0
):
    """
    Exact annual joint earthquake payout-step distribution.

    Each country's cumulative earthquake payout is represented
    by a state in {0, 1, 2, 3, 4}, capped at 4.

    An event scope can affect any subset of countries.

    For each event:
      - the event scope determines which countries are hit;
      - each affected country's severity is drawn independently;
      - each country's payout state is capped at 4.

    Returns:
      states: array with shape (5**n, n)
      probabilities: array with shape (5**n,)
    """
    n_countries = len(country_rates)

    marginal_rates = np.array(
        [sum(rates) for rates in country_rates],
        dtype=float
    )

    validate_event_scopes(
        n_countries,
        event_scope_rates,
        marginal_rates,
        "Earthquake"
    )

    weights = severity_weights(country_rates)

    states = list(product(range(5), repeat=n_countries))
    state_to_index = {
        state: i for i, state in enumerate(states)
    }

    rows = []
    cols = []
    values = []

    for source_index, state in enumerate(states):
        outgoing_rate = 0.0

        for scope, event_rate in event_scope_rates.items():
            if event_rate == 0:
                continue

            # Severity is independently drawn for each country
            # participating in this shared event.
            severity_options = [
                range(1, 5) for _ in scope
            ]

            for severities in product(*severity_options):
                probability = event_rate

                for country, severity in zip(scope, severities):
                    probability *= weights[country][severity - 1]

                new_state = list(state)

                for country, severity in zip(scope, severities):
                    new_state[country] = min(
                        new_state[country] + severity, 4
                    )

                new_state = tuple(new_state)

                # Events that leave a capped state unchanged
                # do not create an off-diagonal CTMC transition.
                if new_state == state:
                    continue

                target_index = state_to_index[new_state]

                rows.append(source_index)
                cols.append(target_index)
                values.append(probability)

                outgoing_rate += probability

        rows.append(source_index)
        cols.append(source_index)
        values.append(-outgoing_rate)

    generator = coo_matrix(
        (values, (rows, cols)),
        shape=(len(states), len(states))
    ).tocsr()

    # Row-vector convention: p(t) = p(0) exp(Gt).
    # Therefore evolve the transposed column vector.
    initial = np.zeros(len(states))
    initial[state_to_index[(0,) * n_countries]] = 1.0

    probabilities = expm_multiply(
        generator.T * year_length,
        initial
    )

    probabilities = np.maximum(probabilities, 0.0)
    probabilities /= probabilities.sum()

    return np.asarray(states), probabilities


# ------------------------------------------------------------
# 4. Multi-country shared precipitation model
# ------------------------------------------------------------

def precipitation_distribution(
    country_rates,
    event_scope_rates
):
    """
    Exact distribution of annual precipitation trigger bits.

    Each event scope is an independent Poisson process.
    If one or more events occur in a scope, every country
    in that scope receives a precipitation trigger.

    Multiple event scopes combine through logical OR.

    Each country's marginal Poisson rate is preserved.
    """
    n_countries = len(country_rates)

    marginal_rates = np.asarray(country_rates, dtype=float)

    validate_event_scopes(
        n_countries,
        event_scope_rates,
        marginal_rates,
        "Precipitation"
    )

    # A mask represents which countries have triggered.
    distribution = {0: 1.0}

    for scope, rate in event_scope_rates.items():
        mask = sum(1 << country for country in scope)

        event_probability = 1.0 - exp(-rate)
        no_event_probability = exp(-rate)

        updated = defaultdict(float)

        for old_mask, old_probability in distribution.items():
            updated[old_mask] += (
                old_probability * no_event_probability
            )

            new_mask = old_mask | mask
            updated[new_mask] += (
                old_probability * event_probability
            )

        distribution = dict(updated)

    states = []
    probabilities = []

    for mask in range(2 ** n_countries):
        bits = tuple(
            (mask >> country) & 1
            for country in range(n_countries)
        )

        states.append(bits)
        probabilities.append(distribution.get(mask, 0.0))

    probabilities = np.asarray(probabilities, dtype=float)
    probabilities /= probabilities.sum()

    return np.asarray(states), probabilities


# ------------------------------------------------------------
# 5. Combine earthquake and precipitation
# ------------------------------------------------------------

def annual_joint_distribution(
    earthquake_states,
    earthquake_probabilities,
    rain_states,
    rain_probabilities
):
    """
    Combine earthquake and precipitation distributions.

    Assumption: earthquake and precipitation event systems
    are independent, as in the ADB worked example.

    Annual state ordering:
        (earthquake_1, ..., earthquake_n,
         rain_1, ..., rain_n)
    """
    annual_states = []
    annual_probabilities = []

    for e_state, e_probability in zip(
        earthquake_states, earthquake_probabilities
    ):
        for r_state, r_probability in zip(
            rain_states, rain_probabilities
        ):
            annual_states.append(
                tuple(e_state) + tuple(r_state)
            )

            annual_probabilities.append(
                e_probability * r_probability
            )

    annual_probabilities = np.asarray(
        annual_probabilities,
        dtype=float
    )

    annual_probabilities /= annual_probabilities.sum()

    return (
        np.asarray(annual_states),
        annual_probabilities
    )


# ------------------------------------------------------------
# 6. Example: three countries
# ------------------------------------------------------------

def three_country_example():
    """
    Illustrative three-country model.

    These are demonstration parameters, NOT ADB data.

    Countries:
      0 = Country A
      1 = Country B
      2 = Country C

    The example includes:
      - country-specific earthquakes;
      - a shared A-B earthquake process;
      - a shared A-B-C earthquake process;
      - country-specific rain;
      - shared A-B rain;
      - shared A-B-C rain.
    """

    # Each row contains lambda_1, lambda_2, lambda_3, lambda_4.
    country_rates = [
        [0.016147, 0.005498, 0.006215, 0.011743],
        [0.022711, 0.004020, 0.002915, 0.010395],
        [0.018000, 0.004500, 0.003500, 0.009000],
    ]

    marginal_rates = [
        sum(rates) for rates in country_rates
    ]

    # Earthquake event scopes.
    #
    # Shared rates are illustrative. The remaining rate for
    # each country is assigned to its country-specific events.
    shared_ab = 0.0010
    shared_abc = 0.0005

    earthquake_scope_rates = {
        (0,): marginal_rates[0] - shared_ab - shared_abc,
        (1,): marginal_rates[1] - shared_ab - shared_abc,
        (2,): marginal_rates[2] - shared_abc,
        (0, 1): shared_ab,
        (0, 1, 2): shared_abc,
    }

    # Precipitation marginal rates:
    # q = 2%, so mu = -ln(1 - q).
    rain_rate = -np.log(1.0 - 0.02)

    shared_rain_ab = 0.25 * rain_rate
    shared_rain_abc = 0.10 * rain_rate

    rain_scope_rates = {
        (0,): rain_rate - shared_rain_ab - shared_rain_abc,
        (1,): rain_rate - shared_rain_ab - shared_rain_abc,
        (2,): rain_rate - shared_rain_abc,
        (0, 1): shared_rain_ab,
        (0, 1, 2): shared_rain_abc,
    }

    e_states, e_probs = earthquake_distribution(
        country_rates,
        earthquake_scope_rates
    )

    r_states, r_probs = precipitation_distribution(
        [rain_rate] * 3,
        rain_scope_rates
    )

    annual_states, annual_probs = annual_joint_distribution(
        e_states,
        e_probs,
        r_states,
        r_probs
    )

    print("THREE-COUNTRY MODEL")
    print("-------------------")
    print("Earthquake states:", len(e_states))
    print("Rain states:", len(r_states))
    print("Annual joint states:", len(annual_states))
    print("Probability sum:", annual_probs.sum())

    no_payout = np.all(annual_states == 0, axis=1)
    print(
        "Probability of no payout in any country:",
        annual_probs[no_payout].sum()
    )

    # Probability that all three countries have at least
    # one earthquake payout step or precipitation trigger.
    n = len(country_rates)

    all_three_pay = np.all(
        (annual_states[:, :n] > 0) |
        (annual_states[:, n:] > 0),
        axis=1
    )

    print(
        "Probability all three countries trigger:",
        annual_probs[all_three_pay].sum()
    )

    return annual_states, annual_probs


if __name__ == "__main__":
    three_country_example()
