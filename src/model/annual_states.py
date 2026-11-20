
from itertools import product


def generate_annual_states():
    """
    Generate all possible annual ADB states.

    a  = earthquake payout level, country 1
    b  = earthquake payout level, country 2
    r1 = precipitation payout, country 1
    r2 = precipitation payout, country 2
    """

    earthquake_levels = range(5)
    precipitation_levels = range(2)

    states = list(product(
        earthquake_levels,
        earthquake_levels,
        precipitation_levels,
        precipitation_levels
    ))

    return states


if __name__ == "__main__":
    states = generate_annual_states()

    print("ADB ANNUAL STATE MODEL")
    print("----------------------")

    print("Number of possible states:", len(states))

    print("\nFirst 10 states:")

    for state in states[:10]:
        print(state)

    print("\nLast state:")
    print(states[-1])

    assert len(states) == 100

    print("\nSUCCESS: All 100 annual states generated.")
