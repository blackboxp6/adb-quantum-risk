
import numpy as np

from probability_model import annual_probability_distribution


# --------------------------------------------------
# 1. THREE-YEAR DISTRIBUTION BY CONVOLUTION
# --------------------------------------------------

def calculate_term_distribution(years=3):

    annual = annual_probability_distribution()

    # Start with no accumulated disaster payout.
    term = {
        (0, 0, 0, 0): 1.0
    }

    for year in range(years):

        next_term = {}

        for old_state, old_probability in term.items():

            for annual_state, annual_probability in annual.items():

                a, b, r1, r2 = old_state

                da, db, dr1, dr2 = annual_state

                # Add annual payouts and apply contract caps.
                new_state = (
                    min(a + da, 4),
                    min(b + db, 4),
                    min(r1 + dr1, 1),
                    min(r2 + dr2, 1)
                )

                probability = (
                    old_probability * annual_probability
                )

                next_term[new_state] = (
                    next_term.get(new_state, 0.0)
                    + probability
                )

        term = next_term

        print(
            f"After year {year + 1}: "
            f"{len(term)} term states"
        )

    return term


# --------------------------------------------------
# 2. CALCULATE INSURANCE LOSSES
# --------------------------------------------------

def calculate_losses(state):

    a, b, r1, r2 = state

    loss_1 = 70_000_000 * a / 4 + 10_000_000 * r1

    loss_2 = 70_000_000 * b / 4 + 10_000_000 * r2

    portfolio_loss = loss_1 + loss_2

    return loss_1, loss_2, portfolio_loss


# --------------------------------------------------
# 3. CALCULATE THE FIVE ADB TARGETS
# --------------------------------------------------

def calculate_metrics(term):

    expected_loss = 0.0
    joint_attachment = 0.0
    exceedance = 0.0
    stop_loss = 0.0
    both_exhausted = 0.0

    threshold = 80_000_000

    for state, probability in term.items():

        loss_1, loss_2, total_loss = calculate_losses(state)

        # Target 1: Expected portfolio principal loss
        expected_loss += probability * total_loss

        # Target 2: Both countries receive a payout
        if loss_1 > 0 and loss_2 > 0:
            joint_attachment += probability

        # Target 3: Portfolio loss exceeds $80 million
        if total_loss > threshold:
            exceedance += probability

        # Target 4: Expected portfolio stop-loss
        stop_loss += probability * max(
            total_loss - threshold, 0
        )

        # Target 5: Both bonds are exhausted
        if loss_1 == 80_000_000 and loss_2 == 80_000_000:
            both_exhausted += probability

    return {
        "Expected principal loss": expected_loss,
        "Both countries pay": joint_attachment,
        "Portfolio loss exceeds 80m": exceedance,
        "Expected stop-loss": stop_loss,
        "Both bonds exhausted": both_exhausted
    }


# --------------------------------------------------
# 4. RUN THE COMPLETE MODEL
# --------------------------------------------------

if __name__ == "__main__":

    print("\nADB THREE-YEAR LOSS MODEL")
    print("=========================")

    term = calculate_term_distribution(years=3)

    print("\nTotal term probability:")
    print(sum(term.values()))

    print("\nNumber of term states:")
    print(len(term))

    metrics = calculate_metrics(term)

    print("\nADB RISK METRICS")
    print("----------------")

    for name, value in metrics.items():

        if name in [
            "Expected principal loss",
            "Expected stop-loss"
        ]:
            print(f"{name}: ${value:,.2f}")

        else:
            print(f"{name}: {value * 100:.6f}%")

    assert np.isclose(sum(term.values()), 1.0)

    print("\nSUCCESS: Three-year model completed.")
