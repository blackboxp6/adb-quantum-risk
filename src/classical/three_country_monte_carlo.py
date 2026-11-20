
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path

# --------------------------------------------------
# 1. FILES AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "outputs" /
    "three_country_common_shock_term_distribution.csv"
)

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 120_000_000

SAMPLE_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000
]

SEED = 42

# --------------------------------------------------
# 2. LOAD EXACT TERM DISTRIBUTION
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

probabilities = df["probability"].to_numpy(
    dtype=float,
    copy=True
)

# Normalize to protect against tiny floating-point errors
probabilities = probabilities / probabilities.sum()

# --------------------------------------------------
# 3. CALCULATE PORTFOLIO LOSSES
# --------------------------------------------------

earthquake_A = df["earthquake_A"].to_numpy()
earthquake_B = df["earthquake_B"].to_numpy()
earthquake_C = df["earthquake_C"].to_numpy()

rain_A = df["rain_A"].to_numpy()
rain_B = df["rain_B"].to_numpy()
rain_C = df["rain_C"].to_numpy()

# Each country's loss:
# $70 million maximum earthquake payout
# $10 million precipitation payout

loss_A = (
    70_000_000 * earthquake_A / 4
    + 10_000_000 * rain_A
)

loss_B = (
    70_000_000 * earthquake_B / 4
    + 10_000_000 * rain_B
)

loss_C = (
    70_000_000 * earthquake_C / 4
    + 10_000_000 * rain_C
)

portfolio_loss = loss_A + loss_B + loss_C

# --------------------------------------------------
# 4. EXACT CLASSICAL REFERENCE PROBABILITY
# --------------------------------------------------

marked = portfolio_loss > THRESHOLD

exact_probability = np.sum(
    probabilities[marked]
)

print("=" * 65)
print("THREE-COUNTRY CLASSICAL MONTE CARLO BENCHMARK")
print("=" * 65)

print(f"Number of term outcomes: {len(df)}")
print(f"Portfolio threshold: ${THRESHOLD:,.0f}")
print(f"Exact tail probability: {exact_probability:.10%}")

# Check against the quantum experiment's reference
expected_probability = 0.007263678322602455

print(
    "Difference from quantum reference:",
    f"{abs(exact_probability - expected_probability):.3e}"
)

# --------------------------------------------------
# 5. GENERATE CLASSICAL MONTE CARLO SAMPLES
# --------------------------------------------------

rng = np.random.default_rng(SEED)

max_samples = max(SAMPLE_SIZES)

start_time = time.perf_counter()

# Draw outcomes from the exact probability distribution.
# Sampling is with replacement.
sample_indices = rng.choice(
    len(df),
    size=max_samples,
    replace=True,
    p=probabilities
)

sampled_losses = portfolio_loss[sample_indices]

# Mark whether each sampled portfolio loss exceeds $120M
sampled_marked = sampled_losses > THRESHOLD

total_runtime = time.perf_counter() - start_time

# --------------------------------------------------
# 6. EVALUATE CONVERGENCE
# --------------------------------------------------

results = []

print("\nMONTE CARLO RESULTS")
print("-" * 90)

print(
    f"{'Samples':>12} | "
    f"{'Estimated':>12} | "
    f"{'Abs. error':>12} | "
    f"{'Rel. error':>12} | "
    f"{'Std. error':>12}"
)

for n in SAMPLE_SIZES:

    estimate = np.mean(sampled_marked[:n])

    absolute_error = abs(
        estimate - exact_probability
    )

    relative_error = (
        absolute_error / exact_probability * 100
    )

    standard_error = np.sqrt(
        estimate * (1 - estimate) / n
    )

    results.append({
        "samples": n,
        "estimate": estimate,
        "exact_probability": exact_probability,
        "absolute_error": absolute_error,
        "relative_error_percent": relative_error,
        "standard_error": standard_error
    })

    print(
        f"{n:12,} | "
        f"{estimate:11.6%} | "
        f"{absolute_error:11.6%} | "
        f"{relative_error:11.4f}% | "
        f"{standard_error:11.6%}"
    )

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "three_country_monte_carlo.csv",
    index=False
)

# --------------------------------------------------
# 8. RUNTIME INFORMATION
# --------------------------------------------------

print("\nCOMPUTATIONAL COST")
print("-" * 40)
print(f"Total samples: {max_samples:,}")
print(f"Sampling runtime: {total_runtime:.6f} seconds")
print(
    f"Time per sample: "
    f"{total_runtime / max_samples * 1e6:.4f} microseconds"
)

# --------------------------------------------------
# 9. PLOT CONVERGENCE
# --------------------------------------------------

sample_counts = results_df["samples"]
estimates = results_df["estimate"] * 100

plt.figure(figsize=(9, 5))

plt.plot(
    sample_counts,
    estimates,
    marker="o",
    label="Classical Monte Carlo"
)

plt.axhline(
    exact_probability * 100,
    linestyle="--",
    label="Exact probability"
)

plt.xscale("log")

plt.xlabel("Number of Monte Carlo samples")
plt.ylabel("Estimated tail probability (%)")

plt.title(
    "Three-Country Portfolio Tail-Risk Estimation"
)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "three_country_monte_carlo_convergence.png",
    dpi=300
)

plt.show()

print("\nResults saved.")
print("THREE-COUNTRY MONTE CARLO COMPLETE.")
