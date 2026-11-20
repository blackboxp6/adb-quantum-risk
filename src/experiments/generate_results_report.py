
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs"

MASTER = OUTPUT / "master_evaluation_results.csv"
REPORT = OUTPUT / "adb_results_draft.md"

df = pd.read_csv(MASTER)

lines = []

lines.append("# ADB Quantum Risk Evaluation — Results Draft\n")
lines.append(
    "This report summarizes numerical benchmark results from the "
    "classical sampling and quantum amplitude estimation experiments.\n"
)

# --------------------------------------------------
# 1. ACCURACY COMPARISON
# --------------------------------------------------

lines.append("## 1. Estimation Accuracy\n")

accuracy = df[
    df["Experiment"] == "Classical and QMLAE accuracy comparison"
].copy()

if not accuracy.empty:

    if "Budget" in accuracy.columns:
        accuracy = accuracy[
            accuracy["Budget"] == accuracy["Budget"].max()
        ]

    lines.append(
        "The following table summarizes the estimation accuracy "
        "at the largest available budget.\n"
    )

    display_cols = [
        col for col in [
            "Method",
            "Metric",
            "Budget",
            "Relative MAE (%)",
            "Relative RMSE (%)",
            "MAE",
            "RMSE",
        ]
        if col in accuracy.columns
    ]

    if display_cols:
        lines.append(
            accuracy[display_cols]
            .to_markdown(index=False)
        )
        lines.append("\n")

else:
    lines.append("Accuracy comparison data was not found.\n")

# --------------------------------------------------
# 2. SCALABILITY
# --------------------------------------------------

lines.append("## 2. Circuit Scalability\n")

scaling = df[
    df["Experiment"] == "Quantum circuit scalability"
].copy()

if not scaling.empty:

    display_cols = [
        col for col in [
            "countries",
            "Countries",
            "outcomes",
            "qubits",
            "depth",
            "total_gates",
            "cx_gates",
            "u_gates",
        ]
        if col in scaling.columns
    ]

    if display_cols:
        lines.append(
            scaling[display_cols].to_markdown(index=False)
        )
        lines.append("\n")

    lines.append(
        "These scaling results are based on synthetic probability "
        "distributions and generic state preparation. They should not "
        "be interpreted as a fully calibrated multi-country ADB model.\n"
    )

else:
    lines.append("Circuit scaling data was not found.\n")

# --------------------------------------------------
# 3. NOISE
# --------------------------------------------------

lines.append("## 3. Quantum Noise Sensitivity\n")

noise = df[
    df["Experiment"] == "Quantum noise experiment"
].copy()

if not noise.empty:

    display_cols = [
        col for col in [
            "Noise rate",
            "noise_rate",
            "Mean relative error (%)",
            "mean_relative_error_percent",
            "SD relative error (%)",
            "sd_relative_error_percent",
            "Mean invalid-state rate",
            "mean_invalid_state_rate",
        ]
        if col in noise.columns
    ]

    if display_cols:
        lines.append(
            noise[display_cols].to_markdown(index=False)
        )
        lines.append("\n")

    lines.append(
        "The noise experiment uses a simulated noise model. "
        "The noise parameters do not represent measured error rates "
        "from an actual quantum processor.\n"
    )

else:
    lines.append("Noise data was not found.\n")

# --------------------------------------------------
# 4. QUANTUM RESOURCE ESTIMATION
# --------------------------------------------------

lines.append("## 4. Quantum Resource Estimation\n")

resources = df[
    df["Experiment"] == "Quantum resource estimation"
].copy()

if not resources.empty:

    display_cols = [
        col for col in [
            "Grover iterations",
            "Shots",
            "Circuit depth",
            "U gates per circuit",
            "CX gates per circuit",
            "Total CX gates across shots",
            "Total gates across shots",
            "Total oracle calls across shots",
            "Total state preparations across shots",
        ]
        if col in resources.columns
    ]

    if display_cols:
        lines.append(
            resources[display_cols].to_markdown(index=False)
        )
        lines.append("\n")

else:
    lines.append("Quantum resource data was not found.\n")

# --------------------------------------------------
# 5. COMPONENT RESOURCES
# --------------------------------------------------

lines.append("## 5. Quantum Circuit Components\n")

components = df[
    df["Experiment"] == "Quantum circuit component resources"
].copy()

if not components.empty:

    display_cols = [
        col for col in [
            "Component",
            "Depth",
            "U gates",
            "CX gates",
            "Total gates",
        ]
        if col in components.columns
    ]

    if display_cols:
        lines.append(
            components[display_cols].to_markdown(index=False)
        )
        lines.append("\n")

# --------------------------------------------------
# 6. LIMITATIONS
# --------------------------------------------------

lines.append("## 6. Limitations\n")

lines.append(
    "- Quantum results are simulator-based.\n"
    "- No physical quantum hardware experiment was conducted.\n"
    "- The classical benchmarks use a precomputed probability distribution.\n"
    "- Equal nominal sample or shot budgets do not represent equal "
    "computational costs.\n"
    "- The ideal statistical QMLAE runtime is not quantum hardware runtime.\n"
    "- State preparation and oracle construction are significant "
    "resource considerations.\n"
    "- Noise parameters are simulated and are not hardware measurements.\n"
    "- The scaling experiment uses synthetic distributions.\n"
    "- No quantum advantage is claimed.\n"
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

REPORT.write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("Results report generated:")
print(REPORT)
