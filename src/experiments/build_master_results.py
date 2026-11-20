
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# SOURCE FILES
# --------------------------------------------------

source_files = {
    "Classical and QMLAE accuracy comparison":
        "fair_classical_qmlae_comparison.csv",

    "Quantum circuit scalability":
        "quantum_circuit_scaling.csv",

    "Quantum noise experiment":
        "two_country_noise_repeated_summary.csv",

    "Quantum resource estimation":
        "quantum_resource_results.csv",

    "Quantum circuit component resources":
        "quantum_component_resources.csv",

    "Classical and QMLAE computational cost":
        "resource_cost_comparison.csv",
}

# --------------------------------------------------
# LOAD AND COMBINE RESULTS
# --------------------------------------------------

all_results = []
included_files = []
missing_files = []

for experiment, filename in source_files.items():

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        missing_files.append(filename)
        print(f"WARNING: Missing file: {filename}")
        continue

    try:
        df = pd.read_csv(file_path)

        # Preserve original data and add experiment metadata
        df.insert(0, "Experiment", experiment)
        df.insert(1, "Source file", filename)

        all_results.append(df)
        included_files.append(filename)

        print(
            f"Loaded: {filename} "
            f"({len(df)} rows)"
        )

    except Exception as error:
        print(f"ERROR reading {filename}: {error}")

# --------------------------------------------------
# COMBINE DATA
# --------------------------------------------------

if not all_results:
    raise FileNotFoundError(
        "No source CSV files were found. "
        "Check the filenames in the outputs folder."
    )

master_df = pd.concat(
    all_results,
    ignore_index=True,
    sort=False
)

# --------------------------------------------------
# SAVE MASTER CSV
# --------------------------------------------------

master_csv = OUTPUT_DIR / "master_evaluation_results.csv"

master_df.to_csv(
    master_csv,
    index=False
)

# --------------------------------------------------
# CREATE SUMMARY REPORT
# --------------------------------------------------

summary = (
    master_df
    .groupby("Experiment", dropna=False)
    .size()
    .reset_index(name="Number of rows")
)

summary_path = OUTPUT_DIR / "master_evaluation_summary.csv"

summary.to_csv(
    summary_path,
    index=False
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("MASTER EVALUATION RESULTS")
print("=" * 55)

print(f"\nTotal rows: {len(master_df)}")
print(f"Total columns: {len(master_df.columns)}")
print(f"Files included: {len(included_files)}")

print("\nIncluded files:")
for filename in included_files:
    print(f"  - {filename}")

if missing_files:
    print("\nMissing files:")
    for filename in missing_files:
        print(f"  - {filename}")

print("\nRows by experiment:")
print(summary.to_string(index=False))

print("\nSaved files:")
print(master_csv)
print(summary_path)

print("\nConsolidation completed.")
