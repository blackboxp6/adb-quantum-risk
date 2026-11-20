
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CLASSICAL_FILE = OUTPUT_DIR / "fair_classical_summary.csv"
QMLAE_FILE = OUTPUT_DIR / "fair_qmlae_summary.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUDGETS = [1024, 4096, 8192, 16384]

METRICS = [
    "Expected principal loss",
    "Both countries pay",
    "Portfolio loss exceeds 80m",
    "Expected stop-loss",
    "Both bonds exhausted",
]

# ============================================================
# 2. STANDARDIZE COLUMN NAMES
# ============================================================

def standardize_columns(df):

    df = df.copy()

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    return df


def find_column(df, possible_names, description):

    for name in possible_names:

        if name in df.columns:
            return name

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# 3. STANDARDIZE METRIC NAMES
# ============================================================

def normalize_metric(value):

    name = str(value).lower().strip()

    if "principal" in name:
        return "Expected principal loss"

    if "both" in name and "exhaust" in name:
        return "Both bonds exhausted"

    if "both" in name and (
        "pay" in name or "country" in name
    ):
        return "Both countries pay"

    if "exceed" in name or "80m" in name:
        return "Portfolio loss exceeds 80m"

    if "stop" in name:
        return "Expected stop-loss"

    raise ValueError(
        f"Unknown metric name: {value}"
    )


# ============================================================
# 4. LOAD CLASSICAL RESULTS
# ============================================================

def load_classical_results():

    print("\nLOADING CLASSICAL RESULTS")
    print("-" * 60)

    if not CLASSICAL_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {CLASSICAL_FILE}"
        )

    df = pd.read_csv(CLASSICAL_FILE)
    df = standardize_columns(df)

    print("Classical columns:", list(df.columns))

    method_col = find_column(
        df,
        ["method", "sampling_method", "algorithm"],
        "method",
    )

    metric_col = find_column(
        df,
        ["metric", "risk_metric"],
        "metric",
    )

    budget_col = find_column(
        df,
        ["budget", "samples", "sample_budget"],
        "budget",
    )

    mae_col = find_column(
        df,
        ["mae", "mean_absolute_error"],
        "MAE",
    )

    rmse_col = find_column(
        df,
        ["rmse", "root_mean_squared_error"],
        "RMSE",
    )

    result = pd.DataFrame({

        "Method": df[method_col].astype(str),

        "Metric": df[metric_col].apply(
            normalize_metric
        ),

        "Budget": pd.to_numeric(
            df[budget_col],
            errors="raise",
        ),

        "MAE": pd.to_numeric(
            df[mae_col],
            errors="raise",
        ),

        "RMSE": pd.to_numeric(
            df[rmse_col],
            errors="raise",
        ),
    })

    result = result[
        result["Budget"].isin(BUDGETS)
    ].copy()

    print("Classical methods:")
    print(sorted(result["Method"].unique()))

    return result


# ============================================================
# 5. LOAD QMLAE RESULTS
# ============================================================

def load_qmlae_results():

    print("\nLOADING QMLAE RESULTS")
    print("-" * 60)

    if not QMLAE_FILE.exists():

        raise FileNotFoundError(
            f"Missing file: {QMLAE_FILE}"
        )

    df = pd.read_csv(QMLAE_FILE)
    df = standardize_columns(df)

    print("QMLAE columns:", list(df.columns))

    metric_col = find_column(
        df,
        ["metric", "risk_metric"],
        "metric",
    )

    budget_col = find_column(
        df,
        ["budget", "shots", "shot_budget"],
        "budget",
    )

    mae_col = find_column(
        df,
        ["mae", "mean_absolute_error"],
        "MAE",
    )

    rmse_col = find_column(
        df,
        ["rmse", "root_mean_squared_error"],
        "RMSE",
    )

    result = pd.DataFrame({

        "Method": "QMLAE",

        "Metric": df[metric_col].apply(
            normalize_metric
        ),

        "Budget": pd.to_numeric(
            df[budget_col],
            errors="raise",
        ),

        "MAE": pd.to_numeric(
            df[mae_col],
            errors="raise",
        ),

        "RMSE": pd.to_numeric(
            df[rmse_col],
            errors="raise",
        ),
    })

    result = result[
        result["Budget"].isin(BUDGETS)
    ].copy()

    return result


# ============================================================
# 6. COMBINE RESULTS
# ============================================================

def combine_results():

    classical = load_classical_results()
    qmlae = load_qmlae_results()

    combined = pd.concat(
        [classical, qmlae],
        ignore_index=True,
    )

    combined["Relative MAE (%)"] = np.nan
    combined["Relative RMSE (%)"] = np.nan

    # Exact reference values are used to calculate
    # relative errors consistently across methods.

    exact_values = (
        combined.groupby("Metric")["MAE"]
        .first()
        .to_dict()
    )

    # Prefer the reference values from the QMLAE
    # benchmark, which independently recalculates them.

    qmlae_df = pd.read_csv(QMLAE_FILE)

    qmlae_df = standardize_columns(qmlae_df)

    exact_col = find_column(
        qmlae_df,
        ["exact_value", "exact"],
        "exact reference value",
    )

    metric_col = find_column(
        qmlae_df,
        ["metric", "risk_metric"],
        "metric",
    )

    reference = {}

    for _, row in qmlae_df.drop_duplicates(
        subset=[metric_col]
    ).iterrows():

        metric = normalize_metric(row[metric_col])

        reference[metric] = float(row[exact_col])

    combined["Exact value"] = combined["Metric"].map(
        reference
    )

    combined["Relative MAE (%)"] = (
        combined["MAE"]
        / combined["Exact value"].abs()
        * 100
    )

    combined["Relative RMSE (%)"] = (
        combined["RMSE"]
        / combined["Exact value"].abs()
        * 100
    )

    combined = combined.sort_values(
        ["Metric", "Budget", "Method"]
    )

    output_file = (
        OUTPUT_DIR / "fair_classical_qmlae_comparison.csv"
    )

    combined.to_csv(
        output_file,
        index=False,
    )

    print("\nCOMBINED DATASET")
    print("-" * 60)

    print(
        combined[
            [
                "Metric",
                "Method",
                "Budget",
                "MAE",
                "RMSE",
                "Relative MAE (%)",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_file)

    return combined


# ============================================================
# 7. GENERATE FIVE MAE GRAPHS
# ============================================================

def plot_mae(combined):

    for metric in METRICS:

        subset = combined[
            combined["Metric"] == metric
        ]

        if subset.empty:
            print("No results for:", metric)
            continue

        plt.figure(figsize=(9, 6))

        for method in sorted(
            subset["Method"].unique()
        ):

            method_data = subset[
                subset["Method"] == method
            ].sort_values("Budget")

            plt.plot(
                method_data["Budget"],
                method_data["Relative MAE (%)"],
                marker="o",
                linewidth=2,
                label=method,
            )

        plt.xscale("log", base=2)
        plt.yscale("log")

        plt.xticks(
            BUDGETS,
            [f"{budget:,}" for budget in BUDGETS],
        )

        plt.xlabel("Nominal samples / measurement shots")
        plt.ylabel("Relative MAE (%)")

        plt.title(
            f"Classical Methods vs QMLAE\n{metric}"
        )

        plt.grid(
            True,
            which="both",
            alpha=0.3,
        )

        plt.legend()
        plt.tight_layout()

        safe_name = (
            metric.lower()
            .replace(" ", "_")
            .replace("$", "")
        )

        output_file = (
            OUTPUT_DIR
            / f"classical_vs_qmlae_{safe_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=250,
        )

        plt.close()

        print("Saved graph:", output_file)


# ============================================================
# 8. GENERATE FIVE RMSE GRAPHS
# ============================================================

def plot_rmse(combined):

    for metric in METRICS:

        subset = combined[
            combined["Metric"] == metric
        ]

        if subset.empty:
            continue

        plt.figure(figsize=(9, 6))

        for method in sorted(
            subset["Method"].unique()
        ):

            method_data = subset[
                subset["Method"] == method
            ].sort_values("Budget")

            plt.plot(
                method_data["Budget"],
                method_data["Relative RMSE (%)"],
                marker="o",
                linewidth=2,
                label=method,
            )

        plt.xscale("log", base=2)
        plt.yscale("log")

        plt.xticks(
            BUDGETS,
            [f"{budget:,}" for budget in BUDGETS],
        )

        plt.xlabel("Nominal samples / measurement shots")
        plt.ylabel("Relative RMSE (%)")

        plt.title(
            f"Classical Methods vs QMLAE — RMSE\n{metric}"
        )

        plt.grid(
            True,
            which="both",
            alpha=0.3,
        )

        plt.legend()
        plt.tight_layout()

        safe_name = (
            metric.lower()
            .replace(" ", "_")
            .replace("$", "")
        )

        output_file = (
            OUTPUT_DIR
            / f"classical_vs_qmlae_rmse_{safe_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=250,
        )

        plt.close()

        print("Saved graph:", output_file)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\nCLASSICAL VS QMLAE COMPARISON")
    print("=" * 65)

    combined = combine_results()

    plot_mae(combined)

    plot_rmse(combined)

    print("\nCOMPARISON COMPLETED")
    print("Five MAE graphs and five RMSE graphs generated.")
    print("Results saved in outputs folder.")


if __name__ == "__main__":
    main()
