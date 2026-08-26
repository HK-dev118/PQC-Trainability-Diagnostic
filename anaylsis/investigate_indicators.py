"""
Task 30: Investigate relationships between indicators and training.

Merges Hiba's development-set indicator measurements with your own
training results (final_loss, successful), and checks whether each
indicator actually relates to training difficulty.

IMPORTANT: Uses the DEVELOPMENT set only. Never touch the test set
here -- that's reserved for task 39.

Save this as: analysis/investigate_indicators.py
"""

import pandas as pd


def load_and_merge():
    indicators = pd.read_csv("data/dev_measurement_dataset_normalized.csv")
    training = pd.read_csv("data/dev_training_results.csv")

    merged = indicators.merge(
        training[["circuit_id", "final_loss", "successful",
                  "n_iterations", "final_grad_norm"]],
        on="circuit_id",
        how="inner",
    )

    print(f"Indicator rows: {len(indicators)}")
    print(f"Training rows: {len(training)}")
    print(f"Merged rows (should match, on circuit_id): {len(merged)}")

    if len(merged) != len(indicators):
        print("WARNING: some circuit_ids didn't match between "
              "the two files -- check for typos or mismatched runs.")

    return merged


def compare_successful_vs_failed(merged, indicator_columns):
    """
    For each indicator, compare its average value between circuits
    that trained successfully vs. those that didn't.
    A meaningful gap suggests the indicator carries predictive signal.
    """
    successful = merged[merged["successful"] == True]
    failed = merged[merged["successful"] == False]

    print(f"\nSuccessful circuits: {len(successful)}")
    print(f"Failed circuits: {len(failed)}")
    print()

    summary = []
    for col in indicator_columns:
        succ_mean = successful[col].mean()
        fail_mean = failed[col].mean()
        summary.append({
            "indicator": col,
            "mean_if_successful": succ_mean,
            "mean_if_failed": fail_mean,
            "difference": succ_mean - fail_mean,
        })

    summary_df = pd.DataFrame(summary).sort_values(
        "difference", key=abs, ascending=False
    )
    return summary_df


def correlation_with_outcome(merged, indicator_columns):
    """
    Correlation between each indicator and final_loss (lower loss
    = easier training). Also correlate with the successful flag
    (converted to 0/1) for a second view.
    """
    merged = merged.copy()
    merged["successful_numeric"] = merged["successful"].astype(int)

    correlations = []
    for col in indicator_columns:
        corr_loss = merged[col].corr(merged["final_loss"])
        corr_success = merged[col].corr(merged["successful_numeric"])
        correlations.append({
            "indicator": col,
            "correlation_with_final_loss": corr_loss,
            "correlation_with_success": corr_success,
        })

    return pd.DataFrame(correlations).sort_values(
        "correlation_with_success", key=abs, ascending=False
    )


if __name__ == "__main__":
    merged = load_and_merge()
    merged.to_csv("data/dev_merged_indicators_training.csv", index=False)

    indicator_columns = [
        "mean_gradient_magnitude",
        "gradient_variance",
        "near_zero_percentage",
        "n_qubits",
        "depth",
        "n_trainable_parameters",
        "n_gates",
    ]

    print("\n=== Successful vs. failed comparison ===")
    comparison = compare_successful_vs_failed(merged, indicator_columns)
    print(comparison.to_string(index=False))

    print("\n=== Correlation with training outcome ===")
    correlations = correlation_with_outcome(merged, indicator_columns)
    print(correlations.to_string(index=False))

    comparison.to_csv("data/task30_success_vs_failed.csv", index=False)
    correlations.to_csv("data/task30_correlations.csv", index=False)

    print("\nSaved: data/dev_merged_indicators_training.csv")
    print("Saved: data/task30_success_vs_failed.csv")
    print("Saved: data/task30_correlations.csv")
