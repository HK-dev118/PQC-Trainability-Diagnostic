"""
Task 38: Freeze the diagnostic (weights/thresholds locked below,
taken directly from development-set results -- NOT recalculated
here, and never touched again after this point).

Task 39: Test the finished diagnostic on unseen circuits.

Applies the exact frozen weights and thresholds to the held-out
test set for the first time. No calibration happens here.

Save this as: analysis/test_on_holdout.py
"""

import pandas as pd


# --- FROZEN at task 38, taken from dev-set task 30 correlations ---
# Sum of |correlation_with_success| across all 7 indicators = 2.947046
_TOTAL_ABS_CORRELATION = 2.947046

FROZEN_WEIGHTS = {
    "mean_gradient_magnitude": 0.637117 / _TOTAL_ABS_CORRELATION,
    "gradient_variance": 0.544252 / _TOTAL_ABS_CORRELATION,
    "near_zero_percentage": -0.287846 / _TOTAL_ABS_CORRELATION,
    "n_qubits": -0.643604 / _TOTAL_ABS_CORRELATION,
    "depth": 0.043967 / _TOTAL_ABS_CORRELATION,
    "n_trainable_parameters": -0.392090 / _TOTAL_ABS_CORRELATION,
    "n_gates": -0.398170 / _TOTAL_ABS_CORRELATION,
}

FROZEN_LOW_THRESHOLD = 0.370
FROZEN_HIGH_THRESHOLD = 0.462


def calculate_trainability_score(row, weights):
    raw_score = sum(row[indicator] * weight
                     for indicator, weight in weights.items())
    score = (raw_score + 1) / 2
    return max(0.0, min(1.0, score))


def score_to_category(score):
    if score >= FROZEN_HIGH_THRESHOLD:
        return "Likely Trainable"
    elif score >= FROZEN_LOW_THRESHOLD:
        return "Possibly Difficult"
    else:
        return "Likely Difficult to Train"


if __name__ == "__main__":
    indicators = pd.read_csv("data/test_measurement_dataset_normalized.csv")
    training = pd.read_csv("data/test_training_results.csv")

    merged = indicators.merge(
        training[["circuit_id", "final_loss", "successful"]],
        on="circuit_id",
        how="inner",
    )

    print(f"Test-set circuits: {len(merged)}")

    merged["trainability_score"] = merged.apply(
        lambda row: calculate_trainability_score(row, FROZEN_WEIGHTS),
        axis=1,
    )
    merged["predicted_category"] = merged["trainability_score"].apply(
        score_to_category
    )

    print("\nCategory counts on test set:")
    print(merged["predicted_category"].value_counts())

    print("\nActual success rate WITHIN each predicted category:")
    print(merged.groupby("predicted_category")["successful"].mean())

    merged.to_csv("data/test_scored_and_categorized.csv", index=False)
    print("\nSaved: data/test_scored_and_categorized.csv")
    print("\nThis is your task 40 input -- diagnostic prediction vs. "
          "actual training outcome, per circuit.")
