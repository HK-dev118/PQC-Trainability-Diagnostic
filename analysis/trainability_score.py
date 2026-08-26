"""
Task 31: Develop the Trainability Score.

Combines the 7 indicators into a single 0-1 score, using weights
derived from task 30's correlation-with-success results (not
arbitrarily chosen). Indicators that correlate positively with
success add to the score; indicators that correlate negatively
subtract from it. Weight magnitude = strength of that correlation.

Only uses the DEVELOPMENT set to derive weights -- never the test set.

Save this as: analysis/trainability_score.py
"""

import pandas as pd


INDICATOR_COLUMNS = [
    "mean_gradient_magnitude",
    "gradient_variance",
    "near_zero_percentage",
    "n_qubits",
    "depth",
    "n_trainable_parameters",
    "n_gates",
]


def derive_weights_from_correlations(correlations_df):
    """
    Turn task 30's correlation-with-success numbers into signed
    weights: positive correlation -> positive weight (raises score),
    negative correlation -> negative weight (lowers score).
    Weights are normalized so their absolute values sum to 1,
    making the final score interpretable and bounded.
    """
    weights = {}
    total_abs = correlations_df["correlation_with_success"].abs().sum()

    for _, row in correlations_df.iterrows():
        indicator = row["indicator"]
        raw_corr = row["correlation_with_success"]
        weights[indicator] = raw_corr / total_abs

    return weights


def calculate_trainability_score(row, weights):
    """
    Calculate a single circuit's Trainability Score (0-1) from its
    normalized indicator values and the derived weights.

    Since indicators are already min-max normalized to [0, 1],
    and weights are signed and sum (in absolute value) to 1,
    the raw weighted sum can go negative (if failure-indicators
    dominate) or exceed 1 is not possible by construction -- but
    we still clip defensively and rescale to keep it interpretable.
    """
    raw_score = sum(
        row[indicator] * weight
        for indicator, weight in weights.items()
    )

    # Weights are signed and centered around 0, so raw_score can
    # range roughly [-1, 1]. Rescale to [0, 1] for interpretability.
    score = (raw_score + 1) / 2

    return max(0.0, min(1.0, score))


if __name__ == "__main__":
    merged = pd.read_csv("data/dev_merged_indicators_training.csv")
    correlations = pd.read_csv("data/task30_correlations.csv")

    weights = derive_weights_from_correlations(correlations)

    print("Derived weights (from task 30 correlations):")
    for indicator, weight in weights.items():
        direction = "raises score" if weight > 0 else "lowers score"
        print(f"  {indicator}: {weight:.3f} ({direction})")

    merged["trainability_score"] = merged.apply(
        lambda row: calculate_trainability_score(row, weights),
        axis=1,
    )

    print("\nScore distribution:")
    print(merged["trainability_score"].describe())

    print("\nAverage score, successful vs failed:")
    print(merged.groupby("successful")["trainability_score"].mean())

    output_columns = ["circuit_id", "trainability_score", "successful",
                       "final_loss"] + INDICATOR_COLUMNS
    merged[output_columns].to_csv(
        "data/dev_trainability_scores.csv", index=False
    )

    print("\nSaved: data/dev_trainability_scores.csv")
