"""
Task 34: Test indicators separately.

Compares 4 approaches at predicting training success:
1. Individual indicators (each one alone)
2. Gradient-only (mean_gradient_magnitude, gradient_variance, near_zero_percentage)
3. Circuit-only (n_qubits, depth, n_trainable_parameters, n_gates)
4. Combined (your full Trainability Score from task 31)

Uses a simple, consistent scoring method for all groups: sum of
normalized indicator values (sign-adjusted so higher = more
trainable), so the comparison is fair across groups.

Save this as: analysis/compare_approaches.py
"""

import pandas as pd


GRADIENT_INDICATORS = [
    "mean_gradient_magnitude",
    "gradient_variance",
    "near_zero_percentage",
]

CIRCUIT_INDICATORS = [
    "n_qubits",
    "depth",
    "n_trainable_parameters",
    "n_gates",
]

ALL_INDICATORS = GRADIENT_INDICATORS + CIRCUIT_INDICATORS


def score_from_indicators(row, indicators, weights):
    """
    Weighted sum of the given indicators, using the same signed
    weights derived in task 31 (positive = raises score).
    """
    raw = sum(row[ind] * weights[ind] for ind in indicators)
    # Rescale using the same convention as task 31.
    return max(0.0, min(1.0, (raw + 1) / 2))


def evaluate_individual_indicator(merged, indicator, weight):
    """
    Fairly evaluate a SINGLE indicator's predictive power by using
    a threshold on that indicator's own value (its median), rather
    than reusing the small combined-score weight -- which would
    barely move the score away from 0.5 and produce a near-constant
    prediction for every circuit.

    Direction (does higher value mean more trainable?) comes from
    the sign of its task-30 correlation weight.
    """
    median_value = merged[indicator].median()

    if weight >= 0:
        # Higher indicator value -> predict success
        predicted_success = merged[indicator] >= median_value
    else:
        # Higher indicator value -> predict failure
        predicted_success = merged[indicator] < median_value

    correct = (predicted_success == merged["successful"]).mean()
    return correct


def evaluate_approach(merged, score_column, threshold=0.5):
    """
    Simple accuracy: does score >= threshold correctly predict
    'successful'? This is a rough evaluation for comparison
    purposes -- the real evaluation with proper metrics happens
    in tasks 40-41 on the held-out test set.
    """
    predicted_success = merged[score_column] >= threshold
    correct = (predicted_success == merged["successful"]).mean()
    return correct


if __name__ == "__main__":
    merged = pd.read_csv("data/dev_merged_indicators_training.csv")
    correlations = pd.read_csv("data/task30_correlations.csv")

    weights = {
        row["indicator"]: row["correlation_with_success"]
        / correlations["correlation_with_success"].abs().sum()
        for _, row in correlations.iterrows()
    }

    print("=== Individual indicators ===")
    individual_results = []
    for indicator in ALL_INDICATORS:
        accuracy = evaluate_individual_indicator(
            merged, indicator, weights[indicator]
        )
        individual_results.append({"indicator": indicator, "accuracy": accuracy})
        print(f"  {indicator}: {accuracy:.3f}")

    print("\n=== Gradient-only (combined) ===")
    merged["score_gradient_only"] = merged.apply(
        lambda row: score_from_indicators(row, GRADIENT_INDICATORS, weights),
        axis=1,
    )
    gradient_accuracy = evaluate_approach(merged, "score_gradient_only")
    print(f"  accuracy: {gradient_accuracy:.3f}")

    print("\n=== Circuit-only (combined) ===")
    merged["score_circuit_only"] = merged.apply(
        lambda row: score_from_indicators(row, CIRCUIT_INDICATORS, weights),
        axis=1,
    )
    circuit_accuracy = evaluate_approach(merged, "score_circuit_only")
    print(f"  accuracy: {circuit_accuracy:.3f}")

    print("\n=== Combined (all 7 indicators, task 31's full score) ===")
    scores = pd.read_csv("data/dev_trainability_scores.csv")
    merged_full = merged.merge(
        scores[["circuit_id", "trainability_score"]], on="circuit_id"
    )
    combined_accuracy = evaluate_approach(
        merged_full, "trainability_score"
    )
    print(f"  accuracy: {combined_accuracy:.3f}")

    print("\n=== Summary ===")
    summary = pd.DataFrame(individual_results)
    summary = pd.concat([summary, pd.DataFrame([
        {"indicator": "gradient_only_combined", "accuracy": gradient_accuracy},
        {"indicator": "circuit_only_combined", "accuracy": circuit_accuracy},
        {"indicator": "all_combined (task 31 score)", "accuracy": combined_accuracy},
    ])], ignore_index=True)
    summary = summary.sort_values("accuracy", ascending=False)
    print(summary.to_string(index=False))

    summary.to_csv("data/task34_approach_comparison.csv", index=False)
    print("\nSaved: data/task34_approach_comparison.csv")
