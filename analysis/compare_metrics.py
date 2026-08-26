"""
Task 35: Compare individual vs. combined approaches (with proper metrics).
Task 36: Check whether circuit complexity adds info beyond gradients.

Builds on task 34's groups but uses precision/recall/F1 instead of
plain accuracy, since accuracy alone can be misleading when classes
are imbalanced (this dataset: 147 successful / 196 failed).

Save this as: analysis/compare_metrics.py
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


def score_from_indicators(row, indicators, weights):
    raw = sum(row[ind] * weights[ind] for ind in indicators)
    return max(0.0, min(1.0, (raw + 1) / 2))


def precision_recall_f1(actual, predicted):
    true_positive = ((predicted == True) & (actual == True)).sum()
    false_positive = ((predicted == True) & (actual == False)).sum()
    false_negative = ((predicted == False) & (actual == True)).sum()

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0 else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0 else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return precision, recall, f1


def evaluate(merged, score_column, threshold=0.5):
    predicted = merged[score_column] >= threshold
    actual = merged["successful"]
    accuracy = (predicted == actual).mean()
    precision, recall, f1 = precision_recall_f1(actual, predicted)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    merged = pd.read_csv("data/dev_merged_indicators_training.csv")
    correlations = pd.read_csv("data/task30_correlations.csv")
    scores = pd.read_csv("data/dev_trainability_scores.csv")

    weights = {
        row["indicator"]: row["correlation_with_success"]
        / correlations["correlation_with_success"].abs().sum()
        for _, row in correlations.iterrows()
    }

    merged["score_gradient_only"] = merged.apply(
        lambda row: score_from_indicators(row, GRADIENT_INDICATORS, weights),
        axis=1,
    )
    merged["score_circuit_only"] = merged.apply(
        lambda row: score_from_indicators(row, CIRCUIT_INDICATORS, weights),
        axis=1,
    )
    merged = merged.merge(
        scores[["circuit_id", "trainability_score"]], on="circuit_id"
    )

    results = []
    for label, column in [
        ("Gradient-only", "score_gradient_only"),
        ("Circuit-only", "score_circuit_only"),
        ("Combined (all 7)", "trainability_score"),
    ]:
        metrics = evaluate(merged, column)
        metrics["approach"] = label
        results.append(metrics)

    results_df = pd.DataFrame(results)[
        ["approach", "accuracy", "precision", "recall", "f1"]
    ]

    print("=== Task 35: Approach comparison (proper metrics) ===")
    print(results_df.to_string(index=False))

    print("\n=== Task 36: Does circuit complexity add beyond gradients? ===")
    gradient_f1 = results_df.loc[
        results_df["approach"] == "Gradient-only", "f1"
    ].values[0]
    combined_f1 = results_df.loc[
        results_df["approach"] == "Combined (all 7)", "f1"
    ].values[0]
    improvement = combined_f1 - gradient_f1

    print(f"Gradient-only F1: {gradient_f1:.3f}")
    print(f"Combined (gradient + circuit) F1: {combined_f1:.3f}")
    print(f"Improvement from adding circuit info: {improvement:+.3f}")

    if improvement > 0.01:
        print("-> Circuit complexity DOES add useful information "
              "beyond gradients alone.")
    elif improvement < -0.01:
        print("-> Adding circuit complexity actually HURT performance "
              "vs. gradients alone.")
    else:
        print("-> Circuit complexity adds negligible information "
              "beyond gradients alone.")

    results_df.to_csv("data/task35_metrics_comparison.csv", index=False)
    print("\nSaved: data/task35_metrics_comparison.csv")
