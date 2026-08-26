"""
Task 42: Analyze incorrect predictions.

Looks closely at the false positives (predicted trainable, actually
failed) and false negatives (predicted difficult, actually succeeded)
to understand patterns in where the diagnostic goes wrong.

Save this as: analysis/analyze_errors.py
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


if __name__ == "__main__":
    df = pd.read_csv("data/test_scored_and_categorized.csv")
    df["predicted_successful"] = df["predicted_category"] == "Likely Trainable"

    false_positives = df[
        (df["predicted_successful"] == True) & (df["successful"] == False)
    ]
    false_negatives = df[
        (df["predicted_successful"] == False) & (df["successful"] == True)
    ]
    correct = df[df["predicted_successful"] == df["successful"]]

    print(f"False positives (predicted trainable, actually failed): "
          f"{len(false_positives)}")
    print(f"False negatives (predicted difficult, actually succeeded): "
          f"{len(false_negatives)}")
    print(f"Correct predictions: {len(correct)}")

    print("\n=== False positives: full detail ===")
    print(false_positives[
        ["circuit_id", "trainability_score", "final_loss"] + INDICATOR_COLUMNS
    ].to_string(index=False))

    print("\n=== False negatives: summary statistics ===")
    print(false_negatives[INDICATOR_COLUMNS + ["trainability_score"]].describe())

    print("\n=== Comparison: false negatives vs. correctly-identified "
          "successes ===")
    correct_successes = df[
        (df["predicted_successful"] == True) & (df["successful"] == True)
    ]

    comparison = pd.DataFrame({
        "false_negatives_mean": false_negatives[INDICATOR_COLUMNS].mean(),
        "correct_successes_mean": correct_successes[INDICATOR_COLUMNS].mean(),
    })
    comparison["difference"] = (
        comparison["false_negatives_mean"] - comparison["correct_successes_mean"]
    )
    print(comparison.to_string())

    print("\n=== False negatives: how close were they to the threshold? ===")
    print("(scores close to 0.462 suggest borderline cases, not "
          "fundamental errors)")
    print(false_negatives[
        ["circuit_id", "trainability_score"]
    ].sort_values("trainability_score", ascending=False).to_string(index=False))

    false_positives.to_csv("data/task42_false_positives.csv", index=False)
    false_negatives.to_csv("data/task42_false_negatives.csv", index=False)
    print("\nSaved: data/task42_false_positives.csv")
    print("Saved: data/task42_false_negatives.csv")
