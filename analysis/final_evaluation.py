"""
Task 40: Compare predictions with actual training on the test set.
Task 41: Calculate evaluation metrics (accuracy, precision, recall,
F1, confusion matrix). This is the FINAL, official evaluation of
the diagnostic -- uses the frozen model on the untouched test set.

Save this as: analysis/final_evaluation.py
"""

import pandas as pd


def build_confusion_matrix(df):
    """
    Treat "Likely Trainable" as a positive prediction (predicted
    successful) and the other two categories as a negative
    prediction (predicted not successful), matched against the
    real 'successful' outcome.
    """
    df = df.copy()
    df["predicted_successful"] = df["predicted_category"] == "Likely Trainable"

    true_positive = (
        (df["predicted_successful"] == True) & (df["successful"] == True)
    ).sum()
    false_positive = (
        (df["predicted_successful"] == True) & (df["successful"] == False)
    ).sum()
    true_negative = (
        (df["predicted_successful"] == False) & (df["successful"] == False)
    ).sum()
    false_negative = (
        (df["predicted_successful"] == False) & (df["successful"] == True)
    ).sum()

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
    }


def calculate_metrics(confusion):
    tp = confusion["true_positive"]
    fp = confusion["false_positive"]
    tn = confusion["true_negative"]
    fn = confusion["false_negative"]

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/test_scored_and_categorized.csv")

    print("=== Task 40: Prediction vs. actual outcome (per circuit) ===")
    comparison = df[
        ["circuit_id", "predicted_category", "trainability_score",
         "successful", "final_loss"]
    ]
    print(comparison.head(15).to_string(index=False))
    print(f"... ({len(comparison)} total circuits)")
    comparison.to_csv("data/task40_prediction_vs_actual.csv", index=False)

    print("\n=== Task 41: Confusion matrix ===")
    print("(Positive = predicted 'Likely Trainable', matched against "
          "actual training success)")
    confusion = build_confusion_matrix(df)
    print(f"True Positive  (predicted trainable, actually succeeded): "
          f"{confusion['true_positive']}")
    print(f"False Positive (predicted trainable, actually failed):    "
          f"{confusion['false_positive']}")
    print(f"True Negative  (predicted not trainable, actually failed): "
          f"{confusion['true_negative']}")
    print(f"False Negative (predicted not trainable, actually "
          f"succeeded): {confusion['false_negative']}")

    print("\n=== Task 41: Final evaluation metrics (on held-out test set) ===")
    metrics = calculate_metrics(confusion)
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    results = pd.DataFrame([{**confusion, **metrics}])
    results.to_csv("data/task41_final_metrics.csv", index=False)
    print("\nSaved: data/task40_prediction_vs_actual.csv")
    print("Saved: data/task41_final_metrics.csv")
