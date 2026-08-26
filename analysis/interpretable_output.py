"""
Task 45: Check interpretability.

Formats a single circuit's diagnostic result in a clear,
human-readable way -- the kind of output a researcher could
read and immediately understand, without needing to know how
the score was calculated.

Save this as: analysis/interpretable_output.py
"""

import pandas as pd


def explain_prediction(row, indicator_columns, raw_row=None):
    """
    Build a human-readable explanation of one circuit's diagnostic
    result, highlighting which indicators most influenced the score.
    """
    lines = []
    lines.append(f"Circuit: {row['circuit_id']}")
    lines.append(f"Trainability Score: {row['trainability_score']:.2f}")
    lines.append(f"Prediction: {row['predicted_category']}")

    # Identify which indicators are notably high or low (using
    # normalized values: >0.7 = high, <0.3 = low) to name as the
    # "key indicators" driving this prediction.
    high_indicators = [
        col for col in indicator_columns if row[col] > 0.7
    ]
    low_indicators = [
        col for col in indicator_columns if row[col] < 0.3
    ]

    key_points = []
    if "mean_gradient_magnitude" in high_indicators:
        key_points.append("strong average gradients")
    if "mean_gradient_magnitude" in low_indicators:
        key_points.append("weak average gradients")
    if "near_zero_percentage" in low_indicators:
        key_points.append("low percentage of near-zero gradients")
    if "near_zero_percentage" in high_indicators:
        key_points.append("high percentage of near-zero gradients")
    if "n_qubits" in high_indicators:
        key_points.append("high qubit count (harder to train)")
    if "n_qubits" in low_indicators:
        key_points.append("low qubit count (easier to train)")

    if key_points:
        lines.append(f"Key indicators: {' + '.join(key_points)}")
    else:
        lines.append("Key indicators: no strongly dominant factors "
                      "(mixed/moderate values)")

    if raw_row is not None:
        lines.append(f"  (Qubits: {int(raw_row['n_qubits'])}, "
                      f"Depth: {int(raw_row['depth'])}, "
                      f"Parameters: {int(raw_row['n_trainable_parameters'])}, "
                      f"Gates: {int(raw_row['n_gates'])})")

    return "\n".join(lines)


if __name__ == "__main__":
    scored = pd.read_csv("data/test_scored_and_categorized.csv")
    raw = pd.read_csv("data/test_measurement_dataset.csv")

    indicator_columns = [
        "mean_gradient_magnitude", "gradient_variance",
        "near_zero_percentage", "n_qubits", "depth",
        "n_trainable_parameters", "n_gates",
    ]

    print("=== Task 45: Interpretability check ===")
    print("(Sample output for 3 circuits, showing the diagnostic's "
          "full explanation)\n")

    for _, row in scored.head(3).iterrows():
        raw_row = raw[raw["circuit_id"] == row["circuit_id"]].iloc[0]
        print(explain_prediction(row, indicator_columns, raw_row))
        print(f"  Actual outcome: "
              f"{'Trained successfully' if row['successful'] else 'Did not train successfully'}")
        print()
