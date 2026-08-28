"""
Task 46: Connect the scoring system to the tool.
Task 47: Connect the prediction to the tool.

This is the final, end-to-end diagnostic function: give it a
circuit specification (qubits, depth, ansatz, seed), and it runs
Hiba's quick diagnostic, normalizes using the FROZEN dev-set
parameters, calculates the Trainability Score, and returns the
category -- all in one call. This is the actual deliverable tool
described in Hiba's tasks 20-23.

Save this as: analysis/diagnose_circuit.py
"""

import pandas as pd

from diagnostics.quick_diagnostic import run_quick_diagnostic
from diagnostics.normalization import apply_min_max_normalization
from analysis.test_on_holdout import (
    FROZEN_WEIGHTS,
    calculate_trainability_score,
    score_to_category,
)


def get_frozen_normalization_params():
    """
    The exact min/max normalization parameters fit on the
    development set. Frozen at task 38 -- never re-fit.
    """
    dev_raw = pd.read_csv("data/dev_measurement_dataset.csv")
    columns = [
        "mean_gradient_magnitude", "gradient_variance",
        "near_zero_percentage", "n_qubits", "depth",
        "n_trainable_parameters", "n_gates",
    ]
    params = {}
    for col in columns:
        params[col] = {
            "min": dev_raw[col].min(),
            "max": dev_raw[col].max(),
        }
    return params


def diagnose_circuit(n_qubits, depth, ansatz="ring", seed=42):
    """
    Full end-to-end diagnostic: run the quick simulation, calculate
    indicators, normalize, score, and categorize -- for a single
    user-provided circuit. This is the main entry point for the tool.

    Returns
    -------
    dict with the score, category, raw measurements, and a plain
    human-readable explanation.
    """
    result = run_quick_diagnostic(
        n_qubits=n_qubits, depth=depth, ansatz=ansatz, seed=seed
    )

    measurement_row = pd.DataFrame([{
        "mean_gradient_magnitude": float(result["mean_gradient_magnitude"]),
        "gradient_variance": float(result["gradient_variance"]),
        "near_zero_percentage": float(result["near_zero_percentage"]),
        "n_qubits": result["n_qubits"],
        "depth": result["depth"],
        "n_trainable_parameters": int(result["n_trainable_parameters"]),
        "n_gates": int(result["n_gates"]),
    }])

    norm_params = get_frozen_normalization_params()
    normalized = apply_min_max_normalization(measurement_row, norm_params)
    normalized_row = normalized.iloc[0]

    score = calculate_trainability_score(normalized_row, FROZEN_WEIGHTS)
    category = score_to_category(score)

    high = [c for c in norm_params if normalized_row[c] > 0.7]
    low = [c for c in norm_params if normalized_row[c] < 0.3]
    key_points = []
    if "mean_gradient_magnitude" in high:
        key_points.append("strong average gradients")
    if "mean_gradient_magnitude" in low:
        key_points.append("weak average gradients")
    if "near_zero_percentage" in low:
        key_points.append("low percentage of near-zero gradients")
    if "near_zero_percentage" in high:
        key_points.append("high percentage of near-zero gradients")
    if "n_qubits" in high:
        key_points.append("high qubit count (harder to train)")
    if "n_qubits" in low:
        key_points.append("low qubit count (easier to train)")

    return {
        "trainability_score": round(score, 2),
        "prediction": category,
        "key_indicators": " + ".join(key_points) if key_points else
                           "no strongly dominant factors",
        "mean_gradient_magnitude": round(
            float(result["mean_gradient_magnitude"]), 4
        ),
        "gradient_variance": round(float(result["gradient_variance"]), 4),
        "near_zero_percentage": round(
            float(result["near_zero_percentage"]), 2
        ),
        "n_qubits": result["n_qubits"],
        "depth": result["depth"],
        "n_trainable_parameters": int(result["n_trainable_parameters"]),
        "n_gates": int(result["n_gates"]),
    }


def print_diagnosis(diagnosis):
    """
    Task 23 / 45 style readable printout.
    """
    print(f"Trainability Score: {diagnosis['trainability_score']}")
    print(f"Prediction: {diagnosis['prediction']}")
    print(f"Key indicators: {diagnosis['key_indicators']}")
    print(f"  Mean gradient magnitude: "
          f"{diagnosis['mean_gradient_magnitude']}")
    print(f"  Gradient variance: {diagnosis['gradient_variance']}")
    print(f"  Near-zero gradients: {diagnosis['near_zero_percentage']}%")
    print(f"  Qubits: {diagnosis['n_qubits']}")
    print(f"  Depth: {diagnosis['depth']}")
    print(f"  Parameters: {diagnosis['n_trainable_parameters']}")
    print(f"  Gates: {diagnosis['n_gates']}")


if __name__ == "__main__":
    print("=== Diagnostic tool demo ===\n")

    print("Example 1: small, shallow circuit")
    diagnosis = diagnose_circuit(n_qubits=2, depth=1, ansatz="ring", seed=42)
    print_diagnosis(diagnosis)

    print("\nExample 2: large, deep circuit")
    diagnosis = diagnose_circuit(n_qubits=14, depth=10, ansatz="ring", seed=42)
    print_diagnosis(diagnosis)

    print("\nExample 3: a circuit a user might provide (task 21)")
    diagnosis = diagnose_circuit(n_qubits=8, depth=4, ansatz="linear", seed=100)
    print_diagnosis(diagnosis)
