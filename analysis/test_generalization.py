"""
Task 44: Test generalization.

Your original grid only used EVEN qubit counts (2,4,6,8,10,12,14).
This generates a small set of circuits with ODD qubit counts
(3,5,7,9,11,13) -- a structure never seen during development --
trains them for real, and checks whether the frozen diagnostic
still predicts sensibly on this genuinely novel structure.

Save this as: analysis/test_generalization.py
"""

import pandas as pd

from circuits.circuits_config import create_experiment_configurations
from diagnostics.measurement_dataset import create_measurement_dataset
from diagnostics.normalization import apply_min_max_normalization
from training.train_circuits import train_single_run

# Re-use the exact frozen weights/thresholds from task 38/39
from analysis.test_on_holdout import (
    FROZEN_WEIGHTS,
    calculate_trainability_score,
    score_to_category,
)


def get_dev_normalization_params():
    """
    Re-derive the min/max normalization parameters that were fit
    on the development set, so the new (odd-qubit) circuits are
    normalized the SAME way -- never re-fit normalization on new
    data, that would defeat the point of a frozen diagnostic.
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


if __name__ == "__main__":
    print("Generating odd-qubit-count circuits (novel structure, "
          "never seen in development)...")

    odd_configs = create_experiment_configurations(
        qubit_counts=(3, 5, 7, 9, 11, 13),
        depths=(2, 4, 6),
        ansatzes=("linear", "ring"),
        seeds=(42, 43),
    )

    print(f"Generated {len(odd_configs)} novel-structure circuits")

    print("\nRunning quick diagnostic on novel circuits...")
    novel_measurements = create_measurement_dataset(odd_configs)

    print("\nNormalizing using DEV-SET parameters (frozen, not re-fit)...")
    norm_params = get_dev_normalization_params()
    novel_normalized = apply_min_max_normalization(
        novel_measurements, norm_params
    )

    print("\nFully training the novel circuits (real ground truth)...")
    training_results = []
    for _, config in odd_configs.iterrows():
        result = train_single_run(
            n_qubits=int(config["n_qubits"]),
            depth=int(config["depth"]),
            ansatz=config["ansatz"],
            seed=int(config["seed"]),
        )
        result["circuit_id"] = config["circuit_id"]
        training_results.append(result)

    training_df = pd.DataFrame(training_results)

    print("\nApplying the FROZEN diagnostic to novel circuits...")
    novel_normalized["trainability_score"] = novel_normalized.apply(
        lambda row: calculate_trainability_score(row, FROZEN_WEIGHTS),
        axis=1,
    )
    novel_normalized["predicted_category"] = novel_normalized[
        "trainability_score"
    ].apply(score_to_category)

    merged = novel_normalized.merge(
        training_df[["circuit_id", "successful", "final_loss"]],
        on="circuit_id",
    )

    print(f"\nNovel-structure test set: {len(merged)} circuits "
          f"(odd qubit counts: 3,5,7,9,11,13)")
    print("\nActual success rate WITHIN each predicted category:")
    print(merged.groupby("predicted_category")["successful"].mean())

    accuracy = (
        (merged["predicted_category"] == "Likely Trainable")
        == merged["successful"]
    ).mean()
    print(f"\nRough accuracy on novel structure: {accuracy:.3f}")
    print("(Compare to test-set accuracy of 0.830 on familiar "
          "even-qubit structures -- a similar number here would "
          "support genuine generalization.)")

    merged.to_csv("data/task44_generalization_test.csv", index=False)
    print("\nSaved: data/task44_generalization_test.csv")
