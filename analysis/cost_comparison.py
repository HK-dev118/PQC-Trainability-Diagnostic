"""
Task 43: Compare computational cost.

Compares time spent on full training (already recorded in your
training results) vs. an estimate of the quick diagnostic's cost.
Since Hiba's quick_diagnostic.py doesn't currently log its own
runtime, this script times it directly for a representative sample.

Save this as: analysis/cost_comparison.py
"""

import time
import pandas as pd

from diagnostics.quick_diagnostic import run_quick_diagnostic


def time_diagnostic_runs(sample_configs):
    """
    Time how long run_quick_diagnostic takes for a sample of
    circuit configurations, to compare against full training time.
    """
    timings = []

    for _, config in sample_configs.iterrows():
        start = time.perf_counter()

        run_quick_diagnostic(
            n_qubits=int(config["n_qubits"]),
            depth=int(config["depth"]),
            ansatz=config["ansatz"],
            seed=int(config["seed"]),
        )

        elapsed = time.perf_counter() - start
        timings.append({
            "circuit_id": config["circuit_id"],
            "diagnostic_time_sec": elapsed,
        })

    return pd.DataFrame(timings)


if __name__ == "__main__":
    training = pd.read_csv("data/dev_training_results.csv")

    print("Timing the quick diagnostic on all dev-set circuits "
          "(this re-runs Hiba's diagnostic, just to measure time)...")

    # Reconstruct the configs needed to re-run the diagnostic timing.
    sample_configs = training[
        ["circuit_id", "n_qubits", "depth", "ansatz", "seed"]
    ]

    diagnostic_timings = time_diagnostic_runs(sample_configs)

    merged = training.merge(diagnostic_timings, on="circuit_id")

    total_training_time = merged["training_time_sec"].sum()
    total_diagnostic_time = merged["diagnostic_time_sec"].sum()
    speedup = total_training_time / total_diagnostic_time

    print(f"\nTotal full-training time (343 circuits): "
          f"{total_training_time:.2f} sec")
    print(f"Total quick-diagnostic time (343 circuits): "
          f"{total_diagnostic_time:.2f} sec")
    print(f"Speedup factor: {speedup:.1f}x faster")

    print(f"\nAverage per-circuit training time: "
          f"{merged['training_time_sec'].mean():.4f} sec")
    print(f"Average per-circuit diagnostic time: "
          f"{merged['diagnostic_time_sec'].mean():.4f} sec")

    merged[["circuit_id", "training_time_sec", "diagnostic_time_sec"]].to_csv(
        "data/task43_cost_comparison.csv", index=False
    )
    print("\nSaved: data/task43_cost_comparison.csv")
