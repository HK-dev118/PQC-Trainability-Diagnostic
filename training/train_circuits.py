"""
Task 25-29: Fully train the circuits, log progress, and define
what counts as "successful training".

Assumes this file lives in training/train_circuits.py and Hiba's
code lives in circuits/create_pqc.py, circuits/experiment_config.py,
circuits/seeds.py (each importable via circuits/__init__.py).
"""

import time
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp

# --- Hiba's code ---
from circuits.initialisation import create_pqc
from circuits.circuits_config import (
    create_experiment_configurations,
    split_configurations,
)
from circuits.seeds import generate_seeds


# ---------------------------------------------------------------
# Task 29: Define successful training BEFORE looking at results.
# ---------------------------------------------------------------
LOSS_TARGET = 0.05          # "close enough to the minimum"
STABLE_WINDOW = 20          # last N iterations
STABLE_TOLERANCE = 1e-3     # max fluctuation to count as "stable"
MAX_ITERATIONS = 200
LEARNING_RATE = 0.1


def is_successful(loss_history: list[float]) -> bool:
    """
    A run counts as successfully trained if EITHER:
    - the final loss is below LOSS_TARGET, OR
    - the loss has stabilized (barely moving) over the last
      STABLE_WINDOW iterations, even if it plateaued above target.
    This distinguishes "converged but stuck" from "converged and good".
    """
    if not loss_history:
        return False

    final_loss = loss_history[-1]
    if final_loss <= LOSS_TARGET:
        return True

    if len(loss_history) >= STABLE_WINDOW:
        window = loss_history[-STABLE_WINDOW:]
        if (max(window) - min(window)) <= STABLE_TOLERANCE:
            return True

    return False


# ---------------------------------------------------------------
# Task 25-28: Train one circuit/seed combination and log everything.
# ---------------------------------------------------------------
def train_single_run(n_qubits, depth, ansatz, seed):
    circuit, weights = create_pqc(
        n_qubits=n_qubits, depth=depth, seed=seed, ansatz=ansatz
    )

    # Random fixed target vector, one value per qubit, in [-1, 1].
    # Requires Hiba's create_pqc to return a list of per-qubit
    # expvals (qml.expval(qml.PauliZ(i)) for each i), not just
    # qubit 0 -- otherwise depth/width never affect the loss.
    target_rng = pnp.random.default_rng(seed)
    target = target_rng.uniform(low=-1.0, high=1.0, size=n_qubits)

    def cost_fn(v):
        outputs = pnp.array(circuit(v))
        return pnp.mean((outputs - target) ** 2)

    opt = qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)
    w = pnp.array(weights, requires_grad=True)

    loss_history = []
    start_time = time.perf_counter()

    for it in range(MAX_ITERATIONS):
        w, loss = opt.step_and_cost(cost_fn, w)
        loss_history.append(float(loss))

        # Early stop if we've already met the success criterion,
        # no need to burn compute past that point.
        if is_successful(loss_history):
            break

    elapsed = time.perf_counter() - start_time
    final_grad = qml.grad(cost_fn)(w)
    final_grad_norm = float(pnp.linalg.norm(final_grad))

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "ansatz": ansatz,
        "seed": seed,
        "n_iterations": len(loss_history),
        "training_time_sec": elapsed,
        "final_loss": loss_history[-1],
        "final_grad_norm": final_grad_norm,
        "successful": is_successful(loss_history),
        "loss_history": loss_history,  # keep full curve for task 26
    }


# ---------------------------------------------------------------
# Orchestration: run every configuration Hiba generated.
# ---------------------------------------------------------------
def run_all_training(configurations: pd.DataFrame) -> pd.DataFrame:
    results = []
    total = len(configurations)

    for i, row in configurations.iterrows():
        print(f"[{i + 1}/{total}] training {row['circuit_id']} "
              f"(q={row['n_qubits']}, d={row['depth']}, {row['ansatz']})")

        result = train_single_run(
            n_qubits=row["n_qubits"],
            depth=row["depth"],
            ansatz=row["ansatz"],
            seed=row["seed"],
        )
        result["circuit_id"] = row["circuit_id"]
        results.append(result)

    return pd.DataFrame(results)


def filter_hard_subset(configurations: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the hardest region: 10/12/14 qubits with depth 6/8/10.
    Used to sanity-check that harder circuits actually produce
    failures before committing to the full 490-circuit run.
    """
    return configurations[
        configurations["n_qubits"].isin([10, 12, 14])
        & configurations["depth"].isin([6, 8, 10])
    ].reset_index(drop=True)


if __name__ == "__main__":
    import sys

    configs = create_experiment_configurations(
        qubit_counts=(2, 4, 6, 8, 10, 12, 14),
        depths=(1, 2, 3, 4, 6, 8, 10),
    )

    # Run with "hard" as a command-line argument to test just the
    # hard subset first: python -m training.train_circuits hard
    if len(sys.argv) > 1 and sys.argv[1] == "hard":
        hard_set = filter_hard_subset(configs)
        print(f"Training {len(hard_set)} hard-region circuits "
              f"(10/12/14 qubits, depth 6/8/10)...")
        hard_results = run_all_training(hard_set)
        hard_results.to_csv("data/hard_subset_results.csv", index=False)
        print("Done. Hard-subset success rate:",
              hard_results["successful"].mean())

    else:
        dev_set, test_set = split_configurations(configs)

        print(f"Training {len(dev_set)} development-set circuits...")
        dev_results = run_all_training(dev_set)
        dev_results.to_csv("data/dev_training_results.csv", index=False)

        print(f"Training {len(test_set)} held-out test-set circuits...")
        test_results = run_all_training(test_set)
        test_results.to_csv("data/test_training_results.csv", index=False)

        print("Done. Dev-set success rate:",
              dev_results["successful"].mean())
