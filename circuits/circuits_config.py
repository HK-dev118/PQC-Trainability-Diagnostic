import pandas as pd


def create_experiment_configurations(
    qubit_counts=(2, 4, 6, 8, 10, 12, 14),
depths=(1, 2, 3, 4, 6, 8, 10),
    ansatzes=("linear", "ring"),
    seeds=(42, 43, 44, 45, 46)
):
    """
    Create a structured list of PQC experiment configurations.

    Each row represents one circuit instance with a specific
    circuit structure and parameter initialization.

    Returns
    -------
    pandas.DataFrame
        Experiment configuration table.
    """

    configurations = []

    circuit_id = 1

    for n_qubits in qubit_counts:
        for depth in depths:
            for ansatz in ansatzes:
                for seed in seeds:

                    configurations.append({
                        "circuit_id": f"C{circuit_id:04d}",
                        "n_qubits": n_qubits,
                        "depth": depth,
                        "ansatz": ansatz,
                        "seed": seed
                    })

                    circuit_id += 1

    return pd.DataFrame(configurations)


def split_configurations(
    configurations,
    development_fraction=0.70,
    random_state=42
):
    """
    Split circuit configurations into development and test sets.

    The split is performed at the circuit-configuration level so that
    the test configurations remain separate from development.
    """

    shuffled = configurations.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    split_index = int(len(shuffled) * development_fraction)

    development_set = shuffled.iloc[:split_index].reset_index(drop=True)
    test_set = shuffled.iloc[split_index:].reset_index(drop=True)

    return development_set, test_set