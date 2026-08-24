import pandas as pd

from circuits.circuits_config import create_experiment_configurations
from diagnostics.quick_diagnostic import run_quick_diagnostic


MEASUREMENT_COLUMNS = [
    "circuit_id",
    "n_qubits",
    "depth",
    "ansatz",
    "seed",
    "mean_gradient_magnitude",
    "gradient_variance",
    "near_zero_percentage",
    "n_trainable_parameters",
    "n_gates"
]


def collect_measurement(config):
    """
    Run the quick diagnostic for one experiment configuration
    and return the measured indicators as a single dictionary.
    """

    result = run_quick_diagnostic(
        n_qubits=int(config["n_qubits"]),
        depth=int(config["depth"]),
        ansatz=config["ansatz"],
        seed=int(config["seed"])
    )

    return {
        "circuit_id": config["circuit_id"],
        "n_qubits": result["n_qubits"],
        "depth": result["depth"],
        "ansatz": result["ansatz"],
        "seed": result["seed"],
        "mean_gradient_magnitude": float(
            result["mean_gradient_magnitude"]
        ),
        "gradient_variance": float(
            result["gradient_variance"]
        ),
        "near_zero_percentage": float(
            result["near_zero_percentage"]
        ),
        "n_trainable_parameters": int(
            result["n_trainable_parameters"]
        ),
        "n_gates": int(
            result["n_gates"]
        )
    }


def create_measurement_dataset(
    configurations,
    max_circuits=None
):
    """
    Run the quick diagnostic for a collection of circuit
    configurations and create a structured measurement dataset.
    """

    if max_circuits is not None:
        configurations = configurations.head(max_circuits)

    measurements = []

    total = len(configurations)

    for index, (_, config) in enumerate(
        configurations.iterrows(),
        start=1
    ):

        print(
            f"Running diagnostic "
            f"{index}/{total}: "
            f"{config['circuit_id']}"
        )

        measurement = collect_measurement(config)

        measurements.append(measurement)

    return pd.DataFrame(
        measurements,
        columns=MEASUREMENT_COLUMNS
    )


def check_measurement_consistency(
    config,
    tolerance=1e-12
):
    """
    Run the same circuit configuration twice and check
    whether the diagnostic measurements are consistent.

    Parameters
    ----------
    config : pandas.Series or dict
        Circuit configuration to test.

    tolerance : float
        Maximum allowed numerical difference between
        repeated measurements.

    Returns
    -------
    bool
        True if the measurements are consistent.
    """

    first_measurement = collect_measurement(config)
    second_measurement = collect_measurement(config)

    numeric_columns = [
        "mean_gradient_magnitude",
        "gradient_variance",
        "near_zero_percentage",
        "n_trainable_parameters",
        "n_gates"
    ]

    for column in numeric_columns:

        difference = abs(
            first_measurement[column]
            - second_measurement[column]
        )

        if difference > tolerance:
            return False

    return True


def save_measurement_dataset(
    measurement_data,
    output_path
):
    """
    Save the measurement dataset as a CSV file.
    """

    measurement_data.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    print("Creating experiment configurations...")

    configurations = create_experiment_configurations()

    print(
        f"Total configurations: "
        f"{len(configurations)}"
    )

    # ---------------------------------------------------------
    # Task 18: Measurement consistency check
    # ---------------------------------------------------------

    test_configuration = configurations.iloc[0]

    print()
    print("Checking measurement consistency...")
    print(
        f"Configuration: "
        f"{test_configuration['circuit_id']}"
    )

    consistent = check_measurement_consistency(
        test_configuration
    )

    if consistent:
        print(
            "Measurement consistency check: PASSED"
        )
    else:
        print(
            "Measurement consistency check: FAILED"
        )

    # ---------------------------------------------------------
    # Task 17: Small dataset pipeline test
    # ---------------------------------------------------------

    print()
    print("Creating small measurement dataset...")

    measurement_data = create_measurement_dataset(
        configurations,
        max_circuits=3
    )

    print()
    print(
        "Measurement dataset created successfully."
    )
    print()
    print(measurement_data)