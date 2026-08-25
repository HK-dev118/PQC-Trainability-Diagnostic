import os
import pandas as pd

from circuits.circuits_config import (
    create_experiment_configurations,
    split_configurations
)

from diagnostics.quick_diagnostic import (
    run_quick_diagnostic
)

from diagnostics.normalization import (
    fit_normalization_parameters,
    apply_min_max_normalization
)


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


INDICATOR_COLUMNS = [
    "mean_gradient_magnitude",
    "gradient_variance",
    "near_zero_percentage",
    "n_qubits",
    "depth",
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


def create_measurement_dataset(configurations):
    """
    Run the quick diagnostic for all supplied circuit
    configurations and create a measurement dataset.
    """

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
    Save a measurement dataset as a CSV file.
    """

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    measurement_data.to_csv(
        output_path,
        index=False
    )


if __name__ == "__main__":

    # =========================================================
    # 1. Create all experiment configurations
    # =========================================================

    print("Creating experiment configurations...")

    configurations = create_experiment_configurations()

    print(
        f"Total configurations: "
        f"{len(configurations)}"
    )

    # =========================================================
    # 2. Check measurement consistency
    # =========================================================

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

    if not consistent:
        print(
            "Measurement consistency check: FAILED"
        )
        raise RuntimeError(
            "Measurement consistency check failed."
        )

    print(
        "Measurement consistency check: PASSED"
    )

    # =========================================================
    # 3. Run diagnostic on ALL 490 circuits
    # =========================================================

    print()
    print(
        "Creating full measurement dataset..."
    )

    measurement_data = create_measurement_dataset(
        configurations
    )

    print()
    print(
        "Full measurement dataset created successfully."
    )

    print(
        f"Number of measurements: "
        f"{len(measurement_data)}"
    )

    # =========================================================
    # 4. Save raw measurements
    # =========================================================

    raw_output_path = (
        "diagnostics/full_measurement_dataset.csv"
    )

    save_measurement_dataset(
        measurement_data,
        raw_output_path
    )

    print()
    print(
        f"Saved raw measurement dataset to: "
        f"{raw_output_path}"
    )

    # =========================================================
    # 5. Create the same 70/30 development/test split
    # =========================================================

    (
        development_configurations,
        test_configurations
    ) = split_configurations(
        configurations,
        development_fraction=0.70,
        random_state=42
    )

    # =========================================================
    # 6. Match configurations to measurement rows
    # =========================================================

    development_measurements = (
        measurement_data[
            measurement_data["circuit_id"].isin(
                development_configurations["circuit_id"]
            )
        ]
        .reset_index(drop=True)
    )

    test_measurements = (
        measurement_data[
            measurement_data["circuit_id"].isin(
                test_configurations["circuit_id"]
            )
        ]
        .reset_index(drop=True)
    )

    print()
    print(
        f"Development measurements: "
        f"{len(development_measurements)}"
    )

    print(
        f"Test measurements: "
        f"{len(test_measurements)}"
    )

    # =========================================================
    # 7. Fit normalization ONLY on development data
    # =========================================================

    print()
    print(
        "Fitting normalization parameters "
        "using development data only..."
    )

    normalization_parameters = (
        fit_normalization_parameters(
            development_measurements,
            INDICATOR_COLUMNS
        )
    )

    # =========================================================
    # 8. Normalize all measurements using development parameters
    # =========================================================

    normalized_measurement_data = (
        apply_min_max_normalization(
            measurement_data,
            normalization_parameters
        )
    )

    # =========================================================
    # 9. Save normalized dataset
    # =========================================================

    normalized_output_path = (
        "diagnostics/"
        "full_measurement_dataset_normalized.csv"
    )

    save_measurement_dataset(
        normalized_measurement_data,
        normalized_output_path
    )

    print()
    print(
        "Normalized measurement dataset "
        "created successfully."
    )

    print(
        f"Saved normalized dataset to: "
        f"{normalized_output_path}"
    )

    # =========================================================
    # 10. Save development/test normalized datasets
    # =========================================================

    development_normalized = (
        normalized_measurement_data[
            normalized_measurement_data["circuit_id"].isin(
                development_configurations["circuit_id"]
            )
        ]
        .reset_index(drop=True)
    )

    test_normalized = (
        normalized_measurement_data[
            normalized_measurement_data["circuit_id"].isin(
                test_configurations["circuit_id"]
            )
        ]
        .reset_index(drop=True)
    )

    development_output_path = (
        "diagnostics/"
        "development_measurement_dataset_normalized.csv"
    )

    test_output_path = (
        "diagnostics/"
        "test_measurement_dataset_normalized.csv"
    )

    save_measurement_dataset(
        development_normalized,
        development_output_path
    )

    save_measurement_dataset(
        test_normalized,
        test_output_path
    )

    print()
    print(
        f"Saved normalized development dataset to: "
        f"{development_output_path}"
    )

    print(
        f"Saved normalized test dataset to: "
        f"{test_output_path}"
    )

    # =========================================================
    # 11. Final verification
    # =========================================================

    print()
    print("Final dataset verification:")
    print(
        f"Raw rows: "
        f"{len(measurement_data)}"
    )

    print(
        f"Normalized rows: "
        f"{len(normalized_measurement_data)}"
    )

    print(
        f"Expected rows: "
        f"{len(configurations)}"
    )

    if (
        len(measurement_data)
        == len(configurations)
        == len(normalized_measurement_data)
    ):
        print(
            "Verification PASSED: "
            "all 490 configurations are present."
        )
    else:
        raise RuntimeError(
            "Verification FAILED: "
            "dataset row count does not match "
            "the number of configurations."
        )

    print()
    print("Measurement pipeline completed successfully.")