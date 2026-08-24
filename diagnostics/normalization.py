import pandas as pd


def fit_normalization_parameters(
    development_data,
    indicator_columns
):
    """
    Calculate normalization parameters using only the
    development dataset.

    Parameters
    ----------
    development_data : pandas.DataFrame
        Development-set measurements.

    indicator_columns : list[str]
        Columns that should be normalized.

    Returns
    -------
    dict
        Minimum and maximum values for each indicator.
    """

    normalization_parameters = {}

    for column in indicator_columns:

        minimum = development_data[column].min()
        maximum = development_data[column].max()

        normalization_parameters[column] = {
            "min": minimum,
            "max": maximum
        }

    return normalization_parameters


def apply_min_max_normalization(
    data,
    normalization_parameters
):
    """
    Apply min-max normalization using parameters
    previously calculated from the development set.

    The normalized range is [0, 1].
    """

    normalized_data = data.copy()

    for column, parameters in normalization_parameters.items():

        minimum = parameters["min"]
        maximum = parameters["max"]

        if maximum == minimum:
            normalized_data[column] = 0.0

        else:
            normalized_data[column] = (
                data[column] - minimum
            ) / (
                maximum - minimum
            )

    return normalized_data


def fit_and_normalize_development_data(
    development_data,
    indicator_columns
):
    """
    Fit normalization parameters on the development data
    and return the normalized development data.
    """

    parameters = fit_normalization_parameters(
        development_data,
        indicator_columns
    )

    normalized_data = apply_min_max_normalization(
        development_data,
        parameters
    )

    return normalized_data, parameters


if __name__ == "__main__":

    # Small demonstration dataset for testing.
    example_data = pd.DataFrame({
        "mean_gradient_magnitude": [0.10, 0.20, 0.30],
        "gradient_variance": [0.01, 0.05, 0.10],
        "near_zero_percentage": [5.0, 20.0, 40.0],
        "n_qubits": [2, 4, 8],
        "depth": [1, 2, 4],
        "n_trainable_parameters": [4, 16, 64],
        "n_gates": [6, 12, 24]
    })

    indicator_columns = [
        "mean_gradient_magnitude",
        "gradient_variance",
        "near_zero_percentage",
        "n_qubits",
        "depth",
        "n_trainable_parameters",
        "n_gates"
    ]

    normalized_data, parameters = (
        fit_and_normalize_development_data(
            example_data,
            indicator_columns
        )
    )

    print("Normalization completed successfully.")
    print()
    print("Original data:")
    print(example_data)
    print()
    print("Normalized data:")
    print(normalized_data)
    print()
    print("Normalization parameters:")
    print(parameters)