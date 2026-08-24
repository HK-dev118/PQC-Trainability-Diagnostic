import pennylane as qml


def calculate_mean_gradient_magnitude(gradients):
    """
    Calculate the mean absolute gradient magnitude.

    Parameters
    ----------
    gradients : array-like
        Gradient values for the circuit outputs.

    Returns
    -------
    float
        Mean absolute gradient magnitude.
    """

    absolute_gradients = qml.numpy.abs(gradients)

    return qml.numpy.mean(absolute_gradients)


def calculate_gradient_variance(gradients):
    """
    Calculate the variance of the gradient values.

    Parameters
    ----------
    gradients : array-like
        Gradient values for the circuit outputs.

    Returns
    -------
    float
        Gradient variance.
    """

    return qml.numpy.var(gradients)


def calculate_near_zero_percentage(
    gradients,
    threshold=0.001
):
    """
    Calculate the percentage of gradients whose absolute
    value is below the specified threshold.

    Parameters
    ----------
    gradients : array-like
        Gradient values for the circuit outputs.

    threshold : float
        Absolute-gradient threshold used to classify
        gradients as near-zero.

    Returns
    -------
    float
        Percentage of near-zero gradients.
    """

    absolute_gradients = qml.numpy.abs(gradients)

    near_zero_count = qml.numpy.sum(
        absolute_gradients < threshold
    )

    total_count = qml.numpy.size(gradients)

    return (
        near_zero_count / total_count
    ) * 100.0


def calculate_circuit_characteristics(
    weights,
    circuit,
    n_qubits,
    depth
):
    """
    Calculate structural characteristics of a PQC.

    Parameters
    ----------
    weights : array-like
        Trainable circuit parameters.

    circuit : qml.QNode
        Parameterized quantum circuit.

    n_qubits : int
        Number of qubits.

    depth : int
        Circuit depth.

    Returns
    -------
    dict
        Circuit characteristics.
    """

    n_trainable_parameters = qml.numpy.size(weights)

    # Execute the circuit so PennyLane records its operations.
    circuit(weights)

    n_gates = len(circuit._tape.operations)

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "n_trainable_parameters": n_trainable_parameters,
        "n_gates": n_gates
    }


def calculate_all_indicators(
    gradients,
    weights,
    circuit,
    n_qubits,
    depth,
    near_zero_threshold=0.001
):
    """
    Calculate all selected trainability indicators.

    Returns
    -------
    dict
        Seven diagnostic indicators.
    """

    mean_gradient_magnitude = (
        calculate_mean_gradient_magnitude(
            gradients
        )
    )

    gradient_variance = (
        calculate_gradient_variance(
            gradients
        )
    )

    near_zero_percentage = (
        calculate_near_zero_percentage(
            gradients,
            threshold=near_zero_threshold
        )
    )

    circuit_characteristics = (
        calculate_circuit_characteristics(
            weights=weights,
            circuit=circuit,
            n_qubits=n_qubits,
            depth=depth
        )
    )

    return {
        "mean_gradient_magnitude": (
            mean_gradient_magnitude
        ),
        "gradient_variance": (
            gradient_variance
        ),
        "near_zero_percentage": (
            near_zero_percentage
        ),
        "n_qubits": (
            circuit_characteristics["n_qubits"]
        ),
        "depth": (
            circuit_characteristics["depth"]
        ),
        "n_trainable_parameters": (
            circuit_characteristics[
                "n_trainable_parameters"
            ]
        ),
        "n_gates": (
            circuit_characteristics["n_gates"]
        )
    }


if __name__ == "__main__":

    # Small demonstration using a simple gradient array.
    gradients = qml.numpy.array(
        [
            [0.10, 0.20],
            [0.001, 0.0001]
        ]
    )

    print("Indicator module test")
    print(
        "Mean gradient magnitude:",
        calculate_mean_gradient_magnitude(
            gradients
        )
    )

    print(
        "Gradient variance:",
        calculate_gradient_variance(
            gradients
        )
    )

    print(
        "Near-zero percentage:",
        calculate_near_zero_percentage(
            gradients
        ),
        "%"
    )