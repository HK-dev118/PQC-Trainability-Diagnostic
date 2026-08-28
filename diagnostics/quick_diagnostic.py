import pennylane as qml

from circuits.initialisation import create_pqc


def calculate_gradients(circuit, weights, n_qubits):
    """
    Calculate gradients of every circuit output with respect
    to all trainable parameters.
    """

    gradients = []

    for output_index in range(n_qubits):

        def scalar_output(params):
            output = circuit(params)
            return output[output_index]

        gradient_function = qml.grad(scalar_output)

        gradient = gradient_function(weights)

        gradients.append(gradient)

    return qml.numpy.stack(gradients)


def calculate_mean_gradient_magnitude(gradients):
    """
    Calculate the mean absolute gradient magnitude.
    """

    return qml.numpy.mean(
        qml.numpy.abs(gradients)
    )


def calculate_gradient_variance(gradients):
    """
    Calculate the variance of the gradient values.
    """

    return qml.numpy.var(gradients)


def calculate_near_zero_percentage(
    gradients,
    threshold=0.001
):
    """
    Calculate the percentage of gradients whose absolute
    value is below the specified threshold.
    """

    absolute_gradients = qml.numpy.abs(
        gradients
    )

    near_zero_count = qml.numpy.sum(
        absolute_gradients < threshold
    )

    total_count = qml.numpy.size(
        gradients
    )

    return (
        near_zero_count / total_count
    ) * 100.0


def calculate_circuit_characteristics(
    circuit,
    weights,
    n_qubits
):
    """
    Calculate basic structural characteristics of a PQC.
    """

    n_trainable_parameters = qml.numpy.size(
        weights
    )

    # Execute once so PennyLane records operations.
    circuit(weights)

    n_gates = len(
        circuit._tape.operations
    )

    return {
        "n_qubits": n_qubits,
        "n_trainable_parameters": (
            n_trainable_parameters
        ),
        "n_gates": n_gates
    }


def run_quick_diagnostic(
    n_qubits: int,
    depth: int,
    ansatz: str = "ring",
    seed: int = 42
):
    """
    Run a lightweight diagnostic simulation on a generated PQC.
    """

    circuit, weights = create_pqc(
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
        ansatz=ansatz
    )

    return run_custom_circuit_diagnostic(
        circuit=circuit,
        weights=weights,
        n_qubits=n_qubits,
        depth=depth
    )


def run_custom_circuit_diagnostic(
    circuit,
    weights,
    n_qubits: int,
    depth: int
):
    """
    Run the diagnostic on a user-provided parameterized
    PennyLane quantum circuit.
    """

    if not isinstance(
        circuit,
        qml.QNode
    ):
        raise TypeError(
            "circuit must be a PennyLane QNode."
        )

    if n_qubits < 1:
        raise ValueError(
            "n_qubits must be at least 1."
        )

    if depth < 1:
        raise ValueError(
            "depth must be at least 1."
        )

    # Ensure parameters are differentiable.
    weights = qml.numpy.array(
        weights,
        requires_grad=True
    )

    # Forward simulation.
    output = circuit(weights)

    # Calculate gradients.
    gradients = calculate_gradients(
        circuit,
        weights,
        n_qubits
    )

    # Calculate indicators.
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
            gradients
        )
    )

    # Circuit characteristics.
    characteristics = (
        calculate_circuit_characteristics(
            circuit,
            weights,
            n_qubits
        )
    )

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "output": output,
        "gradients": gradients,
        "mean_gradient_magnitude": (
            mean_gradient_magnitude
        ),
        "gradient_variance": (
            gradient_variance
        ),
        "near_zero_percentage": (
            near_zero_percentage
        ),
        "n_trainable_parameters": (
            characteristics[
                "n_trainable_parameters"
            ]
        ),
        "n_gates": (
            characteristics[
                "n_gates"
            ]
        )
    }


def print_diagnostic_result(result):
    """
    Display diagnostic measurements in a readable format.
    """

    print("\n" + "=" * 55)

    print(
        "QUICK PQC TRAINABILITY DIAGNOSTIC"
    )

    print("=" * 55)

    print(
        "Mean gradient magnitude: "
        f"{float(result['mean_gradient_magnitude']):.6f}"
    )

    print(
        "Gradient variance: "
        f"{float(result['gradient_variance']):.6f}"
    )

    print(
        "Near-zero gradients: "
        f"{float(result['near_zero_percentage']):.2f}%"
    )

    print(
        f"Qubits: {result['n_qubits']}"
    )

    print(
        f"Depth: {result['depth']}"
    )

    print(
        "Parameters: "
        f"{result['n_trainable_parameters']}"
    )

    print(
        f"Gates: {result['n_gates']}"
    )

    print("=" * 55)


if __name__ == "__main__":

    result = run_quick_diagnostic(
        n_qubits=4,
        depth=2,
        ansatz="ring",
        seed=42
    )

    print(
        "Quick diagnostic completed successfully."
    )

    print_diagnostic_result(result)