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

    absolute_gradients = qml.numpy.abs(gradients)

    return qml.numpy.mean(absolute_gradients)


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

    absolute_gradients = qml.numpy.abs(gradients)

    near_zero_count = qml.numpy.sum(
        absolute_gradients < threshold
    )

    total_count = qml.numpy.size(gradients)

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

    n_trainable_parameters = qml.numpy.size(weights)

    # Build the circuit once so PennyLane records its operations.
    circuit(weights)

    n_gates = len(circuit._tape.operations)

    return {
        "n_qubits": n_qubits,
        "n_trainable_parameters": n_trainable_parameters,
        "n_gates": n_gates
    }


def run_quick_diagnostic(
    n_qubits: int,
    depth: int,
    ansatz: str = "ring",
    seed: int = 42
):
    """
    Run a lightweight diagnostic simulation.
    """

    # Create the parameterized quantum circuit
    circuit, weights = create_pqc(
        n_qubits=n_qubits,
        depth=depth,
        seed=seed,
        ansatz=ansatz
    )

    # Make parameters explicitly differentiable
    weights = qml.numpy.array(
        weights,
        requires_grad=True
    )

    # Task 10: forward simulation
    output = circuit(weights)

    # Task 11: calculate gradients
    gradients = calculate_gradients(
        circuit,
        weights,
        n_qubits
    )

    # Task 12: mean gradient magnitude
    mean_gradient_magnitude = calculate_mean_gradient_magnitude(
        gradients
    )

    # Task 13: gradient variance
    gradient_variance = calculate_gradient_variance(
        gradients
    )

    # Task 14: near-zero gradient percentage
    near_zero_percentage = calculate_near_zero_percentage(
        gradients,
        threshold=0.001
    )

    # Task 15: circuit characteristics
    circuit_characteristics = calculate_circuit_characteristics(
        circuit,
        weights,
        n_qubits
    )

    return {
        "n_qubits": n_qubits,
        "depth": depth,
        "ansatz": ansatz,
        "seed": seed,
        "output": output,
        "gradients": gradients,
        "mean_gradient_magnitude": mean_gradient_magnitude,
        "gradient_variance": gradient_variance,
        "near_zero_percentage": near_zero_percentage,
        "n_trainable_parameters": (
            circuit_characteristics[
                "n_trainable_parameters"
            ]
        ),
        "n_gates": (
            circuit_characteristics[
                "n_gates"
            ]
        )
    }


if __name__ == "__main__":

    result = run_quick_diagnostic(
        n_qubits=4,
        depth=2,
        ansatz="ring",
        seed=42
    )

    print("Quick diagnostic completed successfully.")

    print(
        f"Qubits: "
        f"{result['n_qubits']}"
    )

    print(
        f"Depth: "
        f"{result['depth']}"
    )

    print(
        f"Ansatz: "
        f"{result['ansatz']}"
    )

    print(
        f"Seed: "
        f"{result['seed']}"
    )

    print(
        f"Circuit output: "
        f"{result['output']}"
    )

    print(
        f"Gradient shape: "
        f"{result['gradients'].shape}"
    )

    print(
        f"Mean gradient magnitude: "
        f"{result['mean_gradient_magnitude']}"
    )

    print(
        f"Gradient variance: "
        f"{result['gradient_variance']}"
    )

    print(
        f"Near-zero gradient percentage: "
        f"{result['near_zero_percentage']}%"
    )

    print(
        f"Trainable parameters: "
        f"{result['n_trainable_parameters']}"
    )

    print(
        f"Number of gates: "
        f"{result['n_gates']}"
    )