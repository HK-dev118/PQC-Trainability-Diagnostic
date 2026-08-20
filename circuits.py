import pennylane as qml


def create_pqc(
    n_qubits: int,
    depth: int,
    seed: int = 42,
    ansatz: str = "ring"
):
    """
    Create a parameterized quantum circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.

    depth : int
        Number of repeated variational layers.

    seed : int
        Random seed used to initialize the circuit parameters.

    ansatz : str
        Entanglement structure:
        - "linear": nearest-neighbor entanglement
        - "ring": nearest-neighbor entanglement + final-to-first connection

    Returns
    -------
    circuit : qml.QNode
        Parameterized quantum circuit.

    weights : numpy.ndarray
        Initial parameter values.
    """

    if n_qubits < 2:
        raise ValueError("n_qubits must be at least 2.")

    if depth < 1:
        raise ValueError("depth must be at least 1.")

    if ansatz not in {"linear", "ring"}:
        raise ValueError("ansatz must be 'linear' or 'ring'.")

    # Reproducible random initialization
    rng = qml.numpy.random.default_rng(seed)

    # Two trainable parameters (RX + RY) per qubit per layer
    weights = rng.uniform(
        low=-qml.numpy.pi,
        high=qml.numpy.pi,
        size=(depth, n_qubits, 2),
        requires_grad=True
    )

    dev = qml.device(
        "default.qubit",
        wires=n_qubits
    )

    @qml.qnode(dev)
    def circuit(weights):

        for layer in range(depth):

            # Trainable single-qubit rotations
            for qubit in range(n_qubits):
                qml.RX(
                    weights[layer, qubit, 0],
                    wires=qubit
                )

                qml.RY(
                    weights[layer, qubit, 1],
                    wires=qubit
                )

            # Linear nearest-neighbor entanglement
            for qubit in range(n_qubits - 1):
                qml.CNOT(
                    wires=[qubit, qubit + 1]
                )

            # Additional connection for ring topology
            if ansatz == "ring" and n_qubits > 2:
                qml.CNOT(
                    wires=[n_qubits - 1, 0]
                )

        return qml.expval(qml.PauliZ(0))

    return circuit, weights