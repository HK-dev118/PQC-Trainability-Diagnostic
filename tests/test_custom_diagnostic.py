import pennylane as qml

from diagnostics.quick_diagnostic import (
    run_custom_circuit_diagnostic,
    print_diagnostic_result
)


# =========================================================
# TEST 1: Basic 2-qubit circuit
# =========================================================

def test_basic_two_qubit_circuit():

    dev = qml.device(
        "default.qubit",
        wires=2
    )

    @qml.qnode(dev)
    def circuit(weights):

        qml.RX(
            weights[0],
            wires=0
        )

        qml.RY(
            weights[1],
            wires=1
        )

        qml.CNOT(
            wires=[0, 1]
        )

        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1))
        ]

    weights = qml.numpy.array(
        [0.5, 0.7],
        requires_grad=True
    )

    result = run_custom_circuit_diagnostic(
        circuit=circuit,
        weights=weights,
        n_qubits=2,
        depth=1
    )

    assert result["n_qubits"] == 2
    assert result["depth"] == 1
    assert result["n_trainable_parameters"] == 2
    assert result["n_gates"] == 3
    assert len(result["output"]) == 2

    print("\nTEST 1 PASSED: Basic 2-qubit circuit")
    print_diagnostic_result(result)

    return result


# =========================================================
# TEST 2: Different circuit configuration
# =========================================================

def test_three_qubit_circuit():

    dev = qml.device(
        "default.qubit",
        wires=3
    )

    @qml.qnode(dev)
    def circuit(weights):

        for i in range(3):

            qml.RY(
                weights[i],
                wires=i
            )

        qml.CNOT(
            wires=[0, 1]
        )

        qml.CNOT(
            wires=[1, 2]
        )

        return [
            qml.expval(qml.PauliZ(i))
            for i in range(3)
        ]

    weights = qml.numpy.array(
        [0.2, 0.5, 0.8],
        requires_grad=True
    )

    result = run_custom_circuit_diagnostic(
        circuit=circuit,
        weights=weights,
        n_qubits=3,
        depth=1
    )

    assert result["n_qubits"] == 3
    assert result["depth"] == 1
    assert result["n_trainable_parameters"] == 3
    assert result["n_gates"] == 5
    assert len(result["output"]) == 3

    print("\nTEST 2 PASSED: Different 3-qubit circuit")
    print_diagnostic_result(result)

    return result


# =========================================================
# TEST 3: Different depth
# =========================================================

def test_deeper_circuit():

    dev = qml.device(
        "default.qubit",
        wires=2
    )

    @qml.qnode(dev)
    def circuit(weights):

        for layer in range(2):

            qml.RX(
                weights[layer, 0],
                wires=0
            )

            qml.RY(
                weights[layer, 1],
                wires=1
            )

            qml.CNOT(
                wires=[0, 1]
            )

        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1))
        ]

    weights = qml.numpy.array(
        [
            [0.2, 0.3],
            [0.6, 0.8]
        ],
        requires_grad=True
    )

    result = run_custom_circuit_diagnostic(
        circuit=circuit,
        weights=weights,
        n_qubits=2,
        depth=2
    )

    assert result["n_qubits"] == 2
    assert result["depth"] == 2
    assert result["n_trainable_parameters"] == 4
    assert result["n_gates"] == 6
    assert len(result["output"]) == 2

    print("\nTEST 3 PASSED: Deeper circuit")
    print_diagnostic_result(result)

    return result


# =========================================================
# TEST 4: Different circuits produce different measurements
# =========================================================

def test_measurements_are_different(
    result_1,
    result_2
):

    value_1 = float(
        result_1[
            "mean_gradient_magnitude"
        ]
    )

    value_2 = float(
        result_2[
            "mean_gradient_magnitude"
        ]
    )

    assert value_1 != value_2

    print(
        "\nTEST 4 PASSED: "
        "Different circuits produce different measurements"
    )


# =========================================================
# TEST 5: Invalid number of qubits
# =========================================================

def test_invalid_qubits():

    try:

        run_custom_circuit_diagnostic(
            circuit="not_a_circuit",
            weights=[0.1],
            n_qubits=0,
            depth=1
        )

    except (TypeError, ValueError):

        print(
            "\nTEST 5 PASSED: "
            "Invalid circuit/qubit input handled correctly"
        )

        return

    raise AssertionError(
        "Invalid input should have raised an error."
    )


# =========================================================
# TEST 6: Invalid depth
# =========================================================

def test_invalid_depth():

    dev = qml.device(
        "default.qubit",
        wires=2
    )

    @qml.qnode(dev)
    def circuit(weights):

        qml.RX(
            weights[0],
            wires=0
        )

        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliZ(1))
        ]

    try:

        run_custom_circuit_diagnostic(
            circuit=circuit,
            weights=[0.5],
            n_qubits=2,
            depth=0
        )

    except ValueError:

        print(
            "\nTEST 6 PASSED: "
            "Invalid depth handled correctly"
        )

        return

    raise AssertionError(
        "Invalid depth should have raised an error."
    )


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "RUNNING CUSTOM CIRCUIT DIAGNOSTIC TESTS"
    )

    print(
        "========================================"
    )

    result_1 = test_basic_two_qubit_circuit()

    result_2 = test_three_qubit_circuit()

    result_3 = test_deeper_circuit()

    test_measurements_are_different(
        result_1,
        result_2
    )

    test_invalid_qubits()

    test_invalid_depth()

    print(
        "\n========================================"
    )

    print(
        "ALL CUSTOM DIAGNOSTIC TESTS PASSED!"
    )

    print(
        "========================================"
    )