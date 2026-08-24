# Measurement Procedure

## 1. Purpose

The measurement stage provides lightweight indicators of the potential
trainability of parameterized quantum circuits (PQCs) before full training.

The measurements are used as early diagnostic information and are not
treated as guarantees of successful or unsuccessful training.

The selected indicators consist of:

### Gradient-related indicators

1. Mean gradient magnitude
2. Gradient variance
3. Percentage of near-zero gradients

### Circuit-related indicators

4. Number of qubits
5. Circuit depth
6. Number of trainable parameters
7. Number of gates

---

## 2. Circuit Generation

The experiment configurations are generated using the project's circuit
configuration module.

The current experimental configuration varies:

- Number of qubits: 2, 4, 6, 8, 10, 12, 14
- Circuit depths: 1, 2, 3, 4, 6, 8, 10
- Ansatz structures:
  - Linear
  - Ring
- Random initialization seeds:
  - 42
  - 43
  - 44
  - 45
  - 46

This produces:

7 qubit counts × 7 depths × 2 ansatzes × 5 seeds = 490
circuit configurations.

---

## 3. Parameter Initialization

Each circuit uses a reproducible random initialization.

The initialization seed is stored as part of the experiment configuration.

The current seeds are:

42, 43, 44, 45, 46.

Using explicit seeds allows the same circuit configuration to be recreated
for consistency checks and later experiments.

---

## 4. Quick Diagnostic Simulation

For each circuit configuration, a lightweight forward simulation is performed
before full training.

The circuit returns the expectation value of the Pauli-Z observable for each
qubit.

The resulting outputs are used to verify that the circuit executes
successfully before gradient measurements are calculated.

---

## 5. Gradient Calculation

Gradients are calculated with respect to all trainable circuit parameters.

Because the circuit produces one output for each qubit, the gradient of each
output is calculated separately.

The resulting gradient array contains the gradient of every circuit output
with respect to every trainable parameter.

The gradient tensor is therefore retained for calculating the selected
gradient-based indicators.

---

## 6. Mean Gradient Magnitude

The mean gradient magnitude is calculated as the mean of the absolute values
of all calculated gradients.

Mathematically:

mean gradient magnitude = mean(|gradient|)

This provides a measure of the overall strength of the gradient signal.

A smaller value indicates that the gradient signal is weaker on average,
although a small value alone is not treated as proof of an untrainable
circuit.

---

## 7. Gradient Variance

Gradient variance is calculated across all gradient values.

It measures how much the gradient values vary around their mean.

A circuit may therefore have gradients with different magnitudes and
distributions even when their average magnitude is similar.

---

## 8. Near-Zero Gradient Percentage

A gradient is classified as near-zero when its absolute value is below:

0.001

The percentage is calculated as:

near-zero percentage =
(number of gradients with |gradient| < 0.001 /
 total number of gradients) × 100

The threshold is used as an operational diagnostic threshold.

It is not interpreted as a universal definition of a barren plateau or
training failure.

---

## 9. Circuit Characteristics

The following structural characteristics are recorded for each circuit:

### Number of qubits

The number of qubits used by the circuit.

### Circuit depth

The number of repeated variational layers.

### Number of trainable parameters

The total number of trainable parameter values in the circuit.

For the current circuit design, there are two trainable rotation parameters
per qubit per layer.

### Number of gates

The number of quantum operations recorded in the constructed circuit.

---

## 10. Measurement Dataset

The measurements are stored in a structured tabular dataset.

Each row corresponds to one circuit configuration and initialization.

The dataset contains:

- Circuit ID
- Number of qubits
- Circuit depth
- Ansatz
- Random seed
- Mean gradient magnitude
- Gradient variance
- Near-zero gradient percentage
- Number of trainable parameters
- Number of gates

This dataset forms the input to the later diagnostic development and
evaluation stages.

---

## 11. Measurement Consistency

Measurement consistency is checked by running the same circuit configuration
and initialization more than once.

The repeated measurements are compared using a numerical tolerance.

The current consistency test uses a tolerance of:

1 × 10^-12

The test is considered passed when the repeated numerical measurements agree
within this tolerance.

This verifies that the diagnostic behaves reproducibly for identical circuit
configurations and initialization seeds.

---

## 12. Development and Test Sets

The experiment configurations are divided into approximately:

- 70% development data
- 30% held-out test data

The development set is used for diagnostic development and normalization.

The held-out test set remains separate until the diagnostic has been
finalized.

---

## 13. Indicator Normalization

The selected indicators have different numerical scales.

Min-max normalization is therefore used to transform each indicator to a
comparable scale.

For an indicator x:

normalized x = (x - minimum) / (maximum - minimum)

The minimum and maximum values used for normalization are calculated from
the development set.

The resulting normalization parameters are then reused when transforming
other data, including the held-out test set.

This prevents information from the test set from being used to determine
the normalization parameters.

If an indicator has the same minimum and maximum value throughout the
development data, its normalized value is set to 0.

---

## 14. Reproducibility

The measurement procedure is implemented in Python using the project's
PennyLane-based circuit generation and diagnostic modules.

The experiment configuration, random initialization seed, diagnostic
measurements, and circuit characteristics are recorded for each circuit.

This allows the measurement process to be repeated using the same
configuration and initialization information.

---

## 15. Interpretation

The measurements are treated as diagnostic indicators rather than definitive
proof of trainability.

In particular:

- A small mean gradient does not automatically imply training failure.
- A high percentage of near-zero gradients does not automatically prove a
  barren plateau.
- Circuit size or depth alone does not determine trainability.
- The combined diagnostic must ultimately be evaluated against observed
  training behavior.

The purpose of the measurement stage is therefore to provide lightweight,
interpretable information that can later be compared with full-training
outcomes.