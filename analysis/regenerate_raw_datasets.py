"""
Regenerates data/dev_measurement_dataset.csv (the RAW, un-normalized
version) since only the normalized version was saved earlier.

This just re-runs Hiba's existing measurement pipeline on the same
dev-set configs -- no new data, just recreating a file that was
already produced once but not saved separately.

Save this as: analysis/regenerate_raw_dev_dataset.py
"""

import pandas as pd

from circuits.circuits_config import (
    create_experiment_configurations,
    split_configurations,
)
from diagnostics.measurement_dataset import create_measurement_dataset


if __name__ == "__main__":
    configs = create_experiment_configurations(
        qubit_counts=(2, 4, 6, 8, 10, 12, 14),
        depths=(1, 2, 3, 4, 6, 8, 10),
    )
    dev_set, test_set = split_configurations(configs)

    print(f"Regenerating raw measurements for {len(dev_set)} "
          f"dev-set circuits...")
    dev_measurements = create_measurement_dataset(dev_set)
    dev_measurements.to_csv("data/dev_measurement_dataset.csv", index=False)
    print("Saved: data/dev_measurement_dataset.csv")

    print(f"\nRegenerating raw measurements for {len(test_set)} "
          f"test-set circuits...")
    test_measurements = create_measurement_dataset(test_set)
    test_measurements.to_csv("data/test_measurement_dataset.csv", index=False)
    print("Saved: data/test_measurement_dataset.csv")
