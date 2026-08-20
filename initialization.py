def generate_seeds(
    n_initializations: int = 5,
    start_seed: int = 42
):
    """
    Generate reproducible random seeds for circuit initialization.

    Parameters
    ----------
    n_initializations : int
        Number of different initializations to generate.

    start_seed : int
        First seed value.

    Returns
    -------
    list[int]
        List of reproducible seeds.
    """

    if n_initializations < 1:
        raise ValueError("n_initializations must be at least 1.")

    return [
        start_seed + i
        for i in range(n_initializations)
    ]