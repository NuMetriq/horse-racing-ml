import numpy as np


def log_days_since_run(X):
    """Apply log1p to the fourth input column: days_since_run."""
    transformed = np.array(X, dtype=float, copy=True)
    transformed[:, 3] = np.log1p(transformed[:, 3])
    return transformed