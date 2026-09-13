import numpy as np


def log_days_since_run(X):
    """Apply log1p to the fourth input column: days_since_run."""
    transformed = np.array(X, dtype=float, copy=True)
    transformed[:, 3] = np.log1p(transformed[:, 3])
    return transformed

def encode_previous_position(value):
    """Return numeric finish position and a result-code flag."""
    if value is None:
        return np.nan, 0.0

    text = str(value).strip()

    if text in ("", "0"):
        return np.nan, 0.0

    if text.isdecimal():
        return float(text), 0.0

    return np.nan, 1.0

def encode_previous_position_field(row):
    """Convert six source values into seven numeric model inputs."""
    (
        prior_starts,
        prior_wins,
        prior_win_rate,
        days_since_run,
        previous_position,
        previous_runner_count,
    ) = row

    finish_position, result_was_code = encode_previous_position(
        previous_position
    )

    return (
        prior_starts,
        prior_wins,
        prior_win_rate,
        days_since_run,
        finish_position,
        result_was_code,
        previous_runner_count,
    )