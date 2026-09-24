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

def calculate_relative_finish(position, runner_count):
    """Scale a valid finishing position from 0 (first) to 1 (last)."""
    numeric_position, _ = encode_previous_position(position)

    if runner_count is None:
        return np.nan

    try:
        count = float(runner_count)
    except (TypeError, ValueError):
        return np.nan

    if (
        not np.isfinite(numeric_position)
        or not np.isfinite(count)
        or not count.is_integer()
        or count < 2
        or not 1 <= numeric_position <= count
    ):
        return np.nan

    return (numeric_position - 1) / (count - 1)

def encode_previous_relative_finish(row):
    """Keep the seven existing inputs and append relative finish."""
    existing_inputs = encode_previous_position_field(row)
    relative_finish = calculate_relative_finish(row[4], row[5])

    return (*existing_inputs, relative_finish)

def encode_age(value):
    """Return integer-valued age >= 2, otherwise a missing value."""
    if value is None:
        return np.nan

    try:
        age = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(age) or not age.is_integer() or age < 2:
        return np.nan

    return age

def encode_relative_finish_age(row):
    """Convert seven source values into nine model inputs."""
    (
        prior_starts,
        prior_wins,
        prior_win_rate,
        days_since_run,
        previous_position,
        previous_runner_count,
        age,
    ) = row

    existing_inputs = encode_previous_relative_finish(
        (
            prior_starts,
            prior_wins,
            prior_win_rate,
            days_since_run,
            previous_position,
            previous_runner_count,
        )
    )

    return (*existing_inputs, encode_age(age))

def encode_relative_finish_age_distance_change(row):
    """Convert eight source values into ten model inputs."""
    existing_inputs = encode_relative_finish_age(row[:7])
    change = row[7]

    return (
        *existing_inputs,
        np.nan if change is None else float(change),
    )