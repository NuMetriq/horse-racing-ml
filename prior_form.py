from itertools import groupby
from collections import deque
from datetime import date as calendar_date

def calculate_prior_form(history):
    prior_starts = 0
    prior_wins = 0
    features = []

    for date, records in groupby(history, key=lambda row: row[0]):
        day_records = list(records)

        prior_rate = (
            prior_wins / prior_starts
            if prior_starts > 0
            else None
        )

        for _, course, off, position in day_records:
            features.append(
                (
                    date, course, off, position,
                    prior_starts, prior_wins, prior_rate,
                )
            )

        prior_starts += len(day_records)
        prior_wins += sum(
            position == "1"
            for _, _, _, position in day_records
        )

    return features

def smoothed_win_rate(
    prior_wins: int,
    prior_starts: int,
    reference_rate: float,
    alpha: float = 10.0,
) -> float:
    return (
        prior_wins + alpha * reference_rate
    ) / (
        prior_starts + alpha
    )

def calculate_recent_form(history, window_days=365):
    past_results = deque()
    features = []

    for date, records in groupby(history, key=lambda row: row[0]):
        current_date = calendar_date.fromisoformat(date)
        day_records = list(records)

        while (
            past_results
            and (current_date - past_results[0][0]).days > window_days
        ):
            past_results.popleft()

        prior_starts = len(past_results)
        prior_wins = sum(won for _, won in past_results)
        prior_rate = (
            prior_wins / prior_starts
            if prior_starts > 0
            else None
        )

        for _, course, off, position in day_records:
            features.append(
                (
                    date, course, off, position,
                    prior_starts, prior_wins, prior_rate,
                )
            )

        for _, _, _, position in day_records:
            past_results.append((current_date, position == "1"))

    return features

def calculate_days_since_run(history):
    previous_date = None
    gaps = []

    for date, records in groupby(history, key=lambda row: row[0]):
        current_date = calendar_date.fromisoformat(date)
        day_records = list(records)

        days_since_run = (
            (current_date - previous_date).days
            if previous_date is not None
            else None
        )

        for _ in day_records:
            gaps.append(days_since_run)

        previous_date = current_date

    return gaps

def calculate_previous_position(history):
    """Return the previous recorded position, excluding same-day results."""
    previous_position = None
    positions = []

    for date, records in groupby(history, key=lambda row: row[0]):
        day_records = list(records)

        for _ in day_records:
            positions.append(previous_position)

        if len(day_records) == 1:
            previous_position = day_records[0][3]
        else:
            # Multiple records on one date have no reliable ordering.
            previous_position = None

    return positions

def calculate_previous_runner_count(history, runner_counts):
    """Return the previous race's field size, excluding same-day results."""
    previous_count = None
    counts = []

    paired_records = zip(history, runner_counts, strict=True)

    for date, records in groupby(
        paired_records, key=lambda pair: pair[0][0]
    ):
        day_records = list(records)

        for _ in day_records:
            counts.append(previous_count)

        if len(day_records) == 1:
            previous_count = day_records[0][1]
        else:
            previous_count = None

    return counts

def calculate_features_as_of(history, runner_counts, race_date):
    """Build one horse's source features using dates before race_date."""
    calendar_date.fromisoformat(race_date)

    earlier_records = [
        (record, count)
        for record, count in zip(history, runner_counts, strict=True)
        if record[0] < race_date
    ]
    earlier_records.sort(key=lambda pair: pair[0][:3])

    earlier_history = [record for record, _ in earlier_records]
    earlier_counts = [count for _, count in earlier_records]

    # Temporary row used only to request features for the target date.
    target_history = earlier_history + [(race_date, "", "", None)]
    target_counts = earlier_counts + [None]

    form = calculate_prior_form(target_history)[-1]
    gap = calculate_days_since_run(target_history)[-1]
    position = calculate_previous_position(target_history)[-1]
    runner_count = calculate_previous_runner_count(
        target_history, target_counts
    )[-1]

    return {
        "prior_starts": form[4],
        "prior_wins": form[5],
        "prior_win_rate": form[6],
        "days_since_run": gap,
        "previous_position": position,
        "previous_runner_count": runner_count,
    }