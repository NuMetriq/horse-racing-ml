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