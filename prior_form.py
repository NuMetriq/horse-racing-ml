from itertools import groupby

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