import argparse
from pathlib import Path

from inspect_data import open_database
from calibrate_probabilities import load_predictions
from race_metrics import evaluate_race_scores


def main():
    parser = argparse.ArgumentParser(
        description="Inspect validation race history coverage."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()

    connection = open_database(args.database.resolve())

    try:
        rows = connection.execute(
            """
            SELECT
                date, course, off,
                COUNT(*) AS runners,
                SUM(
                    CASE WHEN prior_starts = 0 THEN 1 ELSE 0 END
                ) AS missing_history
            FROM features
            WHERE date >= '2024-01-01'
              AND date < '2025-01-01'
            GROUP BY date, course, off
            ORDER BY date, course, off
            """
        ).fetchall()
        runner_rows = connection.execute(
            """
            SELECT date, course, off, horse, prior_starts, won
            FROM features
            WHERE date >= '2024-01-01'
              AND date < '2025-01-01'
            ORDER BY date, course, off, horse
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError("No validation races found")

    majority_missing = sum(
        missing > runners / 2
        for _, _, _, runners, missing in rows
    )

    print(f"Validation races: {len(rows):,}")
    print(f"Majority without history: {majority_missing:,}")
    print(f"Other races: {len(rows) - majority_missing:,}")

    predictions = load_predictions(
        args.predictions,
        expected_year=2024,
    )

    expected_counts = {
        (date, course, off): runners
        for date, course, off, runners, _ in rows
    }

    if set(predictions) != set(expected_counts):
        raise ValueError(
            "Prediction races do not match the database selection"
        )

    for race_key, runners in predictions.items():
        if len(runners) != expected_counts[race_key]:
            raise ValueError(
                f"Runner count mismatch for {race_key}"
            )

    majority_keys = {
        (date, course, off)
        for date, course, off, runners, missing in rows
        if missing > runners / 2
    }

    groups = {
        "Majority without history": {},
        "Other races": {},
    }

    for race_key, runners in predictions.items():
        group = (
            "Majority without history"
            if race_key in majority_keys
            else "Other races"
        )
        groups[group][race_key] = runners

    for label, races in groups.items():
        if not races:
            print(f"{label}: no races")
            continue

        model_loss, uniform_loss = evaluate_race_scores(races)

        print(
            f"{label}: {len(races):,} races | "
            f"Model: {model_loss:.6f} | "
            f"Uniform: {uniform_loss:.6f} | "
            f"Improvement: {uniform_loss - model_loss:+.6f}"
        )

    database_runners = {}

    for date, course, off, horse, prior_starts, won in runner_rows:
        key = (date, course, off, horse)

        if key in database_runners:
            raise ValueError(f"Duplicate database runner: {key}")

        if prior_starts is None or prior_starts < 0:
            raise ValueError(f"Invalid prior-start count: {key}")

        database_runners[key] = (prior_starts, int(won))

    predicted_runners = {
        (*race_key, horse): (float(probability), int(won))
        for race_key, runners in predictions.items()
        for horse, won, probability in runners
    }

    if set(database_runners) != set(predicted_runners):
        raise ValueError(
            "Prediction runners do not exactly match database runners"
        )

    coverage_groups = {
        "No earlier recorded history": [],
        "Has earlier recorded history": [],
    }

    for key, (prior_starts, actual_won) in database_runners.items():
        probability, predicted_won = predicted_runners[key]

        if predicted_won != actual_won:
            raise ValueError(f"Winner indicator mismatch: {key}")

        label = (
            "No earlier recorded history"
            if prior_starts == 0
            else "Has earlier recorded history"
        )

        coverage_groups[label].append((probability, actual_won))

    print("\nRunner-level history calibration:")

    for label, values in coverage_groups.items():
        count = len(values)

        if count == 0:
            print(f"{label}: no runners")
            continue

        mean_probability = sum(p for p, _ in values) / count
        observed_rate = sum(won for _, won in values) / count
        gap_pp = 100 * (observed_rate - mean_probability)

        print(
            f"{label} | Runners: {count:,} | "
            f"Mean predicted: {mean_probability:.2%} | "
            f"Observed wins: {observed_rate:.2%} | "
            f"Observed minus predicted: {gap_pp:+.2f} pp"
        )


if __name__ == "__main__":
    main()