import argparse
from pathlib import Path

from inspect_data import open_database
from prior_form import calculate_features_as_of


def main():
    parser = argparse.ArgumentParser(
        description="Compare as-of features with a prepared feature row."
    )
    parser.add_argument("history_database", type=Path)
    parser.add_argument("feature_database", type=Path)
    args = parser.parse_args()

    horse = "Jadavi (AUS)"
    race_date = "2024-01-01"
    course = "Ascot (AUS)"
    off = "7:50"

    connection = open_database(args.history_database.resolve())
    try:
        rows = connection.execute(
            """
            SELECT r.date, r.course, r.off, r.finish_position,
                   races.runner_count
            FROM runners AS r
            JOIN races
                ON r.date = races.date
                AND r.course = races.course
                AND r.off = races.off
            WHERE r.horse = ? AND r.date < ?
            ORDER BY r.date, r.course, r.off
            """,
            (horse, race_date),
        ).fetchall()
    finally:
        connection.close()

    actual = calculate_features_as_of(
        [row[:4] for row in rows],
        [row[4] for row in rows],
        race_date,
    )

    connection = open_database(args.feature_database.resolve())
    try:
        expected_row = connection.execute(
            """
            SELECT prior_starts, prior_wins, prior_win_rate,
                   days_since_run, previous_position,
                   previous_runner_count
            FROM features
            WHERE date = ? AND course = ? AND off = ? AND horse = ?
            """,
            (race_date, course, off, horse),
        ).fetchone()
    finally:
        connection.close()

    if expected_row is None:
        raise ValueError("Prepared feature row not found")

    names = [
        "prior_starts",
        "prior_wins",
        "prior_win_rate",
        "days_since_run",
        "previous_position",
        "previous_runner_count",
    ]
    expected = dict(zip(names, expected_row, strict=True))

    for name in names:
        print(
            f"{name}: calculated={actual[name]!r}, "
            f"prepared={expected[name]!r}"
        )

    if actual != expected:
        raise ValueError("As-of features do not match prepared features")

    print("All six source features match.")


if __name__ == "__main__":
    main()