import argparse
import math
from pathlib import Path

from inspect_data import open_database
from predict_boosting_racecard import build_runner_features


CASES = [
    ("2024-01-01", "Santa Anita (USA)", "11:43"),
    ("2024-01-01", "Ascot (AUS)", "9:02"),
    ("2024-01-13", "Chelmsford (AW)", "4:45"),
    ("2024-01-01", "Newcastle (AW)", "2:00"),
]

FIELDS = [
    "prior_starts",
    "prior_wins",
    "prior_win_rate",
    "days_since_run",
    "previous_position",
    "previous_runner_count",
    "age",
    "distance_change_furlongs",
]


def values_match(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(
        expected, (int, float)
    ):
        return math.isclose(
            actual, expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return actual == expected


def main():
    parser = argparse.ArgumentParser(
        description="Check racecard source features against prepared features."
    )
    parser.add_argument("history_database", type=Path)
    parser.add_argument("feature_database", type=Path)
    args = parser.parse_args()

    history = open_database(args.history_database.resolve())

    try:
        prepared = open_database(args.feature_database.resolve())

        try:
            total_runners = 0
            total_comparisons = 0
            columns = ", ".join(f'"{name}"' for name in FIELDS)

            for race_key in CASES:
                race = history.execute(
                    """
                    SELECT distance_furlongs, runner_count
                    FROM races
                    WHERE date = ? AND course = ? AND off = ?
                    """,
                    race_key,
                ).fetchone()

                if race is None or race[0] is None:
                    raise ValueError(f"Missing race or distance: {race_key}")

                distance, expected_count = race

                runners = history.execute(
                    """
                    SELECT horse, age
                    FROM runners
                    WHERE date = ? AND course = ? AND off = ?
                    ORDER BY horse
                    """,
                    race_key,
                ).fetchall()

                expected_rows = prepared.execute(
                    f"""
                    SELECT horse, {columns}
                    FROM features
                    WHERE date = ? AND course = ? AND off = ?
                    ORDER BY horse
                    """,
                    race_key,
                ).fetchall()

                expected_by_horse = {
                    row[0]: dict(zip(FIELDS, row[1:], strict=True))
                    for row in expected_rows
                }

                if (
                    len(runners) != expected_count
                    or len(expected_rows) != expected_count
                    or {horse for horse, _ in runners}
                    != set(expected_by_horse)
                ):
                    raise ValueError(f"Runner mismatch: {race_key}")

                for horse, age in runners:
                    actual, _ = build_runner_features(
                        history,
                        horse,
                        age,
                        race_key[0],
                        distance,
                    )

                    for field in FIELDS:
                        expected = expected_by_horse[horse][field]

                        if not values_match(actual[field], expected):
                            raise ValueError(
                                f"{race_key} | {horse} | {field}: "
                                f"inference={actual[field]!r}, "
                                f"prepared={expected!r}"
                            )

                        total_comparisons += 1

                    total_runners += 1

                print(
                    f"PASS: {race_key} | "
                    f"{len(runners)} runners | "
                    f"{len(runners) * len(FIELDS)} feature comparisons"
                )

            print(f"\nMatched runners: {total_runners}")
            print(f"Feature comparisons passed: {total_comparisons}")

        finally:
            prepared.close()
    finally:
        history.close()


if __name__ == "__main__":
    main()