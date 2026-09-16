import argparse
import pickle

import numpy as np

from pathlib import Path
from inspect_data import open_database
from feature_transforms import encode_previous_relative_finish


def main():
    parser = argparse.ArgumentParser(
        description="Predict one race from a prepared feature database."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--date", required=True, help="Race date: YYYY-MM-DD")
    parser.add_argument("--course", required=True, help="Exact course name")
    parser.add_argument("--off", required=True, help="Exact recorded off time")

    args = parser.parse_args()

    print(f"Database: {args.database}")
    print(f"Model: {args.model}")
    print(f"Race: {args.date} | {args.course} | {args.off}")

    connection = open_database(args.database.resolve())

    try:
        rows = connection.execute(
            """
            SELECT
                horse,
                prior_starts,
                prior_wins,
                prior_win_rate,
                days_since_run,
                previous_position,
                previous_runner_count
            FROM features
            WHERE date = ? AND course = ? AND off = ?
            ORDER BY horse
            """,
            (args.date, args.course, args.off),
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError("No runners found for the requested race")

    if len(rows) < 2:
        raise ValueError("Expected at least two runners")

    print(f"Runners loaded: {len(rows)}")

    for row in rows:
        print(
            f"{row[0]} | Prior starts: {row[1]} | "
            f"Previous position: {row[5]} | "
            f"Previous field size: {row[6]}"
        )

    with args.model.open("rb") as file:
        bundle = pickle.load(file)

    expected_sources = [
        "prior_starts",
        "prior_wins",
        "prior_win_rate",
        "days_since_run",
        "previous_position",
        "previous_runner_count",
    ]
    expected_inputs = [
        "prior_starts",
        "prior_wins",
        "prior_win_rate",
        "days_since_run",
        "previous_finish_position",
        "previous_result_was_code",
        "previous_runner_count",
        "previous_relative_finish",
    ]

    if (
        bundle.get("input_encoding") != "previous_relative_finish_v1"
        or bundle.get("source_features") != expected_sources
        or bundle.get("input_features") != expected_inputs
    ):
        raise ValueError("Expected the relative-finish model encoding")

    pipeline = bundle["pipeline"]

    X = np.array(
        [
            encode_previous_relative_finish(row[1:])
            for row in rows
        ],
        dtype=float,
    )

    if X.shape[1] != pipeline.n_features_in_:
        raise ValueError("Input count does not match the saved pipeline")

    win_column = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(X)[:, win_column]

    total_score = scores.sum()

    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or total_score <= 0
    ):
        raise ValueError("Model returned invalid scores")

    probabilities = scores / total_score

    ranked_runners = sorted(
        zip(rows, probabilities, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nPredicted win probabilities:")

    for row, probability in ranked_runners:
        print(f"{row[0]} | {probability:.2%}")

    print(f"Probability total: {probabilities.sum():.6f}")


if __name__ == "__main__":
    main()