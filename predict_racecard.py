import argparse
import json
from datetime import date
from pathlib import Path
import pickle
import numpy as np

from inspect_data import open_database
from prior_form import calculate_features_as_of
from feature_transforms import encode_previous_relative_finish


def main():
    parser = argparse.ArgumentParser(
        description="Predict a supplied racecard using earlier horse history."
    )
    parser.add_argument("history_database", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--date", required=True)

    runner_source = parser.add_mutually_exclusive_group(required=True)
    runner_source.add_argument(
        "--horse",
        action="append",
        help="Exact horse name; repeat for every runner",
    )
    runner_source.add_argument(
        "--runners-file",
        type=Path,
        help="UTF-8 text file containing one horse name per line",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save prediction details to a new JSON file",
    )

    args = parser.parse_args()
    if args.report is not None and args.report.exists():
        parser.error(f"Report already exists: {args.report}")

    race_date = date.fromisoformat(args.date).isoformat()

    if args.runners_file is not None:
        try:
            text = args.runners_file.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as error:
            parser.error(f"Cannot read runners file: {error}")

        horses = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        horses = [horse.strip() for horse in args.horse]

    if any(not horse for horse in horses):
        parser.error("Horse names must not be blank")

    if len(horses) < 2:
        parser.error("Supply at least two runners")

    if len(horses) != len(set(horses)):
        parser.error("Duplicate horse names are not allowed")

    print(f"Race date: {race_date}")
    print(f"Supplied runners: {len(horses)}")

    features_by_horse = {}
    connection = open_database(args.history_database.resolve())

    try:
        history_latest_date = connection.execute(
            "SELECT MAX(date) FROM runners"
        ).fetchone()[0]

        print(f"Latest recorded date in history database: {history_latest_date}")
        for horse in horses:
            rows = connection.execute(
                """
                SELECT
                    r.date, r.course, r.off, r.finish_position,
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

            features = calculate_features_as_of(
                [row[:4] for row in rows],
                [row[4] for row in rows],
                race_date,
            )
            features_by_horse[horse] = features

            print(
                f"{horse} | Prior starts: {features['prior_starts']} | "
                f"Prior wins: {features['prior_wins']} | "
                f"Days since run: {features['days_since_run']} | "
                f"Previous position: {features['previous_position']} | "
                f"Previous field: {features['previous_runner_count']}"
            )
    finally:
        connection.close()

    missing_history_count = sum(
        features["prior_starts"] == 0
        for features in features_by_horse.values()
    )

    print(
        f"\nRunners without earlier recorded history: "
        f"{missing_history_count} of {len(horses)} "
        f"({missing_history_count / len(horses):.1%})"
    )

    with args.model.open("rb") as file:
        bundle = pickle.load(file)

    source_features = [
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
        or bundle.get("source_features") != source_features
        or bundle.get("input_features") != expected_inputs
    ):
        raise ValueError("Expected the relative-finish model encoding")

    if (
        "history_window_days" not in bundle
        or bundle["history_window_days"] is not None
    ):
        raise ValueError("This script requires an all-history model")

    X = np.array(
        [
            encode_previous_relative_finish(
                tuple(features_by_horse[horse][name] for name in source_features)
            )
            for horse in horses
        ],
        dtype=float,
    )

    pipeline = bundle["pipeline"]

    if X.shape[1] != pipeline.n_features_in_:
        raise ValueError("Input count does not match the saved pipeline")

    win_column = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(X)[:, win_column]
    total_score = scores.sum()

    if (
        not np.all(np.isfinite(scores))
        or np.any(scores < 0)
        or not np.isfinite(total_score)
        or total_score <= 0
    ):
        raise ValueError("Model returned invalid scores")

    probabilities = scores / total_score

    ranked_runners = sorted(
        zip(horses, probabilities, strict=True),
        key=lambda item: (-item[1], item[0]),
    )

    print("\nPredicted win probabilities:")

    for horse, probability in ranked_runners:
        print(f"{horse} | {probability:.2%}")

    print(f"Probability total: {probabilities.sum():.6f}")

    if args.report is not None:
        report = {
            "race_date": race_date,
            "model_path": str(args.model.resolve()),
            "history_database": str(args.history_database.resolve()),
            "input_encoding": bundle["input_encoding"],
            "history_window_days": bundle["history_window_days"],
            "history_latest_date": history_latest_date,
            "runner_count": len(horses),
            "runners_without_history": missing_history_count,
            "probability_normalization": "divide by race total",
            "predictions": [
                {
                    "horse": horse,
                    "win_probability": float(probability),
                    "source_features": features_by_horse[horse],
                }
                for horse, probability in ranked_runners
            ],
        }

        args.report.parent.mkdir(parents=True, exist_ok=True)

        with args.report.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")

        print(f"Report saved to: {args.report.resolve()}")


if __name__ == "__main__":
    main()