import argparse
import pickle
import csv

import numpy as np

from inspect_data import open_database
from pathlib import Path
from race_metrics import evaluate_race_scores
from feature_transforms import encode_previous_position, encode_previous_position_field, encode_previous_relative_finish


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on validation races."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        help="Destination CSV for validation runner predictions",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Dataset split to evaluate (default: validation)",
    )

    args = parser.parse_args()

    if (
        args.predictions_output is not None
        and args.predictions_output.exists()
    ):
        raise FileExistsError(
            f"Output already exists: {args.predictions_output}"
        )

    with args.model.open("rb") as file:
        bundle = pickle.load(file)

    pipeline = bundle["pipeline"]
    input_features = bundle["input_features"]
    encoding = bundle.get("input_encoding")

    if encoding is None:
        # Older models receive numeric database columns directly.
        allowed_features = {
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "days_since_run",
        }

        if (
            not input_features
            or len(input_features) != len(set(input_features))
            or any(name not in allowed_features for name in input_features)
        ):
            raise ValueError("Invalid input-feature list in saved model")

        source_features = input_features

    elif encoding == "previous_position_v1":
        source_features = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "days_since_run",
            "previous_position",
        ]

        expected_inputs = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "days_since_run",
            "previous_finish_position",
            "previous_result_was_code",
        ]

        if (
            bundle.get("source_features") != source_features
            or input_features != expected_inputs
        ):
            raise ValueError(
                "Feature order does not match previous_position_v1"
            )

    elif encoding == "previous_position_field_v1":
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
        ]

        if (
            bundle.get("source_features") != source_features
            or input_features != expected_inputs
        ):
            raise ValueError(
                "Feature order does not match previous_position_field_v1"
            )

    elif encoding == "previous_relative_finish_v1":
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
            bundle.get("source_features") != source_features
            or input_features != expected_inputs
        ):
            raise ValueError(
                "Feature order does not match previous_relative_finish_v1"
            )

    else:
        raise ValueError(f"Unknown input encoding: {encoding!r}")

    feature_columns = ", ".join(
        f'"{name}"' for name in source_features
    )

    connection = open_database(args.database.resolve())

    try:
        rows = connection.execute(
            f"""
            SELECT date, course, off, horse,
                   {feature_columns}, won
            FROM features
            WHERE split = ?
            ORDER BY date, course, off, horse
            """,
            (args.split,),
        ).fetchall()
    finally:
        connection.close()

    if encoding == "previous_position_v1":
        X = np.array(
            [
                (*row[4:8], *encode_previous_position(row[8]))
                for row in rows
            ],
            dtype=float,
        )

    elif encoding == "previous_position_field_v1":
        X = np.array(
            [
                encode_previous_position_field(row[4:10])
                for row in rows
            ],
            dtype=float,
        )

    elif encoding == "previous_relative_finish_v1":
        X = np.array(
            [
                encode_previous_relative_finish(row[4:10])
                for row in rows
            ],
            dtype=float,
        )

    else:
        X = np.array(
            [row[4:-1] for row in rows],
            dtype=float,
        )
    
    win_column = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(X)[:, win_column]

    race_scores = {}

    for row, probability in zip(rows, probabilities):
        race_key = tuple(row[:3])
        horse = row[3]
        winner_marker = "1" if row[-1] == 1 else "0"

        race_scores.setdefault(race_key, []).append(
            (horse, winner_marker, float(probability))
        )

    normalized_probabilities = []
    outcomes = []

    for runners in race_scores.values():
        total_score = sum(score for _, _, score in runners)

        for horse, winner_marker, score in runners:
            normalized_probabilities.append(score / total_score)
            outcomes.append(int(winner_marker == "1"))

    normalized_probabilities = np.array(normalized_probabilities)
    outcomes = np.array(outcomes)

    print(f"Calibration runner count: {len(outcomes):,}")
    print(f"Mean predicted probability: {normalized_probabilities.mean():.6f}")
    print(f"Observed win rate: {outcomes.mean():.6f}")

    bin_edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (
                (normalized_probabilities >= lower)
                & (normalized_probabilities <= upper)
            )
        else:
            mask = (
                (normalized_probabilities >= lower)
                & (normalized_probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        predicted = normalized_probabilities[mask].mean()
        observed = outcomes[mask].mean()

        print(
            f"{lower:.0%}–{upper:.0%} | Runners: {count:,} | "
            f"Mean predicted: {predicted:.2%} | "
            f"Observed wins: {observed:.2%}"
        )

    model_loss, uniform_loss = evaluate_race_scores(race_scores)

    print(f"{args.split.capitalize()} races: {len(race_scores):,}")
    print(f"Saved-model race log loss: {model_loss:.6f}")
    print(f"Uniform race log loss: {uniform_loss:.6f}")

    field_size_groups = [
        ("2-7 runners", 2, 7),
        ("8-12 runners", 8, 12),
        ("13+ runners", 13, None),
    ]

    for label, minimum, maximum in field_size_groups:
        grouped_scores = {
            race_key: runners
            for race_key, runners in race_scores.items()
            if len(runners) >= minimum
            and (maximum is None or len(runners) <= maximum)
        }

        if not grouped_scores:
            continue

        group_model_loss, group_uniform_loss = evaluate_race_scores(
            grouped_scores
        )

        print(
            f"{label}: {len(grouped_scores):,} races | "
            f"Model: {group_model_loss:.6f} | "
            f"Uniform: {group_uniform_loss:.6f} | "
            f"Improvement: {group_uniform_loss - group_model_loss:.6f}"
        )

    if args.predictions_output is not None:
        output_path = args.predictions_output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        exported_rows = 0

        with output_path.open(
            "x", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "date",
                    "course",
                    "off",
                    "horse",
                    "win_probability",
                    "won",
                ]
            )

            for race_key, runners in race_scores.items():
                date, course, off = race_key
                total_score = sum(score for _, _, score in runners)

                for horse, marker, score in runners:
                    writer.writerow(
                        [
                            date,
                            course,
                            off,
                            horse,
                            score / total_score,
                            int(marker == "1"),
                        ]
                    )
                    exported_rows += 1

        print(
            f"Saved {exported_rows:,} prediction rows to: "
            f"{output_path.resolve()}"
        )


if __name__ == "__main__":
    main()
