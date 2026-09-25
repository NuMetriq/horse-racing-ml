import argparse
import json
import pickle
import numpy as np
from datetime import date as calendar_date
from pathlib import Path

from distance import parse_distance_furlongs
from inspect_data import open_database
from prior_form import (
    calculate_features_as_of,
    calculate_previous_distance,
)
from racecard_inputs import load_racecard
from feature_transforms import (
    encode_relative_finish_age_distance_change,
)
from calibrate_probabilities import adjust_probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Build boosting features for a supplied racecard."
    )
    parser.add_argument("history_database", type=Path)
    parser.add_argument("--date", required=True)
    parser.add_argument(
        "--distance",
        required=True,
        help="Current race distance, for example 6f or 1m2f",
    )
    parser.add_argument("--runners-file", type=Path, required=True)
    parser.add_argument(
        "--features-output",
        type=Path,
        help="Save assembled source features to a new JSON file",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Trusted saved boosting model",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save prediction details to a new JSON file",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Frozen calibration JSON for the initial boosting model",
    )
    args = parser.parse_args()

    if args.report is not None and args.report.exists():
        parser.error(f"Report already exists: {args.report}")

    if args.features_output is not None and args.features_output.exists():
        parser.error(f"Output already exists: {args.features_output}")

    race_date = calendar_date.fromisoformat(args.date).isoformat()
    distance = parse_distance_furlongs(args.distance)

    if distance is None:
        parser.error("Race distance must not be blank")

    runners = load_racecard(args.runners_file)
    features_by_horse = {}

    connection = open_database(args.history_database.resolve())

    try:
        for runner in runners:
            horse = runner["horse"]

            rows = connection.execute(
                """
                SELECT
                    r.date, r.course, r.off, r.finish_position,
                    races.runner_count, races.distance_furlongs
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

            history = [row[:4] for row in rows]
            runner_counts = [row[4] for row in rows]
            distances = [row[5] for row in rows]

            features = calculate_features_as_of(
                history,
                runner_counts,
                race_date,
            )

            # Append a target-date placeholder to retrieve the distance
            # known before that date using the existing historical rule.
            previous_distance = calculate_previous_distance(
                history + [(race_date, "", "", None)],
                distances + [None],
            )[-1]

            features["age"] = runner["age"]
            features["distance_change_furlongs"] = (
                distance - previous_distance
                if previous_distance is not None
                else None
            )

            features_by_horse[horse] = features

            print(
                f"{horse} | Prior starts: {features['prior_starts']} | "
                f"Age: {features['age']} | "
                f"Previous distance: {previous_distance} | "
                f"Distance change: "
                f"{features['distance_change_furlongs']}"
            )
    finally:
        connection.close()

    print(f"Race date: {race_date}")
    print(f"Current distance: {distance} furlongs")
    print(f"Feature rows assembled: {len(features_by_horse)}")

    if args.features_output is not None:
        args.features_output.parent.mkdir(parents=True, exist_ok=True)

        with args.features_output.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "race_date": race_date,
                    "distance_furlongs": distance,
                    "features_by_horse": features_by_horse,
                },
                file,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")

        print(f"Features saved to: {args.features_output}")

    with args.model.open("rb") as file:
        bundle = pickle.load(file)

    source_features = [
        "prior_starts",
        "prior_wins",
        "prior_win_rate",
        "days_since_run",
        "previous_position",
        "previous_runner_count",
        "age",
        "distance_change_furlongs",
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
        "age",
        "distance_change_furlongs",
    ]

    if (
        bundle.get("input_encoding")
        != "relative_finish_age_distance_change_v1"
        or bundle.get("source_features") != source_features
        or bundle.get("input_features") != expected_inputs
    ):
        raise ValueError("Saved model has an unexpected feature schema")

    if (
        "history_window_days" not in bundle
        or bundle["history_window_days"] is not None
    ):
        raise ValueError("This predictor requires an all-history model")

    if bundle.get("probability_normalization") != "divide by race total":
        raise ValueError("Unexpected probability normalization")

    training_end = bundle.get("training_end_exclusive")
    if training_end is None:
        raise ValueError("Saved model is missing its training cutoff")

    if calendar_date.fromisoformat(race_date) < (
        calendar_date.fromisoformat(training_end)
    ):
        raise ValueError("Prediction date overlaps the training period")

    horses = [runner["horse"] for runner in runners]

    X = np.array(
        [
            encode_relative_finish_age_distance_change(
                tuple(
                    features_by_horse[horse][name]
                    for name in source_features
                )
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

    if (
        not np.all(np.isfinite(scores))
        or np.any(scores <= 0)
        or np.any(scores > 1)
    ):
        raise ValueError("Model returned invalid win scores")

    probabilities = scores / scores.sum()

    uncalibrated_probabilities = probabilities.copy()
    calibration_details = None

    if args.calibration is not None:
        with args.calibration.open(encoding="utf-8") as file:
            calibration = json.load(file)

        if (
            calibration.get("method") != "race_probability_power"
            or calibration.get("base_model")
            != "initial_hist_gradient_boosting"
            or calibration.get("feature_set")
            != "relative_finish_age_distance_change"
        ):
            raise ValueError("Unexpected calibration metadata")

        fitting_years = calibration["fitting_years"]
        if (
            not fitting_years
            or calendar_date.fromisoformat(race_date).year
            <= max(fitting_years)
        ):
            raise ValueError(
                "Prediction date must follow the calibration-fitting years"
            )

        # Reject the known tuned configuration or other changed settings.
        expected_settings = {
            "learning_rate": 0.05,
            "max_iter": 200,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
            "early_stopping": False,
            "random_state": 42,
        }
        classifier = pipeline.named_steps["classifier"]
        actual_settings = classifier.get_params()

        if any(
            actual_settings.get(name) != value
            for name, value in expected_settings.items()
        ):
            raise ValueError(
                "Calibration requires the initial boosting settings"
            )

        gamma = float(calibration["gamma"])
        race_key = (race_date, "", "")

        adjusted = adjust_probabilities(
            {
                race_key: [
                    (horse, None, float(probability))
                    for horse, probability in zip(
                        horses, probabilities, strict=True
                    )
                ]
            },
            gamma,
        )

        probabilities = np.array(
            [probability for _, _, probability in adjusted[race_key]],
            dtype=float,
        )

        calibration_details = {
            "path": str(args.calibration.resolve()),
            "method": calibration["method"],
            "gamma": gamma,
            "fitting_years": fitting_years,
        }

    uncalibrated_by_horse = dict(
        zip(horses, uncalibrated_probabilities, strict=True)
    )

    ranked_runners = sorted(
        zip(horses, probabilities, strict=True),
        key=lambda item: (-item[1], item[0]),
    )

    label = (
        "Calibrated"
        if calibration_details is not None
        else "Uncalibrated"
    )
    print(f"\n{label} win probabilities:")

    for horse, probability in ranked_runners:
        print(f"{horse} | {probability:.6%}")

    print(f"Probability total: {probabilities.sum():.12f}")

    if args.report is not None:
        report = {
            "race_date": race_date,
            "distance_furlongs": distance,
            "model_path": str(args.model.resolve()),
            "history_database": str(args.history_database.resolve()),
            "input_encoding": bundle["input_encoding"],
            "calibrated": calibration_details is not None,
            "calibration": calibration_details,
            "predictions": [
                {
                    "horse": horse,
                    "win_probability": float(probability),
                    "source_features": features_by_horse[horse],
                    "uncalibrated_win_probability": float(
                        uncalibrated_by_horse[horse]
                    ),
                }
                for horse, probability in ranked_runners
            ],
        }

        args.report.parent.mkdir(parents=True, exist_ok=True)

        with args.report.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, allow_nan=False)
            file.write("\n")

        print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()