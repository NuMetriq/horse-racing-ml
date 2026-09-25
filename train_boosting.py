import argparse
import numpy as np

import json
from pathlib import Path

import pickle
import sklearn
from datetime import date as calendar_date

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from inspect_data import open_database
from race_metrics import evaluate_race_scores
from feature_transforms import (
    encode_previous_position,
    encode_previous_position_field,
    encode_previous_relative_finish,
    encode_relative_finish_age,
    encode_relative_finish_age_distance_change,
    encode_relative_finish_age_distance_change_field,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a histogram gradient boosting racing model."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Save settings and metrics to a new JSON file",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        help="Save the fitted pipeline to a new file",
    )
    parser.add_argument(
        "--train-end",
        default="2024-01-01",
        help="Exclusive training end and inclusive evaluation start",
    )
    parser.add_argument(
        "--evaluation-end",
        default="2025-01-01",
        help="Exclusive evaluation end",
    )
    parser.add_argument(
        "--feature-set",
        choices=(
            "relative_finish",
            "relative_finish_age",
            "relative_finish_age_distance_change",
            "relative_finish_age_distance_change_field",
        ),
        default="relative_finish_age",
        help="Model inputs to use (default: relative_finish_age)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Number of boosting iterations (default: 200)",
    )
    parser.add_argument(
        "--parameters",
        type=Path,
        help="JSON parameter overrides; takes precedence over --max-iter",
    )
    args = parser.parse_args()

    if args.max_iter < 1:
        parser.error("--max-iter must be at least 1")

    try:
        train_end = calendar_date.fromisoformat(args.train_end)
        evaluation_end = calendar_date.fromisoformat(args.evaluation_end)
    except ValueError:
        parser.error("Dates must be valid ISO dates, such as 2024-01-01")

    if train_end <= calendar_date(2015, 1, 1):
        parser.error("--train-end must be after 2015-01-01")

    if evaluation_end <= train_end:
        parser.error("--evaluation-end must be after --train-end")

    if evaluation_end > calendar_date(2025, 1, 1):
        parser.error("Development evaluation must end by 2025-01-01")

    args.train_end = train_end.isoformat()
    args.evaluation_end = evaluation_end.isoformat()

    if args.model_output is not None and args.model_output.exists():
        parser.error(f"Model output already exists: {args.model_output}")

    connection = open_database(args.database.resolve())

    try:
        periods = [
            ("train", "2015-01-01", args.train_end),
            ("evaluation", args.train_end, args.evaluation_end),
        ]

        for label, start, end in periods:
            runners, winners = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(won), 0)
                FROM features
                WHERE date >= ? AND date < ?
                """,
                (start, end),
            ).fetchone()

            if runners == 0:
                raise ValueError(f"No rows in the {label} period")

            print(
                f"{label} [{start}, {end}): "
                f"{runners:,} runners | "
                f"{winners:,} recorded winners"
            )

        total, missing, minimum, maximum, average = connection.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN days_since_run IS NULL THEN 1 ELSE 0 END),
                MIN(days_since_run),
                MAX(days_since_run),
                AVG(days_since_run)
            FROM features
            WHERE date >= ? AND date < ?
            """,
            ("2015-01-01", args.train_end),
        ).fetchone()

        print(f"Training gaps missing: {missing:,} of {total:,}")
        print(f"Observed gap range: {minimum} to {maximum} days")
        print(f"Mean observed gap: {average:.1f} days")

        source_features = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "days_since_run",
            "previous_position",
            "previous_runner_count",
        ]

        input_features = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "days_since_run",
            "previous_finish_position",
            "previous_result_was_code",
            "previous_runner_count",
            "previous_relative_finish",
        ]

        if args.feature_set == "relative_finish":
            encoder = encode_previous_relative_finish
            input_encoding = "previous_relative_finish_v1"
        else:
            source_features.append("age")
            input_features.append("age")

            if args.feature_set == "relative_finish_age":
                encoder = encode_relative_finish_age
                input_encoding = "relative_finish_age_v1"
            else:
                source_features.append("distance_change_furlongs")
                input_features.append("distance_change_furlongs")

                if args.feature_set == "relative_finish_age_distance_change":
                    encoder = encode_relative_finish_age_distance_change
                    input_encoding = "relative_finish_age_distance_change_v1"
                else:
                    source_features.append("current_runner_count")
                    input_features.append("current_runner_count")
                    encoder = (
                        encode_relative_finish_age_distance_change_field
                    )
                    input_encoding = (
                        "relative_finish_age_distance_change_field_v1"
                    )

        source_count = len(source_features)
        feature_columns = ", ".join(f'"{name}"' for name in source_features)
        print(f"Feature set: {args.feature_set}")

        training_rows = connection.execute(
            f"""
            SELECT {feature_columns}, won
            FROM features
            WHERE date >= ? AND date < ?
            ORDER BY date, course, off, horse
            """,
            ("2015-01-01", args.train_end),
        ).fetchall()

        if not training_rows:
            raise ValueError("No training rows in the selected date range")

        X_train = np.array(
            [
                encoder(row[:source_count])
                for row in training_rows
            ],
            dtype=float,
        )

        y_train = np.array(
            [row[-1] for row in training_rows],
            dtype=int,
        )

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Missing input values: {np.isnan(X_train).sum():,}")

        settings = {
            "learning_rate": 0.05,
            "max_iter": args.max_iter,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "l2_regularization": 1.0,
        }

        if args.parameters is not None:
            with args.parameters.open(encoding="utf-8") as file:
                overrides = json.load(file)

            if not isinstance(overrides, dict):
                raise ValueError("Parameter file must contain a JSON object")

            unknown = set(overrides) - set(settings)
            if unknown:
                raise ValueError(
                    f"Unsupported parameters: {sorted(unknown)}"
                )

            settings.update(overrides)

        print("Boosting settings:")
        print(json.dumps(settings, indent=2))

        pipeline = Pipeline(
            [
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        loss="log_loss",
                        early_stopping=False,
                        random_state=42,
                        **settings,
                    ),
                ),
            ]
        )

        print("Fitting histogram gradient boosting...", flush=True)
        pipeline.fit(X_train, y_train)

        model = pipeline.named_steps["classifier"]
        feature_names = input_features.copy()

        print(f"Model input features: {model.n_features_in_}")
        print(f"Boosting iterations used: {model.n_iter_}")

        if len(feature_names) != model.n_features_in_:
            raise ValueError("Feature labels do not match model inputs")

        print("Input features:")
        for name in feature_names:
            print(f"  {name}")

        validation_rows = connection.execute(
            f"""
            SELECT date, course, off, horse,
                   {feature_columns}, won
            FROM features
            WHERE date >= ? AND date < ?
            ORDER BY date, course, off, horse
            """,
            (args.train_end, args.evaluation_end),
        ).fetchall()

        if not validation_rows:
            raise ValueError("No evaluation rows in the selected date range")

        X_validation = np.array(
            [
                encoder(row[4:4 + source_count])
                for row in validation_rows
            ],
            dtype=float,
        )

        y_validation = np.array(
            [row[-1] for row in validation_rows],
            dtype=int,
        )

        print(f"Validation inputs before preprocessing: {X_validation.shape}")
        print(f"Validation winners: {y_validation.sum():,}")

        win_column = list(pipeline.classes_).index(1)
        probabilities = pipeline.predict_proba(
            X_validation
        )[:, win_column]

        validation_scores = {}

        for row, probability in zip(
            validation_rows, probabilities, strict=True
        ):
            date, course, off, horse = row[:4]
            race_key = (date, course, off)
            winner_marker = "1" if row[-1] == 1 else "0"

            validation_scores.setdefault(race_key, []).append(
                (horse, winner_marker, float(probability))
            )

        model_loss, uniform_loss = evaluate_race_scores(
            validation_scores
        )

        print(f"Validation races evaluated: {len(validation_scores):,}")
        print(f"Boosting race log loss: {model_loss:.6f}")
        print(f"Uniform race log loss: {uniform_loss:.6f}")
        print(f"Improvement over uniform: {uniform_loss - model_loss:.6f}")

        if args.report is not None:
            report = {
            "model": "hist_gradient_boosting",
            "database": str(args.database.resolve()),
            "features": feature_names,
            "source_features": source_features,
            "input_encoding": input_encoding,
            "feature_set": args.feature_set,
            "preprocessing": "encoded inputs; no scaling or log transformation",
            "missing_values": "native handling by histogram gradient boosting",
            "parameters": model.get_params(),
            "iterations_used": int(model.n_iter_),
            "sklearn_version": sklearn.__version__,
            "training_start": "2015-01-01",
            "training_end_exclusive": args.train_end,
            "validation_start": args.train_end,
            "validation_end_exclusive": args.evaluation_end,
            "validation_races": len(validation_scores),
            "probability_normalization": "divide by race total",
            "model_log_loss": model_loss,
            "uniform_log_loss": uniform_loss,
            "improvement": uniform_loss - model_loss,
        }

            args.report.parent.mkdir(parents=True, exist_ok=True)

            with args.report.open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
                file.write("\n")

            print(f"Report saved to: {args.report}")

        if args.model_output is not None:
            bundle = {
                "pipeline": pipeline,
                "source_features": source_features,
                "input_features": input_features,
                "input_encoding": input_encoding,
                "history_window_days": None,
                "training_start": "2015-01-01",
                "training_end_exclusive": args.train_end,
                "sklearn_version": sklearn.__version__,
                "probability_normalization": "divide by race total",
            }

            args.model_output.parent.mkdir(parents=True, exist_ok=True)

            with args.model_output.open("xb") as file:
                pickle.dump(bundle, file)

            print(f"Model saved to: {args.model_output}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()