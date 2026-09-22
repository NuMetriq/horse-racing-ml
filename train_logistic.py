import argparse
import numpy as np

import json
from pathlib import Path

import pickle
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from feature_transforms import log_days_since_run
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from inspect_data import open_database
from race_metrics import evaluate_race_scores
from feature_transforms import encode_previous_position, encode_previous_position_field, encode_previous_relative_finish, encode_relative_finish_age


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression racing model."
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
    args = parser.parse_args()

    if args.model_output is not None and args.model_output.exists():
        parser.error(f"Model output already exists: {args.model_output}")

    connection = open_database(args.database.resolve())

    try:
        counts = connection.execute(
            """
            SELECT split, COUNT(*), SUM(won)
            FROM features
            GROUP BY split
            ORDER BY split
            """
        ).fetchall()

        for split, runners, winners in counts:
            print(
                f"{split}: {runners:,} runners | "
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
            WHERE split = 'train'
            """
        ).fetchone()

        print(f"Training gaps missing: {missing:,} of {total:,}")
        print(f"Observed gap range: {minimum} to {maximum} days")
        print(f"Mean observed gap: {average:.1f} days")

        training_rows = connection.execute(
            """
            SELECT prior_starts, prior_wins, prior_win_rate,
                   days_since_run, previous_position,
                   previous_runner_count, age, won
            FROM features
            WHERE split = 'train'
            ORDER BY date, course, off, horse
            """
        ).fetchall()

        X_train = np.array(
            [
                encode_relative_finish_age(row[:7])
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

        pipeline = Pipeline(
            [
                (
                    "log_gap",
                    FunctionTransformer(log_days_since_run, validate=False),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=0.0,
                        add_indicator=True,
                    ),
                ),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        C=1.0,
                        max_iter=1000,
                    ),
                ),
            ]
        )

        print("Fitting logistic regression pipeline...", flush=True)
        pipeline.fit(X_train, y_train)

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["classifier"]

        feature_names = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "log1p_days_since_run",
            "previous_finish_position",
            "previous_result_was_code",
            "previous_runner_count",
            "previous_relative_finish",
            "age",
            "missing_win_rate",
            "missing_days_since_run",
            "missing_previous_finish_position",
            "missing_previous_runner_count",
            "missing_previous_relative_finish",
            "missing_age"
        ]

        print(f"Coefficient labels: {len(feature_names)}")
        print(f"Model coefficients: {len(model.coef_[0])}")
        print(
            "Training ages encoded as missing: "
            f"{np.isnan(X_train[:, 8]).sum():,}"
        )
        
        if len(feature_names) != len(model.coef_[0]):
            raise ValueError("Coefficient labels do not match model inputs")

        print(f"Iterations used: {model.n_iter_[0]}")
        print(f"Intercept: {model.intercept_[0]:.6f}")

        for name, coefficient in zip(
            feature_names, model.coef_[0], strict=True
        ):
            print(f"{name}: {coefficient:.6f}")

        validation_rows = connection.execute(
            """
            SELECT date, course, off, horse,
                   prior_starts, prior_wins, prior_win_rate,
                   days_since_run, previous_position,
                   previous_runner_count, age, won
            FROM features
            WHERE split = 'validation'
            ORDER BY date, course, off, horse
            """
        ).fetchall()

        X_validation = np.array(
            [
                encode_relative_finish_age(row[4:11])
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

        for row, probability in zip(validation_rows, probabilities):
            date, course, off, horse = row[:4]
            won = row[9]
            race_key = (date, course, off)

            winner_marker = "1" if row[-1] == 1 else "0"

            validation_scores.setdefault(race_key, []).append(
                (horse, winner_marker, float(probability))
            )

        model_loss, uniform_loss = evaluate_race_scores(
            validation_scores
        )

        print(f"Validation races evaluated: {len(validation_scores):,}")
        print(f"Logistic race log loss: {model_loss:.6f}")
        print(f"Uniform race log loss: {uniform_loss:.6f}")
        print(f"Improvement over uniform: {uniform_loss - model_loss:.6f}")

        if args.report is not None:
            report = {
                "model": "logistic_regression",
                "database": str(args.database.resolve()),
                "features": feature_names,
                "imputation": "constant zero with missing indicator",
                "scaling": "standard scaler fitted on training",
                "solver": model.solver,
                "C": model.C,
                "max_iter": model.max_iter,
                "iterations_used": int(model.n_iter_[0]),
                "intercept": float(model.intercept_[0]),
                "coefficients": model.coef_[0].tolist(),
                "validation_start": "2024-01-01",
                "validation_end_exclusive": "2025-01-01",
                "validation_races": len(validation_scores),
                "probability_normalization": "divide by race total",
                "model_log_loss": model_loss,
                "uniform_log_loss": uniform_loss,
                "improvement": uniform_loss - model_loss,
                "input_encoding": "relative_finish_age_v1",
            }

            args.report.parent.mkdir(parents=True, exist_ok=True)

            with args.report.open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
                file.write("\n")

            print(f"Report saved to: {args.report}")

        if args.model_output is not None:
            bundle = {
                "pipeline": pipeline,
                "source_features": [
                    "prior_starts",
                    "prior_wins",
                    "prior_win_rate",
                    "days_since_run",
                    "previous_position",
                    "previous_runner_count",
                    "age",
                ],
                "input_features": [
                    "prior_starts",
                    "prior_wins",
                    "prior_win_rate",
                    "days_since_run",
                    "previous_finish_position",
                    "previous_result_was_code",
                    "previous_runner_count",
                    "previous_relative_finish",
                    "age",
                ],
                "input_encoding": "relative_finish_age_v1",
                "history_window_days": None,
                "training_end_exclusive": "2024-01-01",
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