import argparse
import numpy as np

import json
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from inspect_data import open_database
from race_metrics import evaluate_race_scores


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
    args = parser.parse_args()

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

        training_rows = connection.execute(
            """
            SELECT prior_starts, prior_wins, prior_win_rate, won
            FROM features
            WHERE split = 'train'
            ORDER BY date, course, off, horse
            """
        ).fetchall()

        training_array = np.array(training_rows, dtype=float)

        X_train = training_array[:, :3]
        y_train = training_array[:, 3].astype(int)

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Missing input values: {np.isnan(X_train).sum():,}")

        imputer = SimpleImputer(
            strategy="constant",
            fill_value=0.0,
            add_indicator=True,
        )

        X_train_imputed = imputer.fit_transform(X_train)

        print(f"After imputation: {X_train_imputed.shape}")
        print(f"Remaining missing values: {np.isnan(X_train_imputed).sum()}")
        print(
            f"Rows flagged as missing history: "
            f"{int(X_train_imputed[:, -1].sum()):,}"
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)

        print(
            "Scaled column means:",
            np.round(X_train_scaled.mean(axis=0), 6),
        )
        print(
            "Scaled column standard deviations:",
            np.round(X_train_scaled.std(axis=0), 6),
        )

        model = LogisticRegression(
            solver="lbfgs",
            C=1.0,
            max_iter=1000,
        )

        print("Fitting logistic regression...", flush=True)
        model.fit(X_train_scaled, y_train)

        feature_names = [
            "prior_starts",
            "prior_wins",
            "prior_win_rate",
            "missing_history",
        ]

        print(f"Iterations used: {model.n_iter_[0]}")
        print(f"Intercept: {model.intercept_[0]:.6f}")

        for name, weight in zip(feature_names, model.coef_[0]):
            print(f"{name}: {weight:.6f}")

        validation_rows = connection.execute(
            """
            SELECT
                date, course, off, horse,
                prior_starts, prior_wins, prior_win_rate, won
            FROM features
            WHERE split = 'validation'
            ORDER BY date, course, off, horse
            """
        ).fetchall()

        X_validation = np.array(
            [row[4:7] for row in validation_rows],
            dtype=float,
        )
        y_validation = np.array(
            [row[7] for row in validation_rows],
            dtype=int,
        )

        X_validation_imputed = imputer.transform(X_validation)
        X_validation_scaled = scaler.transform(X_validation_imputed)

        print(f"Validation inputs: {X_validation_scaled.shape}")
        print(f"Validation winners: {y_validation.sum():,}")

        win_column = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            X_validation_scaled
        )[:, win_column]

        validation_scores = {}

        for row, probability in zip(validation_rows, probabilities):
            date, course, off, horse = row[:4]
            won = row[7]
            race_key = (date, course, off)

            winner_marker = "1" if won == 1 else "0"

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
            }

            args.report.parent.mkdir(parents=True, exist_ok=True)

            with args.report.open("x", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
                file.write("\n")

            print(f"Report saved to: {args.report}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()