import argparse
import json
import math
from itertools import groupby
from pathlib import Path

from inspect_data import open_database
from prior_form import calculate_prior_form, smoothed_win_rate
from race_metrics import evaluate_race_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build historical horse features."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Positive smoothing strength (default: 10)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Save run settings and metrics to a new JSON file",
    )
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be a finite number greater than zero")

    print(f"Smoothing strength: {args.alpha}")

    connection = open_database(args.database.resolve())

    try:
        training_wins, training_starts = connection.execute(
            """
            SELECT
                SUM(
                    CASE WHEN finish_position = '1' THEN 1 ELSE 0 END
                ),
                COUNT(*)
            FROM runners
            WHERE date < '2024-01-01'
            """
        ).fetchone()

        reference_rate = training_wins / training_starts

        print(f"Training reference win rate: {reference_rate:.6f}")
        example_score = smoothed_win_rate(
            1, 4, reference_rate, alpha=args.alpha
        )
        print(f"Example smoothed score: {example_score:.6f}")

        rows = connection.execute(
            """
            SELECT horse, date, course, off, finish_position
            FROM runners
            WHERE date < '2025-01-01'
            ORDER BY horse, date, course, off
            """
        )

        horse_count = 0
        feature_count = 0

        split_totals = {"train": 0, "validation": 0}
        missing_history = {"train": 0, "validation": 0}

        validation_scores = {}

        for horse, records in groupby(rows, key=lambda row: row[0]):
            history = [row[1:] for row in records]
            features = calculate_prior_form(history)
            for date, course, off, position, starts, wins, rate in features:
                split = "train" if date < "2024-01-01" else "validation"
                split_totals[split] += 1

                if starts == 0:
                    missing_history[split] += 1

                if split == "validation":
                    race_key = (date, course, off)
                    score = smoothed_win_rate(
                        wins, starts, reference_rate, alpha=args.alpha
                    )

                    validation_scores.setdefault(race_key, []).append(
                        (horse, position, score)
                    )

            horse_count += 1
            feature_count += len(features)

        print(f"Horse names processed: {horse_count:,}")
        print(f"Feature rows calculated: {feature_count:,}")
        for split in ("train", "validation"):
            missing = missing_history[split]
            total = split_totals[split]
            print(
                f"{split}: {missing:,} of {total:,} runners "
                f"have no prior history ({missing / total:.1%})"
            )
        print(f"Validation races scored: {len(validation_scores):,}")

        model_loss, uniform_loss = evaluate_race_scores(
            validation_scores
        )

        print(f"Validation smoothed-form log loss: {model_loss:.6f}")
        print(f"Validation uniform log loss: {uniform_loss:.6f}")
        print(f"Improvement over uniform: {uniform_loss - model_loss:.6f}")

        if args.report is not None:
            report = {
                "model": "smoothed_horse_win_rate",
                "database": str(args.database.resolve()),
                "alpha": args.alpha,
                "training_reference_rate": reference_rate,
                "validation_start": "2024-01-01",
                "validation_end_exclusive": "2025-01-01",
                "validation_races": len(validation_scores),
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