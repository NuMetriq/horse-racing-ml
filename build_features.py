import argparse
import json
import math
import sqlite3
from itertools import groupby
from pathlib import Path

from inspect_data import open_database
from prior_form import (
    calculate_prior_form,
    calculate_recent_form,
    smoothed_win_rate,
    calculate_days_since_run,
    calculate_previous_position,
    calculate_previous_runner_count,
)
from race_metrics import evaluate_race_scores


def save_feature_table(feature_rows, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(output_path)

    try:
        with destination:
            destination.execute(
                """
                CREATE TABLE features (
                    date TEXT NOT NULL,
                    course TEXT NOT NULL,
                    off TEXT NOT NULL,
                    horse TEXT NOT NULL,
                    prior_starts INTEGER NOT NULL,
                    prior_wins INTEGER NOT NULL,
                    prior_win_rate REAL,
                    days_since_run INTEGER,
                    previous_position TEXT,
                    previous_runner_count INTEGER,
                    won INTEGER NOT NULL CHECK (won IN (0, 1)),
                    split TEXT NOT NULL
                        CHECK (split IN ('train', 'validation')),
                    PRIMARY KEY (date, course, off, horse)
                )
                """
            )

            destination.executemany(
                """
                INSERT INTO features (
                    date, course, off, horse,
                    prior_starts, prior_wins, prior_win_rate,
                    days_since_run, previous_position, previous_runner_count, won, split
                )
                VALUES (
                    :date, :course, :off, :horse,
                    :prior_starts, :prior_wins, :prior_win_rate,
                    :days_since_run, :previous_position, :previous_runner_count, :won, :split
                )
                """,
                feature_rows,
            )

            count = destination.execute(
                "SELECT COUNT(*) FROM features"
            ).fetchone()[0]

            if count != len(feature_rows):
                raise ValueError("Exported feature count does not match")

        print(f"Saved {count:,} feature rows to: {output_path}")
    finally:
        destination.close()

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
    parser.add_argument(
        "--window-days",
        type=int,
        default=None,
        help="Use this many prior days of history; omit for all history",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        help="Save the feature table to a new SQLite database",
    )
    args = parser.parse_args()
    if args.features_output is not None:
        args.features_output = args.features_output.resolve()

        if args.features_output.exists():
            parser.error(
                f"Feature output already exists: {args.features_output}"
            )
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be a finite number greater than zero")
    if args.window_days is not None and args.window_days <= 0:
        parser.error("--window-days must be greater than zero")

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
            SELECT
                r.horse, r.date, r.course, r.off,
                r.finish_position, races.runner_count
            FROM runners AS r
            JOIN races
                ON r.date = races.date
                AND r.course = races.course
                AND r.off = races.off
            WHERE r.date < '2025-01-01'
            ORDER BY r.horse, r.date, r.course, r.off
            """
        )

        horse_count = 0
        feature_count = 0

        split_totals = {"train": 0, "validation": 0}
        missing_history = {"train": 0, "validation": 0}

        validation_scores = {}
        validation_history_counts = {}
        feature_rows = []

        for horse, records in groupby(rows, key=lambda row: row[0]):
            horse_rows = list(records)
            history = [row[1:5] for row in horse_rows]
            runner_counts = [row[5] for row in horse_rows]
            previous_runner_counts = calculate_previous_runner_count(
                history, runner_counts
            )
            previous_positions = calculate_previous_position(history)
            gaps = calculate_days_since_run(history)
            if args.window_days is None:
                features = calculate_prior_form(history)
            else:
                features = calculate_recent_form(
                    history, window_days=args.window_days
                )

            for feature, gap, previous_position, previous_runner_count in zip(
                features,
                gaps,
                previous_positions,
                previous_runner_counts,
                strict=True,
            ):
                date, course, off, position, starts, wins, rate = feature
                split = "train" if date < "2024-01-01" else "validation"

                feature_rows.append(
                    {
                        "date": date,
                        "course": course,
                        "off": off,
                        "horse": horse,
                        "prior_starts": starts,
                        "prior_wins": wins,
                        "prior_win_rate": rate,
                        "won": int(position == "1"),
                        "split": split,
                        "days_since_run": gap,
                        "previous_position": previous_position,
                        "previous_runner_count": previous_runner_count,
                    }
                )

                split_totals[split] += 1

                if starts == 0:
                    missing_history[split] += 1

                if split == "validation":
                    race_key = (date, course, off)
                    counts = validation_history_counts.setdefault(
                        race_key, [0, 0]
                    )
                    counts[0] += 1
                    counts[1] += int(starts == 0)
                    score = smoothed_win_rate(
                        wins, starts, reference_rate, alpha=args.alpha
                    )

                    if race_key == ("2024-01-01", "Ascot (AUS)", "7:50"):
                        print(
                            f"{horse} | Prior starts: {starts} | "
                            f"Prior wins: {wins} | "
                            f"Smoothed score: {score:.8f}"
                        )

                    validation_scores.setdefault(race_key, []).append(
                        (horse, position, score)
                    )
                    

            horse_count += 1
            feature_count += len(features)

        print(f"Feature-table rows assembled: {len(feature_rows):,}")
        print("Example feature row:", feature_rows[0])

        if args.features_output is not None:
            save_feature_table(feature_rows, args.features_output)

        majority_missing = sum(
            missing > total / 2
            for total, missing in validation_history_counts.values()
        )

        print(
            f"Validation races with a majority lacking history: "
            f"{majority_missing:,} of {len(validation_history_counts):,}"
        )

        coverage_groups = {
            "Majority without history": {},
            "Other races": {},
        }

        for race_key, runners in validation_scores.items():
            total, missing = validation_history_counts[race_key]

            if missing > total / 2:
                group = "Majority without history"
            else:
                group = "Other races"

            coverage_groups[group][race_key] = runners

        for group, races in coverage_groups.items():
            if not races:
                continue

            model, uniform = evaluate_race_scores(races)

            print(
                f"{group}: {len(races):,} races | "
                f"Model: {model:.6f} | Uniform: {uniform:.6f} | "
                f"Improvement: {uniform - model:.6f}"
            )

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

        example_key = min(validation_scores)
        example_runners = validation_scores[example_key]
        total_score = sum(score for _, _, score in example_runners)

        print(f"\nExample validation race: {example_key}")

        for horse, position, score in sorted(
            example_runners,
            key=lambda runner: runner[2],
            reverse=True,
        ):
            probability = score / total_score

            print(
                f"{horse} | Probability: {probability:.2%} | "
                f"Actual position: {position}"
            )

        model_loss, uniform_loss = evaluate_race_scores(
            validation_scores
        )

        print(f"Validation smoothed-form log loss: {model_loss:.6f}")
        print(f"Validation uniform log loss: {uniform_loss:.6f}")
        print(f"Improvement over uniform: {uniform_loss - model_loss:.6f}")

        field_groups = {
            "2–7 runners": {},
            "8–12 runners": {},
            "13+ runners": {},
        }

        for race_key, runners in validation_scores.items():
            size = len(runners)

            if size <= 7:
                group = "2–7 runners"
            elif size <= 12:
                group = "8–12 runners"
            else:
                group = "13+ runners"

            field_groups[group][race_key] = runners

        field_size_results = {}

        for group, races in field_groups.items():
            if not races:
                continue

            model, uniform = evaluate_race_scores(races)

            field_size_results[group] = {
                "races": len(races),
                "model_log_loss": model,
                "uniform_log_loss": uniform,
                "improvement": uniform - model,
            }

            print(
                f"{group}: {len(races):,} races | "
                f"Model: {model:.6f} | Uniform: {uniform:.6f} | "
                f"Improvement: {uniform - model:.6f}"
            )

        if args.report is not None:
            report = {
                "model": "smoothed_horse_win_rate",
                "database": str(args.database.resolve()),
                "alpha": args.alpha,
                "window_days": args.window_days,
                "training_reference_rate": reference_rate,
                "validation_start": "2024-01-01",
                "validation_end_exclusive": "2025-01-01",
                "validation_races": len(validation_scores),
                "model_log_loss": model_loss,
                "uniform_log_loss": uniform_loss,
                "improvement": uniform_loss - model_loss,
                "by_field_size": field_size_results,
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