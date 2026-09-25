import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from calibrate_probabilities import load_predictions
from check_replay_features import CASES
from inspect_data import open_database


def main():
    parser = argparse.ArgumentParser(
        description="Check racecard CLI predictions against validation exports."
    )
    parser.add_argument("history_database", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("original_predictions", type=Path)
    parser.add_argument("calibrated_predictions", type=Path)
    args = parser.parse_args()

    original = load_predictions(
        args.original_predictions, expected_year=2024
    )
    calibrated = load_predictions(
        args.calibrated_predictions, expected_year=2024
    )

    predictor = Path(__file__).resolve().with_name(
        "predict_boosting_racecard.py"
    )
    history = open_database(args.history_database.resolve())
    total_comparisons = 0

    try:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)

            for index, race_key in enumerate(CASES):
                race = history.execute(
                    """
                    SELECT distance_text, runner_count
                    FROM races
                    WHERE date = ? AND course = ? AND off = ?
                    """,
                    race_key,
                ).fetchone()

                if race is None or not race[0]:
                    raise ValueError(f"Missing race distance: {race_key}")

                runners = history.execute(
                    """
                    SELECT horse, age
                    FROM runners
                    WHERE date = ? AND course = ? AND off = ?
                    ORDER BY horse
                    """,
                    race_key,
                ).fetchall()

                if len(runners) != race[1]:
                    raise ValueError(f"Incomplete racecard: {race_key}")

                racecard_path = folder / f"racecard_{index}.csv"
                report_path = folder / f"report_{index}.json"

                with racecard_path.open(
                    "x", encoding="utf-8", newline=""
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(["horse", "age"])
                    writer.writerows(runners)

                command = [
                    sys.executable,
                    str(predictor),
                    str(args.history_database.resolve()),
                    "--model", str(args.model.resolve()),
                    "--calibration", str(args.calibration.resolve()),
                    "--date", race_key[0],
                    "--distance", race[0],
                    "--runners-file", str(racecard_path),
                    "--report", str(report_path),
                ]

                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                )

                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Predictor failed for {race_key}\n"
                        f"{completed.stdout}\n{completed.stderr}"
                    )

                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )

                if report.get("calibrated") is not True:
                    raise ValueError("Expected calibrated predictions")

                predictions = report["predictions"]
                actual = {row["horse"]: row for row in predictions}
                supplied_horses = {horse for horse, _ in runners}

                if (
                    len(actual) != len(predictions)
                    or set(actual) != supplied_horses
                ):
                    raise ValueError(f"Output runner mismatch: {race_key}")

                checks = [
                    (
                        "uncalibrated_win_probability",
                        original,
                    ),
                    (
                        "win_probability",
                        calibrated,
                    ),
                ]

                largest_difference = 0.0

                for field, reference in checks:
                    if race_key not in reference:
                        raise ValueError(
                            f"Race absent from reference: {race_key}"
                        )

                    expected = {
                        horse: probability
                        for horse, _, probability in reference[race_key]
                    }

                    if set(actual) != set(expected):
                        raise ValueError(
                            f"Reference runner mismatch: {race_key}"
                        )

                    for horse, expected_probability in expected.items():
                        probability = actual[horse][field]

                        if not math.isfinite(probability):
                            raise ValueError("Nonfinite prediction")

                        difference = abs(
                            probability - expected_probability
                        )

                        if difference > 1e-12:
                            raise ValueError(
                                f"{race_key} | {horse} | {field}: "
                                f"difference={difference}"
                            )

                        largest_difference = max(
                            largest_difference, difference
                        )
                        total_comparisons += 1

                print(
                    f"PASS: {race_key} | {len(actual)} runners | "
                    f"Maximum difference: {largest_difference:.3e}"
                )

    finally:
        history.close()

    print(f"\nProbability comparisons passed: {total_comparisons}")


if __name__ == "__main__":
    main()