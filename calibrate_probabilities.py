import argparse
import csv
import math
import numpy as np
import json
from scipy.optimize import minimize_scalar
from datetime import date as calendar_date
from pathlib import Path
from statistics import mean

from race_metrics import evaluate_race_scores


def load_predictions(path, expected_year):
    races = {}
    seen_runners = set()

    required_columns = {
        "date", "course", "off", "horse",
        "win_probability", "won",
    }

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path}")

        for line_number, row in enumerate(reader, start=2):
            race_date = calendar_date.fromisoformat(row["date"])

            if race_date.year != expected_year:
                raise ValueError(
                    f"{path}, line {line_number}: "
                    f"expected year {expected_year}"
                )

            course = row["course"].strip()
            off = row["off"].strip()
            horse = row["horse"].strip()

            if not course or not off or not horse:
                raise ValueError(
                    f"{path}, line {line_number}: blank identifier"
                )

            race_key = (race_date.isoformat(), course, off)
            runner_key = (*race_key, horse)

            if runner_key in seen_runners:
                raise ValueError(f"Duplicate runner: {runner_key}")
            seen_runners.add(runner_key)

            probability = float(row["win_probability"])
            if (
                not math.isfinite(probability)
                or not 0 < probability <= 1
            ):
                raise ValueError(
                    f"Invalid probability for {runner_key}"
                )

            won = row["won"].strip()
            if won not in ("0", "1"):
                raise ValueError(
                    f"Invalid winner indicator for {runner_key}"
                )

            races.setdefault(race_key, []).append(
                (horse, won, probability)
            )

    if not races:
        raise ValueError(f"No predictions in {path}")

    for race_key, runners in races.items():
        if len(runners) < 2:
            raise ValueError(f"Fewer than two runners: {race_key}")

        if sum(won == "1" for _, won, _ in runners) != 1:
            raise ValueError(f"Expected one winner: {race_key}")

        total = math.fsum(p for _, _, p in runners)
        if not math.isclose(
            total, 1.0, rel_tol=0.0, abs_tol=1e-8
        ):
            raise ValueError(
                f"Probabilities do not sum to one: {race_key}"
            )

    return races


def adjust_probabilities(races, gamma):
    """Raise probabilities to gamma and normalize within each race."""
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be finite and positive")

    adjusted = {}

    for race_key, runners in races.items():
        log_probabilities = [
            math.log(probability)
            for _, _, probability in runners
        ]
        largest_log = max(log_probabilities)

        weights = [
            math.exp(gamma * (log_p - largest_log))
            for log_p in log_probabilities
        ]
        total = math.fsum(weights)

        adjusted[race_key] = [
            (horse, won, weight / total)
            for (horse, won, _), weight in zip(
                runners, weights, strict=True
            )
        ]

    return adjusted


def prepare_calibration_arrays(races):
    """Cache log probabilities and each race winner's log probability."""
    max_runners = max(len(runners) for runners in races.values())

    log_probabilities = np.full(
        (len(races), max_runners),
        -np.inf,
    )
    winner_logs = np.empty(len(races))

    for index, runners in enumerate(races.values()):
        for column, (_, won, probability) in enumerate(runners):
            log_p = math.log(probability)
            log_probabilities[index, column] = log_p

            if won == "1":
                winner_logs[index] = log_p

    return log_probabilities, winner_logs


def calibrated_loss(arrays, gamma):
    log_probabilities, winner_logs = arrays

    log_totals = np.logaddexp.reduce(
        gamma * log_probabilities,
        axis=1,
    )

    return float(
        np.mean(log_totals - gamma * winner_logs)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fit race-probability calibration on earlier years."
    )
    parser.add_argument("predictions_2021", type=Path)
    parser.add_argument("predictions_2022", type=Path)
    parser.add_argument("predictions_2023", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Save calibration settings and fitting results to a new JSON file",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    paths = {
        2021: args.predictions_2021,
        2022: args.predictions_2022,
        2023: args.predictions_2023,
    }

    races_by_year = {}
    annual_losses = []

    for year, path in paths.items():
        races = load_predictions(path, expected_year=year)
        races_by_year[year] = races

        model_loss, _ = evaluate_race_scores(races)
        annual_losses.append(model_loss)
        runner_count = sum(len(runners) for runners in races.values())

        print(
            f"{year} | Races: {len(races):,} | "
            f"Runners: {runner_count:,} | "
            f"Original loss: {model_loss:.6f}"
        )

    print(
        f"Mean annual original loss: {mean(annual_losses):.6f}"
    )

    for year, races in races_by_year.items():
        adjusted = adjust_probabilities(races, gamma=1.0)

        largest_difference = max(
            abs(original[2] - transformed[2])
            for race_key in races
            for original, transformed in zip(
                races[race_key],
                adjusted[race_key],
                strict=True,
            )
        )

        if largest_difference > 1e-12:
            raise ValueError(
                f"Gamma=1 changed probabilities for {year}"
            )

        print(
            f"{year} | Gamma=1 check passed | "
            f"Maximum difference: {largest_difference:.3e}"
        )

    arrays_by_year = {
        year: prepare_calibration_arrays(races)
        for year, races in races_by_year.items()
    }

    def objective(gamma):
        return mean(
            calibrated_loss(arrays, gamma)
            for arrays in arrays_by_year.values()
        )

    if not math.isclose(
        objective(1.0),
        mean(annual_losses),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise ValueError("Cached loss does not match original scoring")

    result = minimize_scalar(
        objective,
        bounds=(0.5, 2.0),
        method="bounded",
        options={"xatol": 1e-6},
    )

    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError(f"Calibration failed: {result.message}")

    # Include the boundaries and the unchanged model explicitly.
    gamma = min(
        (0.5, 1.0, 2.0, float(result.x)),
        key=objective,
    )

    print(f"\nFitted gamma: {gamma:.9f}")
    print(f"Original mean annual loss: {objective(1.0):.6f}")
    print(f"Adjusted mean annual loss: {objective(gamma):.6f}")

    for year, arrays in arrays_by_year.items():
        original = calibrated_loss(arrays, 1.0)
        adjusted = calibrated_loss(arrays, gamma)

        print(
            f"{year} | Original: {original:.6f} | "
            f"Adjusted: {adjusted:.6f} | "
            f"Improvement: {original - adjusted:+.6f}"
        )

    if gamma in (0.5, 2.0):
        print("The selected gamma is at a search boundary.")

    report = {
        "method": "race_probability_power",
        "gamma": float(gamma),
        "gamma_bounds": [0.5, 2.0],
        "base_model": "initial_hist_gradient_boosting",
        "feature_set": "relative_finish_age_distance_change",
        "fitting_years": sorted(arrays_by_year),
        "objective": "Equal-weight mean of annual race log losses",
        "prediction_files": {
            str(year): str(path.resolve())
            for year, path in paths.items()
        },
        "original_mean_loss": objective(1.0),
        "adjusted_mean_loss": objective(gamma),
        "annual_results": {
            str(year): {
                "races": len(races_by_year[year]),
                "original_loss": calibrated_loss(arrays, 1.0),
                "adjusted_loss": calibrated_loss(arrays, gamma),
            }
            for year, arrays in arrays_by_year.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    print(f"Calibration saved to: {args.output}")


if __name__ == "__main__":
    main()