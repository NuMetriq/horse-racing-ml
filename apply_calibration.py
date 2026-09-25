import argparse
import csv
import json
from pathlib import Path

from calibrate_probabilities import (
    load_predictions,
    adjust_probabilities,
)
from race_metrics import evaluate_race_scores


def print_calibration_bins(races):
    edges = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0)

    for index, (lower, upper) in enumerate(
        zip(edges[:-1], edges[1:], strict=True)
    ):
        is_last = index == len(edges) - 2

        runners = [
            (probability, int(won))
            for race in races.values()
            for _, won, probability in race
            if lower <= probability
            and (
                probability < upper
                or (is_last and probability <= upper)
            )
        ]

        if not runners:
            continue

        count = len(runners)
        predicted = sum(p for p, _ in runners) / count
        observed = sum(won for _, won in runners) / count

        print(
            f"{lower:.0%}–{upper:.0%} | Runners: {count:,} | "
            f"Mean predicted: {predicted:.2%} | "
            f"Observed wins: {observed:.2%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Apply frozen calibration to annual predictions."
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(args.output)

    with args.calibration.open(encoding="utf-8") as file:
        calibration = json.load(file)

    if calibration.get("method") != "race_probability_power":
        raise ValueError("Unsupported calibration method")

    if calibration.get("base_model") != "initial_hist_gradient_boosting":
        raise ValueError("Expected initial boosting calibration")

    fitting_years = calibration["fitting_years"]
    if not fitting_years or args.year <= max(fitting_years):
        raise ValueError(
            "Evaluation year must follow the calibration-fitting years"
        )

    gamma = float(calibration["gamma"])

    races = load_predictions(
        args.predictions,
        expected_year=args.year,
    )
    adjusted = adjust_probabilities(races, gamma)

    original_loss, uniform_loss = evaluate_race_scores(races)
    adjusted_loss, _ = evaluate_race_scores(adjusted)

    print(f"Frozen gamma: {gamma:.9f}")
    print(f"Evaluation races: {len(races):,}")
    print(f"Original race log loss: {original_loss:.6f}")
    print(f"Calibrated race log loss: {adjusted_loss:.6f}")
    print(f"Uniform race log loss: {uniform_loss:.6f}")
    print(f"Improvement: {original_loss - adjusted_loss:+.6f}")
    print("\nCalibrated probability bins:")
    print_calibration_bins(adjusted)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "date", "course", "off", "horse",
                "win_probability", "won",
            ]
        )

        for race_key in sorted(adjusted):
            for horse, won, probability in adjusted[race_key]:
                writer.writerow(
                    (*race_key, horse, probability, won)
                )

    runner_count = sum(len(runners) for runners in adjusted.values())
    print(f"Saved {runner_count:,} predictions to: {args.output}")


if __name__ == "__main__":
    main()