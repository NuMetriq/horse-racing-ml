import argparse
import csv
import math
from statistics import mean, median
from pathlib import Path


def load_predictions(path):
    predictions = {}

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (
                row["date"],
                row["course"],
                row["off"],
                row["horse"],
            )

            if key in predictions:
                raise ValueError(f"Duplicate runner: {key}")

            predictions[key] = (
                float(row["win_probability"]),
                int(row["won"]),
            )

    return predictions


def calculate_race_losses(predictions):
    grouped = {}

    for key, (probability, won) in predictions.items():
        if not math.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError(f"Invalid probability for {key}")

        if won not in (0, 1):
            raise ValueError(f"Invalid winner label for {key}")

        grouped.setdefault(key[:3], []).append((probability, won))

    if not grouped:
        raise ValueError("No predictions found")

    losses = {}

    for race_key, runners in grouped.items():
        total = math.fsum(probability for probability, _ in runners)
        if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Probabilities do not sum to one: {race_key}")

        winners = [
            probability
            for probability, won in runners
            if won == 1
        ]

        if len(winners) != 1:
            raise ValueError(f"Expected exactly one winner: {race_key}")

        losses[race_key] = -math.log(winners[0])

    return losses


def main():
    parser = argparse.ArgumentParser(
        description="Compare two validation prediction exports."
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = load_predictions(args.baseline)
    candidate = load_predictions(args.candidate)

    if baseline.keys() != candidate.keys():
        raise ValueError("Prediction files contain different runners")

    for key in baseline:
        if baseline[key][1] != candidate[key][1]:
            raise ValueError(f"Winner labels differ: {key}")

    race_keys = {key[:3] for key in baseline}

    print(f"Matched runner rows: {len(baseline):,}")
    print(f"Matched races: {len(race_keys):,}")

    baseline_losses = calculate_race_losses(baseline)
    candidate_losses = calculate_race_losses(candidate)

    differences = [
        baseline_losses[key] - candidate_losses[key]
        for key in sorted(race_keys)
    ]

    tolerance = 1e-12
    improved = sum(value > tolerance for value in differences)
    worsened = sum(value < -tolerance for value in differences)
    unchanged = len(differences) - improved - worsened

    print(f"Baseline race log loss: {mean(baseline_losses.values()):.6f}")
    print(f"Candidate race log loss: {mean(candidate_losses.values()):.6f}")
    print(f"Mean improvement: {mean(differences):.6f}")
    print(f"Median improvement: {median(differences):.6f}")
    print(f"Races improved: {improved:,}")
    print(f"Races worsened: {worsened:,}")
    print(f"Races effectively unchanged: {unchanged:,}")

    monthly_losses = {}

    for race_key in sorted(race_keys):
        month = race_key[0][:7]
        monthly_losses.setdefault(month, []).append(
            (
                baseline_losses[race_key],
                candidate_losses[race_key],
            )
        )

    print("\nMonthly comparison:")

    for month, losses in sorted(monthly_losses.items()):
        baseline_mean = mean(pair[0] for pair in losses)
        candidate_mean = mean(pair[1] for pair in losses)
        improvement = mean(
            baseline_loss - candidate_loss
            for baseline_loss, candidate_loss in losses
        )

        print(
            f"{month} | Races: {len(losses):,} | "
            f"Baseline: {baseline_mean:.6f} | "
            f"Candidate: {candidate_mean:.6f} | "
            f"Improvement: {improvement:+.6f}"
        )

if __name__ == "__main__":
    main()