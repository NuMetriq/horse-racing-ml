import argparse
import pickle

import numpy as np

from inspect_data import open_database
from pathlib import Path
from race_metrics import evaluate_race_scores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved model on validation races."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    with args.model.open("rb") as file:
        bundle = pickle.load(file)

    input_features = bundle["input_features"]

    allowed_features = {
        "prior_starts",
        "prior_wins",
        "prior_win_rate",
        "days_since_run",
    }

    if (
        not input_features
        or len(input_features) != len(set(input_features))
        or any(name not in allowed_features for name in input_features)
    ):
        raise ValueError("Invalid input-feature list in saved model")

    feature_columns = ", ".join(
        f'"{name}"' for name in input_features
    )

    pipeline = bundle["pipeline"]
    connection = open_database(args.database.resolve())

    try:
        rows = connection.execute(
            f"""
            SELECT date, course, off, horse,
                   {feature_columns}, won
            FROM features
            WHERE split = 'validation'
            ORDER BY date, course, off, horse
            """
        ).fetchall()
    finally:
        connection.close()

    X = np.array([row[4:-1] for row in rows], dtype=float)
    win_column = list(pipeline.classes_).index(1)
    probabilities = pipeline.predict_proba(X)[:, win_column]

    race_scores = {}

    for row, probability in zip(rows, probabilities):
        race_key = tuple(row[:3])
        horse = row[3]
        winner_marker = "1" if row[-1] == 1 else "0"

        race_scores.setdefault(race_key, []).append(
            (horse, winner_marker, float(probability))
        )

    normalized_probabilities = []
    outcomes = []

    for runners in race_scores.values():
        total_score = sum(score for _, _, score in runners)

        for horse, winner_marker, score in runners:
            normalized_probabilities.append(score / total_score)
            outcomes.append(int(winner_marker == "1"))

    normalized_probabilities = np.array(normalized_probabilities)
    outcomes = np.array(outcomes)

    print(f"Calibration runner count: {len(outcomes):,}")
    print(f"Mean predicted probability: {normalized_probabilities.mean():.6f}")
    print(f"Observed win rate: {outcomes.mean():.6f}")

    bin_edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]

    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (
                (normalized_probabilities >= lower)
                & (normalized_probabilities <= upper)
            )
        else:
            mask = (
                (normalized_probabilities >= lower)
                & (normalized_probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        predicted = normalized_probabilities[mask].mean()
        observed = outcomes[mask].mean()

        print(
            f"{lower:.0%}–{upper:.0%} | Runners: {count:,} | "
            f"Mean predicted: {predicted:.2%} | "
            f"Observed wins: {observed:.2%}"
        )

    model_loss, uniform_loss = evaluate_race_scores(race_scores)

    print(f"Validation races: {len(race_scores):,}")
    print(f"Saved-model race log loss: {model_loss:.6f}")
    print(f"Uniform race log loss: {uniform_loss:.6f}")


if __name__ == "__main__":
    main()
