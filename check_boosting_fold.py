import argparse
from pathlib import Path

from boosting_experiment import load_fold, fit_and_score_fold
from inspect_data import open_database


def main():
    parser = argparse.ArgumentParser(
        description="Check reusable boosting fold loading and scoring."
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    connection = open_database(args.database.resolve())
    try:
        fold = load_fold(connection, evaluation_year=2021)
    finally:
        connection.close()

    print("Fitting the 2021 check fold...", flush=True)

    result = fit_and_score_fold(
        **fold,
        parameters={"max_iter": 400},
    )

    print(f"Evaluation races: {result['races']:,}")
    print(f"Boosting race log loss: {result['model_log_loss']:.6f}")


if __name__ == "__main__":
    main()