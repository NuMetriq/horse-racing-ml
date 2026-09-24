import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import optuna
import sklearn

from boosting_experiment import load_fold, fit_and_score_fold
from inspect_data import open_database


YEARS = (2021, 2022, 2023)

BASELINE_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
}


def objective(trial, folds):
    parameters = {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.10, log=True
        ),
        "max_iter": trial.suggest_int(
            "max_iter", 100, 400, step=50
        ),
        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes", 7, 31
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 20, 200, step=10
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization", 0.01, 10.0, log=True
        ),
    }

    annual_losses = []

    for year, fold in folds.items():
        print(
            f"Trial {trial.number}: fitting evaluation year {year}...",
            flush=True,
        )

        result = fit_and_score_fold(
            **fold,
            parameters=parameters,
        )

        loss = float(result["model_log_loss"])
        annual_losses.append(loss)

        trial.set_user_attr(f"log_loss_{year}", loss)
        trial.set_user_attr(f"races_{year}", result["races"])

        print(
            f"Trial {trial.number} | {year} | Loss: {loss:.6f}",
            flush=True,
        )

        del result

    return mean(annual_losses)


def main():
    parser = argparse.ArgumentParser(
        description="Tune boosting on chronological annual folds."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument("--study-output", type=Path, required=True)
    args = parser.parse_args()

    database = args.database.resolve()
    study_path = args.study_output.resolve()

    if study_path.exists():
        raise FileExistsError(
            f"Study output already exists: {study_path}"
        )

    folds = {}
    connection = open_database(database)

    try:
        for year in YEARS:
            print(f"Loading {year} fold...", flush=True)
            folds[year] = load_fold(connection, year)
    finally:
        connection.close()

    study_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name="boosting_temporal_v1",
        storage=f"sqlite:///{study_path.as_posix()}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )

    study.set_user_attr("database", str(database))
    study.set_user_attr("evaluation_years", list(YEARS))
    study.set_user_attr(
        "objective",
        "Equal-weight mean of annual race log losses",
    )
    study.set_user_attr(
        "feature_set",
        "relative_finish_age_distance_change",
    )
    study.set_user_attr("trial_budget", 20)
    study.set_user_attr("optuna_version", optuna.__version__)
    study.set_user_attr("sklearn_version", sklearn.__version__)
    study.set_user_attr("numpy_version", np.__version__)

    study.enqueue_trial(BASELINE_PARAMETERS)

    study.optimize(
        lambda trial: objective(trial, folds),
        n_trials=20,
        n_jobs=1,
    )

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best mean annual loss: {study.best_value:.6f}")
    print(json.dumps(study.best_params, indent=2))

    for year in YEARS:
        loss = study.best_trial.user_attrs[f"log_loss_{year}"]
        print(f"{year}: {loss:.6f}")

    print(f"Study saved to: {study_path}")


if __name__ == "__main__":
    main()