import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from race_metrics import evaluate_race_scores
from feature_transforms import (
    encode_relative_finish_age_distance_change,
)

def load_fold(connection, evaluation_year):
    """Load earlier training data and one evaluation calendar year."""
    evaluation_start = f"{evaluation_year}-01-01"
    evaluation_end = f"{evaluation_year + 1}-01-01"

    training_rows = connection.execute(
        """
        SELECT
            prior_starts, prior_wins, prior_win_rate,
            days_since_run, previous_position,
            previous_runner_count, age,
            distance_change_furlongs, won
        FROM features
        WHERE date >= ? AND date < ?
        ORDER BY date, course, off, horse
        """,
        ("2015-01-01", evaluation_start),
    ).fetchall()

    evaluation_rows = connection.execute(
        """
        SELECT
            date, course, off, horse,
            prior_starts, prior_wins, prior_win_rate,
            days_since_run, previous_position,
            previous_runner_count, age,
            distance_change_furlongs, won
        FROM features
        WHERE date >= ? AND date < ?
        ORDER BY date, course, off, horse
        """,
        (evaluation_start, evaluation_end),
    ).fetchall()

    if not training_rows or not evaluation_rows:
        raise ValueError(
            f"Empty training or evaluation data for {evaluation_year}"
        )

    encoder = encode_relative_finish_age_distance_change

    return {
        "X_train": np.array(
            [encoder(row[:-1]) for row in training_rows],
            dtype=float,
        ),
        "y_train": np.array(
            [row[-1] for row in training_rows],
            dtype=int,
        ),
        "X_evaluation": np.array(
            [encoder(row[4:-1]) for row in evaluation_rows],
            dtype=float,
        ),
        "y_evaluation": np.array(
            [row[-1] for row in evaluation_rows],
            dtype=int,
        ),
        "race_keys": [row[:3] for row in evaluation_rows],
    }

def fit_and_score_fold(
    X_train,
    y_train,
    X_evaluation,
    y_evaluation,
    race_keys,
    *,
    parameters=None,
):
    """Fit on earlier data and score a later evaluation period."""
    settings = {
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    }

    if parameters is not None:
        unknown = set(parameters) - set(settings)
        if unknown:
            raise ValueError(
                f"Unsupported tuning parameters: {sorted(unknown)}"
            )
        settings.update(parameters)

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=False,
        random_state=42,
        **settings,
    )

    model.fit(X_train, y_train)

    win_column = list(model.classes_).index(1)
    probabilities = model.predict_proba(
        X_evaluation
    )[:, win_column]

    race_scores = {}

    for index, (race_key, won, probability) in enumerate(
        zip(
            race_keys,
            y_evaluation,
            probabilities,
            strict=True,
        )
    ):
        winner_marker = "1" if won == 1 else "0"

        race_scores.setdefault(race_key, []).append(
            (str(index), winner_marker, float(probability))
        )

    model_loss, uniform_loss = evaluate_race_scores(race_scores)

    return {
        "model": model,
        "model_log_loss": model_loss,
        "uniform_log_loss": uniform_loss,
        "races": len(race_scores),
        "runners": len(y_evaluation),
    }