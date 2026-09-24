import json
from pathlib import Path

import optuna


study_path = Path(
    "outputs/studies/v2_boosting_temporal_v1.db"
).resolve()

if not study_path.is_file():
    raise FileNotFoundError(study_path)

study = optuna.load_study(
    study_name="boosting_temporal_v1",
    storage=f"sqlite:///{study_path.as_posix()}",
)

report = {
    "study_name": study.study_name,
    "direction": study.direction.name,
    "metadata": study.user_attrs,
    "best_trial": study.best_trial.number,
    "best_value": study.best_value,
    "best_parameters": study.best_params,
    "trials": [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "parameters": trial.params,
            "distributions": {
                name: json.loads(
                    optuna.distributions.distribution_to_json(distribution)
                )
                for name, distribution in trial.distributions.items()
            },
            "results": trial.user_attrs,
        }
        for trial in study.trials
    ],
}

output = Path("outputs/reports/v2_boosting_optuna_study.json")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("x", encoding="utf-8") as file:
    json.dump(report, file, indent=2, allow_nan=False)
    file.write("\n")

print(f"Exported {len(study.trials)} trials to: {output}")