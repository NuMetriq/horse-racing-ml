import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from inspect_data import open_database
from compare_predictions import load_predictions, calculate_race_losses
from statistics import mean


def main():
    folder = (
        Path.home()
        / ".cache/kagglehub/datasets/deltaromeo"
        / "horse-racing-results-ukireland-2015-2025"
        / "versions/118/betfair/betfair"
    )

    connection = open_database(
        Path("data/processed/v2_flat_competitive.db").resolve()
    )
    try:
        prepared_rows = connection.execute(
            """
            SELECT date, horse, course, off
            FROM runners
            WHERE date >= '2026-03-01'
              AND date < '2026-04-30'
            """
        ).fetchall()
    finally:
        connection.close()

    prepared = defaultdict(list)
    for date, horse, course, off in prepared_rows:
        prepared[(date, horse)].append((course, off))

    counts = Counter()
    market_keys = Counter()
    bsp_by_runner = {}
    examples = []

    for path in sorted(folder.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                key = (row["date"], row["horse"])
                market_keys[key] += 1
                try:
                    bsp = float(row["bsp"])
                except (TypeError, ValueError):
                    bsp = math.nan

                if math.isfinite(bsp) and bsp > 1:
                    bsp_by_runner[key] = bsp
                matches = prepared.get(key, [])

                if not matches:
                    counts["unmatched"] += 1
                elif len(matches) > 1:
                    counts["ambiguous"] += 1
                else:
                    counts["unique_match"] += 1
                    course, off = matches[0]

                    if row["course"] != course or row["off"] != off:
                        counts["different_course_or_time"] += 1

                        if len(examples) < 5:
                            examples.append(
                                (
                                    key,
                                    (row["course"], row["off"]),
                                    (course, off),
                                )
                            )

    prepared_races = defaultdict(list)

    for date, horse, course, off in prepared_rows:
        prepared_races[(date, course, off)].append((date, horse))

    coverage = Counter()
    complete_runner_count = 0
    complete_race_keys = set()

    for race_key, runner_keys in prepared_races.items():
        if any(key not in market_keys for key in runner_keys):
            coverage["missing_market_rows"] += 1
        elif any(key not in bsp_by_runner for key in runner_keys):
            coverage["missing_or_invalid_bsp"] += 1
        else:
            coverage["complete"] += 1
            complete_runner_count += len(runner_keys)
            complete_race_keys.add(race_key)

    print(f"\nPrepared races in period: {len(prepared_races):,}")
    print(
        "Races missing market rows: "
        f"{coverage['missing_market_rows']:,}"
    )
    print(
        "Races with all market rows but missing/invalid BSP: "
        f"{coverage['missing_or_invalid_bsp']:,}"
    )
    print(f"Races with complete valid BSP: {coverage['complete']:,}")
    print(f"Runners in complete-BSP races: {complete_runner_count:,}")

    missing_prepared = [
        row
        for row in prepared_rows
        if (row[0], row[1]) not in market_keys
    ]

    print(
        f"\nPrepared runners absent from market files: "
        f"{len(missing_prepared):,}"
    )

    for date, horse, course, off in missing_prepared:
        print(f"{date} | {course} | {off} | {horse}")

    print(f"Prepared runner rows in period: {len(prepared_rows):,}")
    print(f"Market rows with one candidate match: {counts['unique_match']:,}")
    print(f"Market rows unmatched: {counts['unmatched']:,}")
    print(f"Market rows with multiple candidate matches: {counts['ambiguous']:,}")
    print(
        "Repeated date/horse keys in market files: "
        f"{sum(count > 1 for count in market_keys.values()):,}"
    )
    print(
        "Unique matches with different course or time text: "
        f"{counts['different_course_or_time']:,}"
    )

    for key, market_race, prepared_race in examples:
        print(f"\nDate/horse: {key}")
        print(f"Market course/time: {market_race}")
        print(f"Prepared course/time: {prepared_race}")

    prediction_path = Path(
        "outputs/predictions/v2_logistic_relative_finish_test.csv"
    )
    model_predictions = load_predictions(prediction_path)

    selected_model = {}
    selected_market = {}

    for race_key in sorted(complete_race_keys):
        date, course, off = race_key
        runner_keys = prepared_races[race_key]

        inverse_total = math.fsum(
            1.0 / bsp_by_runner[key]
            for key in runner_keys
        )

        for runner_key in runner_keys:
            _, horse = runner_key
            prediction_key = (date, course, off, horse)

            if prediction_key not in model_predictions:
                raise ValueError(
                    f"Model prediction missing: {prediction_key}"
                )

            model_probability, won = model_predictions[prediction_key]
            market_probability = (
                1.0 / bsp_by_runner[runner_key]
            ) / inverse_total

            selected_model[prediction_key] = (model_probability, won)
            selected_market[prediction_key] = (market_probability, won)

    model_losses = calculate_race_losses(selected_model)
    market_losses = calculate_race_losses(selected_market)

    model_mean = mean(model_losses.values())
    market_mean = mean(market_losses.values())

    print(f"\nMatched comparison races: {len(model_losses):,}")
    print(f"Matched comparison runners: {len(selected_model):,}")
    print(f"Frozen-model race log loss: {model_mean:.6f}")
    print(f"Normalized-BSP race log loss: {market_mean:.6f}")
    print(
        "Model improvement over BSP: "
        f"{market_mean - model_mean:+.6f}"
    )


if __name__ == "__main__":
    main()