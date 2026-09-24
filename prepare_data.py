import argparse
import sqlite3
import csv
from pathlib import Path
from distance import parse_distance_furlongs

from inspect_data import open_database, summarize_eligible_races, select_eligible_races

def load_review_exclusions(
    review_path: Path,
) -> set[tuple[str, str, str]]:
    excluded_keys = set()
    seen_keys = set()

    with review_path.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {"date", "course", "off", "decision", "evidence"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Review file is missing required columns")

        for row in reader:
            key = tuple(
                row[name] for name in ("date", "course", "off")
            )

            if any(value is None or not value.strip() for value in key):
                raise ValueError("Review file contains a blank race key")

            if key in seen_keys:
                raise ValueError(f"Duplicate reviewed race: {key}")

            seen_keys.add(key)
            decision = (row["decision"] or "").strip()

            if decision not in {"retain_flat", "exclude_non_flat"}:
                raise ValueError(
                    f"Unresolved or invalid decision for {key}: {decision!r}"
                )

            if not (row["evidence"] or "").strip():
                raise ValueError(f"Missing review evidence for {key}")

            if decision == "exclude_non_flat":
                excluded_keys.add(key)

    return excluded_keys

def export_races(
    source: sqlite3.Connection,
    output_path: Path,
    races: list[tuple[str, str, str, int, str | None]],
) -> None:
    prepared_races = []

    for race_date, course, off, runners, distance_text in races:
        try:
            distance_furlongs = parse_distance_furlongs(distance_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid distance for {(race_date, course, off)}: "
                f"{distance_text!r}"
            ) from error

        prepared_races.append(
            (
                race_date,
                course,
                off,
                runners,
                distance_text,
                distance_furlongs,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    destination = sqlite3.connect(output_path)
    destination.execute("PRAGMA foreign_keys = ON")

    try:
        with destination:
            destination.execute(
                """
                CREATE TABLE races (
                    date TEXT NOT NULL,
                    course TEXT NOT NULL,
                    off TEXT NOT NULL,
                    runner_count INTEGER NOT NULL,
                    distance_text TEXT,
                    distance_furlongs REAL
                        CHECK (
                            distance_furlongs IS NULL
                            OR distance_furlongs > 0
                        ),
                    PRIMARY KEY (date, course, off)
                )
                """
            )

            destination.executemany(
                """
                INSERT INTO races (
                    date, course, off, runner_count,
                    distance_text, distance_furlongs
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                prepared_races,
            )

            destination.execute(
                """
                CREATE TABLE runners (
                    date TEXT NOT NULL,
                    course TEXT NOT NULL,
                    off TEXT NOT NULL,
                    horse TEXT NOT NULL,
                    finish_position TEXT,
                    age INTEGER,
                    PRIMARY KEY (date, course, off, horse),
                    FOREIGN KEY (date, course, off)
                        REFERENCES races (date, course, off)
                )
                """
            )

            eligible_keys = set(
                destination.execute(
                    "SELECT date, course, off FROM races"
                )
            )

            source_runners = source.execute(
                """
                SELECT date, course, off, horse, CAST(pos AS TEXT), age
                FROM data
                WHERE type = 'Flat'
                """
            )

            destination.executemany(
                """
                INSERT INTO runners
                    (date, course, off, horse, finish_position, age)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row
                    for row in source_runners
                    if row[:3] in eligible_keys
                ),
            )

            actual = destination.execute(
                "SELECT COUNT(*) FROM runners"
            ).fetchone()[0]

            expected = destination.execute(
                "SELECT SUM(runner_count) FROM races"
            ).fetchone()[0]

            if actual != expected:
                raise ValueError(
                    f"Runner count mismatch: expected {expected}, got {actual}"
                )

        race_count, runner_total = destination.execute(
            "SELECT COUNT(*), SUM(runner_count) FROM races"
        ).fetchone()

        print(f"Exported race groups: {race_count:,}")
        print(f"Expected runner rows: {runner_total:,}")
        print(f"Exported runner rows: {actual:,}")
    finally:
        destination.close()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the reviewed flat-racing dataset."
    )
    parser.add_argument("database", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the prepared SQLite database",
    )
    parser.add_argument(
        "--race-type-review",
        type=Path,
        required=True,
        help="Completed CSV containing race-type review decisions",
    )

    args = parser.parse_args()
    output_path = args.output.resolve()

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    excluded_keys = load_review_exclusions(args.race_type_review)
    connection = open_database(args.database.resolve())

    try:
        original_races = select_eligible_races(connection)

        removed_races = [
            row for row in original_races
            if row[:3] in excluded_keys
        ]
        races = [
            row for row in original_races
            if row[:3] not in excluded_keys
        ]

        if not races:
            raise ValueError("No eligible races remain")

        print(f"Output destination: {output_path}")
        print(f"Reviewed exclusion keys: {len(excluded_keys):,}")
        print(f"Eligible race groups before review: {len(original_races):,}")
        print(f"Eligible race groups removed: {len(removed_races):,}")
        print(
            "Eligible runner rows removed: "
            f"{sum(row[3] for row in removed_races):,}"
        )
        print(
            "Exclusion keys outside the eligible set: "
            f"{len(excluded_keys) - len(removed_races):,}"
        )
        print(f"Eligible race groups after review: {len(races):,}")
        print(
            "Eligible runner rows after review: "
            f"{sum(row[3] for row in races):,}"
        )

        export_races(connection, output_path, races)
    finally:
        connection.close()


if __name__ == "__main__":
    main()