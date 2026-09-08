import argparse
import sqlite3
from pathlib import Path

from inspect_data import open_database, summarize_eligible_races

def export_races(
    source: sqlite3.Connection,
    output_path: Path,
) -> None:
    races = source.execute(
        """
        SELECT date, course, off, COUNT(*) AS runners
        FROM data
        WHERE type = 'Flat'
        GROUP BY date, course, off
        HAVING SUM(CASE WHEN pos = 1 THEN 1 ELSE 0 END) = 1
           AND COUNT(*) = MIN(ran)
           AND MIN(ran) = MAX(ran)
           AND COUNT(ran) = COUNT(*)
           AND COUNT(*) >= 2
        """

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
                    PRIMARY KEY (date, course, off)
                )
                """
            )

            destination.executemany(
                """
                INSERT INTO races (date, course, off, runner_count)
                VALUES (?, ?, ?, ?)
                """,
                races,
            )

            destination.execute(
                """
                CREATE TABLE runners (
                    date TEXT NOT NULL,
                    course TEXT NOT NULL,
                    off TEXT NOT NULL,
                    horse TEXT NOT NULL,
                    finish_position TEXT,
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
                SELECT date, course, off, horse, CAST(pos AS TEXT)
                FROM data
                WHERE type = 'Flat'
                """
            )

            destination.executemany(
                """
                INSERT INTO runners
                    (date, course, off, horse, finish_position)
                VALUES (?, ?, ?, ?, ?)
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
        description="Prepare the initial flat-racing dataset."
    )
    parser.add_argument("database", type=Path)

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the prepared SQLite database",
    )

    args = parser.parse_args()

    output_path = args.output.resolve()

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    print(f"Output destination: {output_path}")

    connection = open_database(args.database.resolve())

    try:
        races, runners = summarize_eligible_races(connection)
        print(f"Eligible race groups: {races:,}")
        print(f"Eligible runner rows: {runners:,}")
        export_races(connection, output_path)
    finally:
        connection.close()


if __name__ == "__main__":
    main()