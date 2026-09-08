import argparse
import math
from pathlib import Path

from inspect_data import open_database


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the prepared racing dataset."
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    connection = open_database(args.database.resolve())

    try:
        invalid = connection.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT date, course, off
                FROM runners
                GROUP BY date, course, off
                HAVING SUM(
                    CASE WHEN finish_position = '1' THEN 1 ELSE 0 END
                ) != 1
            )
            """
        ).fetchone()[0]

        print(f"Race groups without exactly one winner: {invalid:,}")

        counts = connection.execute(
            """
            SELECT
                CASE
                    WHEN finish_position = '1' THEN 1
                    ELSE 0
                END AS won,
                COUNT(*) AS runner_count
            FROM runners
            GROUP BY won
            ORDER BY won
            """
        ).fetchall()

        for won, count in counts:
            print(f"won = {won}: {count:,} runners")

        field_sizes = connection.execute(
            """
            SELECT runner_count, COUNT(*) AS race_count
            FROM races
            GROUP BY runner_count
            ORDER BY runner_count
            """
        ).fetchall()

        for runners, races in field_sizes:
            print(f"{runners} runner(s): {races:,} races")

        examples = connection.execute(
            """
            SELECT
                date,
                course,
                off,
                runner_count,
                1.0 / runner_count AS uniform_probability
            FROM races
            ORDER BY date, course, off
            LIMIT 5
            """
        ).fetchall()

        for date, course, off, runners, probability in examples:
            print(f"\n{date} | {course} | {off}")
            print(f"Runners: {runners}")
            print(f"Probability per runner: {probability:.4f}")

        race_sizes = connection.execute(
            "SELECT runner_count FROM races"
        )

        total_loss = 0.0
        race_count = 0

        for (runners,) in race_sizes:
            total_loss += math.log(runners)
            race_count += 1

        mean_loss = total_loss / race_count
        print(f"\nUniform baseline race log loss: {mean_loss:.6f}")

        yearly_counts = connection.execute(
            """
            SELECT SUBSTR(date, 1, 4) AS year, COUNT(*)
            FROM races
            GROUP BY year
            ORDER BY year
            """
        ).fetchall()

        for year, count in yearly_counts:
            print(f"{year}: {count:,} races")

        split_counts = connection.execute(
            """
            SELECT
                CASE
                    WHEN date < '2024-01-01' THEN 'train'
                    WHEN date < '2025-01-01' THEN 'validation'
                    ELSE 'test'
                END AS split,
                COUNT(*) AS races,
                SUM(runner_count) AS runners
            FROM races
            GROUP BY split
            """
        ).fetchall()

        for split, races, runners in split_counts:
            print(f"{split}: {races:,} races | {runners:,} runners")

        validation_sizes = connection.execute(
            """
            SELECT runner_count
            FROM races
            WHERE date >= ? AND date < ?
            """,
            ("2024-01-01", "2025-01-01"),
        )

        total_loss = 0.0
        race_count = 0

        for (runners,) in validation_sizes:
            total_loss += math.log(runners)
            race_count += 1

        validation_loss = total_loss / race_count

        print(
            f"Validation uniform race log loss: {validation_loss:.6f}"
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()