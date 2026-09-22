import argparse
from pathlib import Path

from inspect_data import open_database


def main():
    parser = argparse.ArgumentParser(
        description="Inspect candidate features in raw flat-racing training data."
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    connection = open_database(args.database.resolve())

    try:
        total = connection.execute(
            """
            SELECT COUNT(*)
            FROM data
            WHERE type = 'Flat' AND date < '2024-01-01'
            """
        ).fetchone()[0]

        print(f"Raw flat training rows: {total:,}")

        for column in ("dist", "going", "age"):
            missing = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM data
                WHERE type = 'Flat'
                  AND date < '2024-01-01'
                  AND (
                      "{column}" IS NULL
                      OR TRIM(CAST("{column}" AS TEXT)) = ''
                  )
                """
            ).fetchone()[0]

            values = connection.execute(
                f"""
                SELECT "{column}", COUNT(*) AS frequency
                FROM data
                WHERE type = 'Flat' AND date < '2024-01-01'
                GROUP BY "{column}"
                ORDER BY frequency DESC
                LIMIT 15
                """
            ).fetchall()

            print(f"\nField: {column}")
            print(f"Null or blank: {missing:,}")
            print("Most frequent values:")

            for value, count in values:
                print(f"  {value!r}: {count:,}")

            age_values = connection.execute(
            """
            SELECT age, typeof(age), COUNT(*)
            FROM data
            WHERE type = 'Flat' AND date < '2024-01-01'
            GROUP BY age, typeof(age)
            ORDER BY age
            """
        ).fetchall()

        print("\nAll age values and storage types:")
        for value, storage_type, count in age_values:
            print(f"{value!r} | {storage_type} | {count:,}")

        unusual_age_rows = connection.execute(
            """
            SELECT date, course, race_id, off, race_name,
                   horse, age, age_band
            FROM data
            WHERE type = 'Flat'
              AND date < '2024-01-01'
              AND age = 1
            ORDER BY date, course, off, horse
            """
        ).fetchall()

        print("\nTraining rows with age 1:")
        for row in unusual_age_rows:
            (
                race_date, course, race_id, off,
                race_name, horse, age, age_band,
            ) = row

            print(f"\n{race_date} | {course} | {off} | ID: {race_id}")
            print(f"Race: {race_name}")
            print(f"Horse: {horse} | Age: {age} | Age band: {age_band}")

        horse_history = connection.execute(
            """
            SELECT date, course, off, age, age_band
            FROM data
            WHERE horse = ?
              AND date < '2024-01-01'
            ORDER BY date, course, off
            """,
            ("Anthem Of Peace (AUS)",),
        ).fetchall()

        print("\nAnthem Of Peace: training-period age history:")
        for race_date, course, off, age, age_band in horse_history:
            print(
                f"{race_date} | {course} | {off} | "
                f"Age: {age} | Age band: {age_band}"
            )
    finally:
        connection.close()


if __name__ == "__main__":
    main()