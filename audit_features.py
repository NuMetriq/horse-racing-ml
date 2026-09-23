import argparse
import re
import csv

from distance import parse_distance_furlongs
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

        distances = connection.execute(
            """
            SELECT dist, COUNT(*)
            FROM data
            WHERE type = 'Flat'
              AND date >= '2015-01-01'
              AND date < '2024-01-01'
            GROUP BY dist
            ORDER BY COUNT(*) DESC, dist
            """
        ).fetchall()

        print(f"\nDistinct training distance values: {len(distances)}")

        for distance, count in distances:
            print(f"{distance!r}: {count:,}")

        print("\nTraining flat races recorded at 3 miles or longer:")

        long_races = connection.execute(
            """
            SELECT date, course, off, race_name, dist, COUNT(*)
            FROM data
            WHERE type = 'Flat'
              AND date >= '2015-01-01'
              AND date < '2024-01-01'
              AND (dist LIKE '3m%' OR dist LIKE '4m%')
            GROUP BY date, course, off, race_name, dist
            ORDER BY date, course, off
            """
        ).fetchall()

        for race_date, course, off, race_name, distance, runners in long_races:
            print(
                f"\n{race_date} | {course} | {off} | "
                f"Distance: {distance} | Runner rows: {runners}"
            )
            print(f"Race: {race_name}")

        print("\nFlat-labelled training races with jumping descriptions:")

        suspicious_races = connection.execute(
            """
            SELECT date, course, off, race_name, dist, COUNT(*)
            FROM data
            WHERE type = 'Flat'
              AND date >= '2015-01-01'
              AND date < '2024-01-01'
              AND (
                  LOWER(race_name) LIKE '%chase%'
                  OR LOWER(race_name) LIKE '%hurdle%'
                  OR LOWER(race_name) LIKE '%steeple%'
                  OR LOWER(race_name) LIKE '%cross country%'
                  OR LOWER(race_name) LIKE '%cross-country%'
                  OR LOWER(race_name) LIKE '% chs%'
              )
            GROUP BY date, course, off, race_name, dist
            ORDER BY date, course, off
            """
        ).fetchall()

        print(f"Flagged race groups: {len(suspicious_races):,}")

        for race_date, course, off, race_name, distance, runners in suspicious_races:
            print(
                f"\n{race_date} | {course} | {off} | "
                f"Distance: {distance} | Runner rows: {runners}"
            )
            print(f"Race: {race_name}")

        jumping_description = re.compile(
            r"\b(?:chase|hurdles?|steeplechase|chs)\b"
            r"|\bcross[- ]country\b",
            re.IGNORECASE,
        )

        race_rows = connection.execute(
            """
            SELECT date, course, off, race_name, dist, COUNT(*)
            FROM data
            WHERE type = 'Flat'
              AND date >= '2015-01-01'
              AND date < '2024-01-01'
            GROUP BY date, course, off, race_name, dist
            ORDER BY date, course, off
            """
        )

        flagged_count = 0
        review_rows = []

        print("\nCombined race-type review candidates:")

        for race_date, course, off, name, distance, runners in race_rows:
            reasons = []
            furlongs = parse_distance_furlongs(distance)

            if furlongs is not None and furlongs >= 24:
                reasons.append("distance at least 3 miles")

            if jumping_description.search(name or ""):
                reasons.append("race-name keyword")

            if not reasons:
                continue

            flagged_count += 1

            review_rows.append(
                {
                    "date": race_date,
                    "course": course,
                    "off": off,
                    "race_name": name,
                    "distance": distance,
                    "runner_rows": runners,
                    "flag_reason": "; ".join(reasons),
                    "decision": "pending",
                    "evidence": "",
                }
            )

            print(
                f"\n{race_date} | {course} | {off} | "
                f"{distance} | Runner rows: {runners}"
            )
            print(f"Race: {name}")
            print(f"Review reason: {'; '.join(reasons)}")

        print(f"\nTotal review candidates: {flagged_count:,}")

        review_path = Path("docs/v2-race-type-review.csv")
        review_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "date",
            "course",
            "off",
            "race_name",
            "distance",
            "runner_rows",
            "flag_reason",
            "decision",
            "evidence",
        ]

        with review_path.open("x", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(review_rows)

        print(f"Saved {len(review_rows):,} review candidates to: {review_path}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()