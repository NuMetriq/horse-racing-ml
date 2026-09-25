import argparse
from pathlib import Path

from inspect_data import open_database


def main():
    parser = argparse.ArgumentParser(
        description="Select historical races for inference replay checks."
    )
    parser.add_argument("database", type=Path)
    args = parser.parse_args()

    connection = open_database(args.database.resolve())

    try:
        rows = connection.execute(
            """
            SELECT
                date, course, off,
                COUNT(*) AS runners,
                SUM(CASE WHEN prior_starts = 0 THEN 1 ELSE 0 END),
                SUM(
                    CASE
                        WHEN previous_position IS NOT NULL
                         AND TRIM(previous_position) != ''
                         AND TRIM(previous_position) GLOB '*[^0-9]*'
                        THEN 1 ELSE 0
                    END
                )
            FROM features
            WHERE date >= '2024-01-01'
              AND date < '2025-01-01'
            GROUP BY date, course, off
            ORDER BY date, course, off
            """
        ).fetchall()
    finally:
        connection.close()

    categories = {
        "Small field": lambda row: 2 <= row[3] <= 7,
        "Large field": lambda row: row[3] >= 13,
        "Previous result code": lambda row: row[5] > 0,
        "All runners have history": lambda row: row[4] == 0,
    }

    selected_keys = set()

    for label, qualifies in categories.items():
        selected = next(
            (
                row for row in rows
                if qualifies(row) and row[:3] not in selected_keys
            ),
            None,
        )

        if selected is None:
            print(f"{label}: no distinct matching race found")
            continue

        selected_keys.add(selected[:3])
        date, course, off, runners, missing, codes = selected

        print(
            f"{label}: {date} | {course} | {off} | "
            f"Runners: {runners} | Without history: {missing} | "
            f"Previous codes: {codes}"
        )


if __name__ == "__main__":
    main()