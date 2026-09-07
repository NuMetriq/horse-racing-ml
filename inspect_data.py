from pathlib import Path
import sqlite3
import argparse

def open_database(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"Database file not found: {path}")

    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)

def check_race_names(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT
            date,
            course,
            off,
            COUNT(DISTINCT race_name) AS name_count,
            GROUP_CONCAT(DISTINCT race_name) AS names
        FROM data
        GROUP BY date, course, off
        HAVING COUNT(DISTINCT race_name) > 1
        LIMIT 10
        """
    ).fetchall()

    if rows:
        for date, course, off, name_count, names in rows:
            print(f"\n{date} | {course} | {off}")
            print(f"Distinct race names: {name_count}")
            print(names)
    else:
        print("No conflicting race names within candidate race groups.")

def summarize_data(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        """
        SELECT COUNT(*), MIN(date), MAX(date)
        FROM data
        WHERE NOT (
            date = 'date'
            AND race_id = 'race_id'
            AND horse = 'horse'
        )
        """
    ).fetchone()

    count, earliest, latest = row

    print(f"Runner rows: {count:,}")
    print(f"Date range: {earliest} through {latest}")

def summarize_winners(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        WITH race_summary AS (
            SELECT
                date, course, off,
                SUM(CASE WHEN pos = 1 THEN 1 ELSE 0 END) AS winners
            FROM data
            WHERE NOT (
                date = 'date'
                AND race_id = 'race_id'
                AND horse = 'horse'
            )
            GROUP BY date, course, off
        )
        SELECT winners, COUNT(*)
        FROM race_summary
        GROUP BY winners
        ORDER BY winners
        """
    ).fetchall()

    for winners, count in rows:
        print(f"{winners} recorded winner(s): {count:,} race groups")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a horse-racing SQLite database."
    )
    parser.add_argument(
        "database",
        type=Path,
        help="Path to the SQLite database file",
    )
    args = parser.parse_args()
    database_path = args.database.resolve()

    connection = open_database(database_path)

    try:
        summarize_data(connection)
        summarize_winners(connection)
        check_race_names(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()