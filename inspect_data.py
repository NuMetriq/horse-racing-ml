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

def summarize_flat_winners(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        WITH race_summary AS (
            SELECT
                date, course, off,
                SUM(CASE WHEN pos = 1 THEN 1 ELSE 0 END) AS winners
            FROM data
            WHERE type = 'Flat'
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

def summarize_race_types(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT type, COUNT(*) AS runner_rows
        FROM data
        GROUP BY type
        ORDER BY runner_rows DESC
        """
    ).fetchall()

    for race_type, count in rows:
        print(f"Race type {race_type!r}: {count:,} runner rows")

def summarize_flat_races(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        """
        SELECT COUNT(*), MIN(date), MAX(date)
        FROM (
            SELECT date, course, off
            FROM data
            WHERE type = ?
            GROUP BY date, course, off
        )
        """,
        ("Flat",),
    ).fetchone()

    races, earliest, latest = row

    print(f"Flat race groups: {races:,}")
    print(f"Flat date range: {earliest} through {latest}")

def check_flat_field_sizes(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT
            date, course, off,
            COUNT(*) AS observed,
            MIN(ran) AS min_ran,
            MAX(ran) AS max_ran
        FROM data
        WHERE type = ?
        GROUP BY date, course, off
        HAVING COUNT(*) != MIN(ran)
            OR MIN(ran) != MAX(ran)
            OR COUNT(ran) != COUNT(*)
        LIMIT 10
        """,
        ("Flat",),
    ).fetchall()

    if rows:
        for date, course, off, observed, min_ran, max_ran in rows:
            print(f"\n{date} | {course} | {off}")
            print(f"Observed rows: {observed}")
            print(f"Reported runner count range: {min_ran} to {max_ran}")
    else:
        print("All flat race groups match their reported runner counts.")

def inspect_flagged_race(connection: sqlite3.Connection) -> None:
    rows = connection.execute(
        """
        SELECT race_id, race_name, horse, type, pos, ran
        FROM data
        WHERE date = ?
          AND course = ?
          AND off = ?
        ORDER BY horse
        """,
        ("2024-09-26", "Funabashi (JPN)", "11:07"),
    ).fetchall()

    for race_id, name, horse, race_type, pos, ran in rows:
        print(f"\nID: {race_id} | Race: {name}")
        print(f"Horse: {horse} | Type: {race_type!r}")
        print(f"Position: {pos!r} | Reported runners: {ran}")

def summarize_eligible_races(
    connection: sqlite3.Connection,
) -> tuple[int, int]:
    row = connection.execute(
        """
        WITH eligible AS (
            SELECT date, course, off, COUNT(*) AS runners
            FROM data
            WHERE type = 'Flat'
            GROUP BY date, course, off
            HAVING SUM(CASE WHEN pos = 1 THEN 1 ELSE 0 END) = 1
               AND COUNT(*) = MIN(ran)
               AND MIN(ran) = MAX(ran)
               AND COUNT(ran) = COUNT(*)
               AND COUNT(*) >= 2
        )
        SELECT COUNT(*), SUM(runners)
        FROM eligible
        """
    ).fetchone()

    races, runners = row
    return races, runners or 0

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
        summarize_flat_winners(connection)
        check_race_names(connection)
        summarize_race_types(connection)
        summarize_flat_races(connection)
        check_flat_field_sizes(connection)
        inspect_flagged_race(connection)
        races, runners = summarize_eligible_races(connection)
        print(f"Eligible race groups: {races:,}")
        print(f"Eligible runner rows: {runners:,}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()