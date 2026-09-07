from pathlib import Path
import sqlite3

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

def main() -> None:
    database_path = (
        Path.home()
        / ".cache/kagglehub/datasets/deltaromeo"
        / "horse-racing-results-ukireland-2015-2025"
        / "versions/118/form_2015-present/form_2015-present/raceform.db"
    )

    connection = open_database(database_path)

    try:
        check_race_names(connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()