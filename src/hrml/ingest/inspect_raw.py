from __future__ import annotations

from pathlib import Path
import sqlite3
import argparse
import pandas as pd


def qident(name: str) -> str:
    """SQLite identifier quoting."""
    return '"' + name.replace('"', '""') + '"'


def inspect_sqlite(db_path: Path) -> None:
    print(f"\n== SQLite: {db_path} ==")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [t[0] for t in tables]
    print("Tables:", tables)

    for t in tables:
        qt = qident(t)

        cols = cur.execute(f"PRAGMA table_info({qt})").fetchall()
        colnames = [c[1] for c in cols]
        print(f"\n-- {t} ({len(colnames)} cols) --")
        print(colnames)

        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {qt}").fetchone()[0]
            print("Row count:", n)
        except Exception as e:
            print("Could not count rows:", e)

        try:
            df = pd.read_sql_query(f"SELECT * FROM {qt} LIMIT 5", con)
            print(df.head())
        except Exception as e:
            print("Could not sample rows:", e)

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw SQLite files")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw SQLite files (default: data/raw)",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    dbs = list(raw_dir.glob("*.db")) + list(raw_dir.glob("*.sqlite")) + list(raw_dir.glob("*.sqlite3"))
    if not dbs:
        raise FileNotFoundError(f"No .db/.sqlite found in {raw_dir.resolve()}")

    for db in dbs:
        inspect_sqlite(db)


if __name__ == "__main__":
    main()