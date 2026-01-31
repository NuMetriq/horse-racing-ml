from __future__ import annotations

from pathlib import Path
import sqlite3
import pandas as pd

RAW_DIR = Path("data/raw")


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

        # columns
        cols = cur.execute(f"PRAGMA table_info({qt})").fetchall()
        colnames = [c[1] for c in cols]
        print(f"\n-- {t} ({len(colnames)} cols) --")
        print(colnames)

        # row count
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {qt}").fetchone()[0]
            print("Row count:", n)
        except Exception as e:
            print("Could not count rows:", e)

        # sample rows
        try:
            df = pd.read_sql_query(f"SELECT * FROM {qt} LIMIT 5", con)
            print(df.head())
        except Exception as e:
            print("Could not sample rows:", e)

    con.close()


def main() -> None:
    dbs = list(RAW_DIR.glob("*.db")) + list(RAW_DIR.glob("*.sqlite")) + list(RAW_DIR.glob("*.sqlite3"))
    if not dbs:
        raise FileNotFoundError(f"No .db/.sqlite found in {RAW_DIR.resolve()}")

    for db in dbs:
        inspect_sqlite(db)


if __name__ == "__main__":
    main()