from __future__ import annotations

from pathlib import Path
import re
import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd

RAW_DB = Path("data/raw/raceform.db")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_NAME = "table"  # confirmed: 1,749,795 rows


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_dist_to_furlongs(dist: object) -> float:
    """
    Dataset 'dist' is typically like:
      - '2m 4f 35y'
      - '1m 7f'
      - '7f'
      - '1m'
      - occasionally '2m' or similar

    Returns furlongs (float). (1m = 8f; 1y = 1/220*8f)
    """
    if dist is None or (isinstance(dist, float) and np.isnan(dist)):
        return np.nan
    s = str(dist).strip().lower()
    if not s or s in {"nan", "none"}:
        return np.nan

    m = re.search(r"(\d+)\s*m", s)
    miles = int(m.group(1)) if m else 0

    f = re.search(r"(\d+)\s*f", s)
    furl = int(f.group(1)) if f else 0

    y = re.search(r"(\d+)\s*y", s)
    yards = int(y.group(1)) if y else 0

    furlongs = miles * 8.0 + furl + (yards / 220.0) * 8.0
    return float(furlongs) if furlongs > 0 else np.nan


def parse_wgt_to_lbs(wgt: object) -> float:
    """
    UK weight often like:
      - '11-7' (stones-lbs)
      - sometimes just numeric
    Returns pounds (float).
    """
    if wgt is None or (isinstance(wgt, float) and np.isnan(wgt)):
        return np.nan
    s = str(wgt).strip().lower()
    if not s or s in {"nan", "none"}:
        return np.nan

    # numeric?
    try:
        v = float(s)
        return v if v > 0 else np.nan
    except ValueError:
        pass

    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        st = int(m.group(1))
        lb = int(m.group(2))
        return float(st * 14 + lb)

    return np.nan


def parse_sp_to_decimal(sp: object) -> float:
    """
    Starting Price (SP) often:
      - '10/1', '5/2', '13/8'
      - sometimes 'Evens', 'EVS'
      - sometimes already decimal

    Returns decimal odds (>=1.01) else NaN.
    """
    if sp is None or (isinstance(sp, float) and np.isnan(sp)):
        return np.nan
    s = str(sp).strip().lower()
    if not s or s in {"nan", "none"}:
        return np.nan

    if s in {"evens", "even", "evs"}:
        return 2.0  # 1/1 fractional = 2.0 decimal

    # already numeric decimal odds?
    try:
        v = float(s)
        if v > 1.0:
            return float(v)
        return np.nan
    except ValueError:
        pass

    if "/" in s:
        try:
            a, b = s.split("/", 1)
            a = float(a.strip())
            b = float(b.strip())
            if b <= 0:
                return np.nan
            return float(1.0 + a / b)
        except Exception:
            return np.nan

    return np.nan


def load_table(con: sqlite3.Connection) -> pd.DataFrame:
    qt = qident(TABLE_NAME)
    df = pd.read_sql_query(f"SELECT * FROM {qt}", con)
    return df


def normalize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Parse date
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Normalize key types
    df["race_id"] = df["race_id"].astype(str)
    df["course"] = df["course"].astype(str)

    # Runner identifiers (v1.0.0: use names as IDs)
    df["horse_id"] = df["horse"].astype(str)
    df["jockey_id"] = df["jockey"].astype(str)
    df["trainer_id"] = df["trainer"].astype(str)

    # Target
    df["finish_position"] = pd.to_numeric(df["pos"], errors="coerce")
    df["is_winner"] = (df["finish_position"] == 1).astype(int)

    # Pre-race numeric fields
    df["post_position"] = pd.to_numeric(df["draw"], errors="coerce")
    df["weight_lbs"] = df["wgt"].map(parse_wgt_to_lbs)

    # Ratings: OR/RPR/TS (handicap & racing post ratings)
    df["or_rating"] = pd.to_numeric(df["or"], errors="coerce")
    df["rpr_rating"] = pd.to_numeric(df["rpr"], errors="coerce")
    df["ts_rating"] = pd.to_numeric(df["ts"], errors="coerce")

    # Distance/going
    df["dist_f"] = df["dist"].map(parse_dist_to_furlongs)
    df["going"] = df["going"].astype(str)
    df["race_type"] = df["type"].astype(str)
    df["race_class"] = df["class"].astype(str)

    # Odds
    df["sp_decimal"] = df["sp"].map(parse_sp_to_decimal)

    # Field size (ran is often field size; but compute robustly)
    df["field_size"] = df.groupby("race_id")["race_id"].transform("size")

    # races table (one row per race_id)
    races = (
        df.groupby("race_id", as_index=False)
        .agg(
            race_date=("date", "min"),
            course=("course", "first"),
            off=("off", "first"),
            race_name=("race_name", "first"),
            race_type=("race_type", "first"),
            race_class=("race_class", "first"),
            pattern=("pattern", "first"),
            rating_band=("rating_band", "first"),
            age_band=("age_band", "first"),
            sex_rest=("sex_rest", "first"),
            dist_f=("dist_f", "first"),
            going=("going", "first"),
            field_size=("field_size", "first"),
            prize=("prize", "first"),
        )
    )

    runners_cols = [
        "race_id",
        "date",
        "course",
        "horse_id",
        "jockey_id",
        "trainer_id",
        "post_position",
        "weight_lbs",
        "or_rating",
        "rpr_rating",
        "ts_rating",
        "sp_decimal",
        "finish_position",
        "is_winner",
        "field_size",
        "num",  # runner number in racecard if useful
        "age",
        "sex",
    ]
    # only keep those that exist
    runners_cols = [c for c in runners_cols if c in df.columns]
    runners = df[runners_cols].rename(columns={"date": "race_date"}).copy()

    return races, runners


def main() -> None:
    if not RAW_DB.exists():
        raise FileNotFoundError(f"Missing {RAW_DB.resolve()}")

    con = sqlite3.connect(RAW_DB)
    df = load_table(con)
    con.close()

    races, runners = normalize(df)

    races_path = OUT_DIR / "races.parquet"
    runners_path = OUT_DIR / "runners.parquet"
    races.to_parquet(races_path, index=False)
    runners.to_parquet(runners_path, index=False)

    print("Saved:")
    print(" ", races_path)
    print(" ", runners_path)
    print("Rows:", {"races": len(races), "runners": len(runners)})
    print("Date range:", races["race_date"].min(), "→", races["race_date"].max())


if __name__ == "__main__":
    main()