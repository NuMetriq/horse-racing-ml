from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import pandas as pd

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Recent:
    finishes: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    dates: Deque[int] = field(default_factory=lambda: deque(maxlen=10))

    def add(self, finish_pos: float, date_ord: int) -> None:
        self.finishes.append(float(finish_pos))
        self.dates.append(int(date_ord))

    def mean_last(self, n: int) -> float:
        if len(self.finishes) == 0:
            return np.nan
        vals = list(self.finishes)[-n:]
        return float(np.mean(vals))

    def win_rate_last(self, n: int) -> float:
        if len(self.finishes) == 0:
            return np.nan
        vals = list(self.finishes)[-n:]
        return float(np.mean([1.0 if v == 1.0 else 0.0 for v in vals]))

    def days_since_last(self, date_ord: int) -> float:
        if len(self.dates) == 0:
            return np.nan
        return float(date_ord - self.dates[-1])

    def starts_365d(self, date_ord: int) -> float:
        if len(self.dates) == 0:
            return 0.0
        cutoff = date_ord - 365
        return float(sum(1 for d in self.dates if d >= cutoff))


@dataclass
class Expanding:
    wins: int = 0
    starts: int = 0

    def add(self, is_win: bool) -> None:
        self.starts += 1
        self.wins += int(is_win)

    def rate(self) -> float:
        if self.starts == 0:
            return np.nan
        return float(self.wins / self.starts)


class Store:
    def __init__(self) -> None:
        self.horse = defaultdict(Recent)
        self.jockey = defaultdict(Expanding)
        self.trainer = defaultdict(Expanding)
        self.combo = defaultdict(Expanding)

    def feats(self, horse_id: str, jockey_id: str, trainer_id: str, date_ord: int) -> dict:
        h = self.horse[horse_id]
        j = self.jockey[jockey_id]
        t = self.trainer[trainer_id]
        c = self.combo[(jockey_id, trainer_id)]
        return {
            "horse_mean_finish_last5": h.mean_last(5),
            "horse_win_rate_last10": h.win_rate_last(10),
            "horse_days_since_last": h.days_since_last(date_ord),
            "horse_starts_365d": h.starts_365d(date_ord),
            "jockey_win_rate_exp": j.rate(),
            "trainer_win_rate_exp": t.rate(),
            "jockey_trainer_win_rate_exp": c.rate(),
        }

    def update(self, horse_id: str, jockey_id: str, trainer_id: str, finish_pos: float, date_ord: int) -> None:
        is_win = (finish_pos == 1.0)
        self.horse[horse_id].add(finish_pos, date_ord)
        self.jockey[jockey_id].add(is_win)
        self.trainer[trainer_id].add(is_win)
        self.combo[(jockey_id, trainer_id)].add(is_win)


def build_model_frame(races: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    races = races.copy()
    runners = runners.copy()

    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    runners["race_date"] = pd.to_datetime(runners["race_date"], errors="coerce")

    # merge race-level context
    keep_race_cols = ["race_id", "race_date", "course", "race_type", "race_class", "dist_f", "going", "field_size"]
    keep_race_cols = [c for c in keep_race_cols if c in races.columns]
    runners = runners.merge(races[keep_race_cols], on="race_id", how="left", validate="many_to_one", suffixes=("", "_r"))

    # ordering
    runners["race_date_ord"] = runners["race_date"].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
    runners = runners.sort_values(["race_date_ord", "race_id"]).reset_index(drop=True)

    store = Store()
    feat_rows = []

    # group by race
    for race_id, grp in runners.groupby("race_id", sort=False):
        feats = []
        date_ord = int(grp["race_date_ord"].iloc[0])

        # compute (history only)
        for _, r in grp.iterrows():
            feats.append(
                store.feats(
                    horse_id=str(r["horse_id"]),
                    jockey_id=str(r.get("jockey_id", "unknown")),
                    trainer_id=str(r.get("trainer_id", "unknown")),
                    date_ord=date_ord,
                )
            )

        feat_df = pd.DataFrame(feats, index=grp.index)
        feat_rows.append(feat_df)

        # update AFTER
        for _, r in grp.iterrows():
            fp = float(r["finish_position"]) if pd.notnull(r["finish_position"]) else np.nan
            if np.isnan(fp):
                continue
            store.update(
                horse_id=str(r["horse_id"]),
                jockey_id=str(r.get("jockey_id", "unknown")),
                trainer_id=str(r.get("trainer_id", "unknown")),
                finish_pos=fp,
                date_ord=date_ord,
            )

    feats_all = pd.concat(feat_rows).sort_index()
    df = pd.concat([runners, feats_all], axis=1)

    # odds implied baseline if available
    if "sp_decimal" in df.columns:
        df["odds_implied"] = 1.0 / df["sp_decimal"].clip(lower=1.01)
        df["odds_implied_norm"] = df["odds_implied"] / df.groupby("race_id")["odds_implied"].transform("sum")

    # target
    df["is_winner"] = df["is_winner"].astype(int)

    return df


def main() -> None:
    races_path = IN_DIR / "races.parquet"
    runners_path = IN_DIR / "runners.parquet"
    if not races_path.exists() or not runners_path.exists():
        raise FileNotFoundError("Run `python -m hrml.ingest.normalize` first.")

    races = pd.read_parquet(races_path)
    runners = pd.read_parquet(runners_path)

    df = build_model_frame(races, runners)
    out_path = OUT_DIR / "model_frame.parquet"
    df.to_parquet(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print("Date range:", df["race_date"].min(), "→", df["race_date"].max())
    print("Has odds baseline:", "odds_implied_norm" in df.columns)


if __name__ == "__main__":
    main()