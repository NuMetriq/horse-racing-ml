from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Deque, Dict, Tuple, Optional

@dataclass
class RecentStats:
    finishes: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    dates: Deque[int] = field(default_factory=lambda: deque(maxlen=10))  # store as ordinal ints

    def add(self, finish_pos: int, date_ordinal: int) -> None:
        self.finishes.append(finish_pos)
        self.dates.append(date_ordinal)

    def mean_finish_last(self, n: int) -> Optional[float]:
        if len(self.finishes) == 0:
            return None
        vals = list(self.finishes)[-n:]
        return sum(vals) / len(vals)

    def win_rate_last(self, n: int) -> Optional[float]:
        if len(self.finishes) == 0:
            return None
        vals = list(self.finishes)[-n:]
        return sum(1 for v in vals if v == 1) / len(vals)

    def days_since_last(self, current_date_ordinal: int) -> Optional[int]:
        if len(self.dates) == 0:
            return None
        return current_date_ordinal - self.dates[-1]

@dataclass
class ExpandingRate:
    wins: int = 0
    starts: int = 0

    def add(self, is_win: bool) -> None:
        self.starts += 1
        self.wins += int(is_win)

    def rate(self) -> Optional[float]:
        if self.starts == 0:
            return None
        return self.wins / self.starts

class HistoryStore:
    def __init__(self) -> None:
        self.horse_recent: Dict[str, RecentStats] = defaultdict(RecentStats)
        self.jockey_exp: Dict[str, ExpandingRate] = defaultdict(ExpandingRate)
        self.trainer_exp: Dict[str, ExpandingRate] = defaultdict(ExpandingRate)
        self.combo_exp: Dict[Tuple[str, str], ExpandingRate] = defaultdict(ExpandingRate)

    def features_for_runner(self, horse_id: str, jockey_id: str, trainer_id: str, race_date_ordinal: int) -> dict:
        h = self.horse_recent[horse_id]
        j = self.jockey_exp[jockey_id]
        t = self.trainer_exp[trainer_id]
        c = self.combo_exp[(jockey_id, trainer_id)]

        return {
            "horse_mean_finish_last5": h.mean_finish_last(5),
            "horse_win_rate_last10": h.win_rate_last(10),
            "horse_days_since_last": h.days_since_last(race_date_ordinal),
            "jockey_win_rate_exp": j.rate(),
            "trainer_win_rate_exp": t.rate(),
            "jockey_trainer_win_rate_exp": c.rate(),
        }

    def update_after_race(self, horse_id: str, jockey_id: str, trainer_id: str, finish_pos: int, race_date_ordinal: int) -> None:
        is_win = (finish_pos == 1)
        self.horse_recent[horse_id].add(finish_pos, race_date_ordinal)
        self.jockey_exp[jockey_id].add(is_win)
        self.trainer_exp[trainer_id].add(is_win)
        self.combo_exp[(jockey_id, trainer_id)].add(is_win)