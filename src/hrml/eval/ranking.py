from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class RankingEvalConfig:
    pred_path: Path = Path("outputs/pred_test.parquet")
    out_json: Path = Path("outputs/reports/metrics_ranking.json")
    out_md: Path = Path("outputs/reports/metrics_ranking.md")
    k_values: tuple[int, ...] = (3, 5)
    # Column inference: if these are missing we try to infer.
    race_col: Optional[str] = None
    score_col: Optional[str] = None
    winner_col: Optional[str] = None


# -------------------------
# Column inference helpers
# -------------------------
def _first_present(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_columns(df: pd.DataFrame, cfg: RankingEvalConfig) -> tuple[str, str, str]:
    """
    Infer required columns: race_id, score (higher=better), and winner indicator (1 for true winner).
    We prioritize explicit cfg values if provided.
    """
    race_col = cfg.race_col or _first_present(df, ["race_id", "raceID", "race", "event_id", "race_key"])
    if not race_col:
        raise ValueError(
            "Could not infer race id column. "
            "Tried: race_id, raceID, race, event_id, race_key. "
            "Pass RankingEvalConfig(race_col=...)."
        )

    # "score" should be a model score/prob; higher should mean more likely winner.
    score_col = cfg.score_col or _first_present(
        df,
        [
            "p_win",
            "pred_win_prob",
            "prob_win",
            "y_pred",
            "pred",
            "score",
            "logit",
        ],
    )
    if not score_col:
        raise ValueError(
            "Could not infer prediction score column. "
            "Tried: p_win, pred_win_prob, prob_win, y_pred, pred, score, logit. "
            "Pass RankingEvalConfig(score_col=...)."
        )

    # Winner indicator (binary): 1 if this runner is true winner.
    winner_col = cfg.winner_col or _first_present(
        df,
        [
            "is_winner",
            "winner",
            "y_true",
            "target",
            "label",
            "won",
            "win",
        ],
    )
    if not winner_col:
        raise ValueError(
            "Could not infer winner/label column. "
            "Tried: is_winner, winner, y_true, target, label, won, win. "
            "Pass RankingEvalConfig(winner_col=...)."
        )

    return race_col, score_col, winner_col


# -------------------------
# Metrics (race-aware)
# -------------------------
def _winner_rank_desc(scores: pd.Series, winner_mask: pd.Series) -> int:
    """
    Return 1-indexed rank of true winner when sorting by score descending.
    If multiple winners in a race, use the best (lowest) rank among them.
    """
    # Sort indices by score descending
    order = scores.sort_values(ascending=False).index
    # ranks: index -> 1..n
    ranks = pd.Series(range(1, len(order) + 1), index=order)
    winner_idx = winner_mask[winner_mask.astype(bool)].index
    if len(winner_idx) == 0:
        return -1  # no winner label found in this race
    return int(ranks.loc[winner_idx].min())


def _ndcg_for_single_relevant(rank: int, k: int) -> float:
    """
    NDCG@k for a single relevant item (the winner) with binary relevance.
    Ideal DCG is 1 (winner at rank 1), so NDCG simplifies to DCG.
    DCG = 1/log2(rank+1) if rank<=k else 0
    """
    if rank <= 0:
        return float("nan")
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _mrr(rank: int) -> float:
    if rank <= 0:
        return float("nan")
    return 1.0 / rank


def compute_race_ranking_metrics(
    df: pd.DataFrame,
    race_col: str,
    score_col: str,
    winner_col: str,
    k_values: Iterable[int] = (3, 5),
) -> dict:
    """
    Compute race-aware ranking metrics across races.
    Metrics are computed per race and then averaged.
    """
    k_values = tuple(sorted(set(int(k) for k in k_values)))

    per_race_rows = []
    missing_winner_races = 0
    multi_winner_races = 0

    for race_id, g in df.groupby(race_col, sort=False):
        y = g[winner_col]
        # Normalize winner indicator to {0,1}
        winner_mask = (y.astype(float) > 0.5)

        n_winners = int(winner_mask.sum())
        if n_winners == 0:
            missing_winner_races += 1
            continue
        if n_winners > 1:
            multi_winner_races += 1

        scores = g[score_col].astype(float)
        r = _winner_rank_desc(scores=scores, winner_mask=winner_mask)

        row = {
            "race_id": race_id,
            "n_runners": int(len(g)),
            "winner_rank": int(r),
            "mrr": float(_mrr(r)),
        }
        for k in k_values:
            row[f"ndcg@{k}"] = float(_ndcg_for_single_relevant(r, k))
        per_race_rows.append(row)

    per_race = pd.DataFrame(per_race_rows)

    if per_race.empty:
        raise ValueError(
            "No valid races found to evaluate ranking metrics. "
            f"missing_winner_races={missing_winner_races}"
        )

    # Aggregate
    agg = {
        "n_races_total": int(df[race_col].nunique()),
        "n_races_used": int(per_race["race_id"].nunique()),
        "n_races_skipped_no_winner": int(missing_winner_races),
        "n_races_multi_winner_label": int(multi_winner_races),
        "mean_winner_rank": float(per_race["winner_rank"].mean()),
        "median_winner_rank": float(per_race["winner_rank"].median()),
        "mrr": float(per_race["mrr"].mean()),
    }
    for k in k_values:
        agg[f"ndcg@{k}"] = float(per_race[f"ndcg@{k}"].mean())

    # Helpful distribution cut
    agg["pct_winner_in_top3"] = float((per_race["winner_rank"] <= 3).mean())
    agg["pct_winner_in_top5"] = float((per_race["winner_rank"] <= 5).mean())
    agg["pct_winner_top1"] = float((per_race["winner_rank"] == 1).mean())

    return {
        "columns": {
            "race_col": race_col,
            "score_col": score_col,
            "winner_col": winner_col,
        },
        "aggregate": agg,
        "per_race_preview": per_race.sort_values("winner_rank", ascending=False).head(10).to_dict(orient="records"),
    }


def _format_md(report: dict) -> str:
    cols = report["columns"]
    agg = report["aggregate"]

    lines = []
    lines.append("# Ranking Metrics (Race-aware)\n")
    lines.append("Computed per race (not per runner) using `pred_test.parquet`.\n")
    lines.append("## Columns used\n")
    lines.append(f"- race id: `{cols['race_col']}`\n")
    lines.append(f"- score: `{cols['score_col']}` (higher = ranked higher)\n")
    lines.append(f"- winner label: `{cols['winner_col']}` (truth)\n")

    lines.append("\n## Aggregate metrics\n")
    # Keep this stable and readable
    def f(x: float) -> str:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "NA"
        return f"{x:.6f}"

    # Order matters in reports
    keys = [
        "n_races_total",
        "n_races_used",
        "n_races_skipped_no_winner",
        "n_races_multi_winner_label",
        "mean_winner_rank",
        "median_winner_rank",
        "mrr",
    ]
    # Add ndcg keys in order
    ndcg_keys = sorted([k for k in agg.keys() if k.startswith("ndcg@")], key=lambda s: int(s.split("@")[1]))
    keys.extend(ndcg_keys)
    keys.extend(["pct_winner_top1", "pct_winner_in_top3", "pct_winner_in_top5"])
    keys.extend(["pct_winner_in_top3", "pct_winner_in_top5"])

    lines.append("| metric | value |\n")
    lines.append("|---|---:|\n")
    for k in keys:
        v = agg[k]
        if isinstance(v, (int,)):
            lines.append(f"| {k} | {v} |\n")
        else:
            lines.append(f"| {k} | {f(float(v))} |\n")

    lines.append("\n## Per-race preview (worst winner ranks)\n")
    lines.append("Top 10 races where the winner was ranked highest by the model.\n\n")
    preview = report.get("per_race_preview", [])
    if preview:
        cols_preview = ["race_id", "n_runners", "winner_rank", "mrr"] + [
            k for k in preview[0].keys() if k.startswith("ndcg@")
        ]
        lines.append("| " + " | ".join(cols_preview) + " |\n")
        lines.append("|" + "|".join(["---"] * len(cols_preview)) + "|\n")
        for row in preview:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols_preview) + " |\n")
    else:
        lines.append("_No preview available._\n")

    return "".join(lines)


def run_ranking_eval(cfg: RankingEvalConfig) -> dict:
    cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.out_md.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.pred_path)
    race_col, score_col, winner_col = infer_columns(df, cfg)

    report = compute_race_ranking_metrics(
        df=df,
        race_col=race_col,
        score_col=score_col,
        winner_col=winner_col,
        k_values=cfg.k_values,
    )

    cfg.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cfg.out_md.write_text(_format_md(report), encoding="utf-8")
    return report