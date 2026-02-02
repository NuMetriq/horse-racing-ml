# Ranking Metrics (Race-aware)
Computed per race (not per runner) using `pred_test.parquet`.
## Columns used
- race id: `race_id`
- score: `score` (higher = ranked higher)
- winner label: `is_winner` (truth)

## Aggregate metrics
| metric | value |
|---|---:|
| n_races_total | 13607 |
| n_races_used | 13607 |
| n_races_skipped_no_winner | 0 |
| n_races_multi_winner_label | 0 |
| mean_winner_rank | 1.378261 |
| median_winner_rank | 1.000000 |
| mrr | 0.890820 |
| ndcg@3 | 0.903648 |
| ndcg@5 | 0.912090 |
| pct_winner_top1 | 0.815830 |
| pct_winner_in_top3 | 0.960829 |
| pct_winner_in_top5 | 0.981186 |
| pct_winner_in_top3 | 0.960829 |
| pct_winner_in_top5 | 0.981186 |

## Per-race preview (worst winner ranks)
Top 10 races where the winner was ranked highest by the model.

| race_id | n_runners | winner_rank | mrr | ndcg@3 | ndcg@5 |
|---|---|---|---|---|---|
| 894652 | 16 | 16 | 0.0625 | 0.0 | 0.0 |
| 905098 | 18 | 16 | 0.0625 | 0.0 | 0.0 |
| 899517 | 16 | 15 | 0.06666666666666667 | 0.0 | 0.0 |
| 902839 | 16 | 15 | 0.06666666666666667 | 0.0 | 0.0 |
| 895539 | 18 | 15 | 0.06666666666666667 | 0.0 | 0.0 |
| 897939 | 17 | 15 | 0.06666666666666667 | 0.0 | 0.0 |
| 894256 | 15 | 14 | 0.07142857142857142 | 0.0 | 0.0 |
| 893102 | 14 | 14 | 0.07142857142857142 | 0.0 | 0.0 |
| 894651 | 14 | 14 | 0.07142857142857142 | 0.0 | 0.0 |
| 899518 | 15 | 13 | 0.07692307692307693 | 0.0 | 0.0 |
