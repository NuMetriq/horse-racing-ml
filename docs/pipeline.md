# Pipeline Contract



| Stage       | Inputs                        | Outputs                 | Must NOT Know About |
|-------------|-------------------------------|-------------------------|---------------------|
| ingest      | raw data sources              | canonical race table    | labels, odds        |
| features    | canonical table               | X, race_ids, runner_ids | outcomes            |
| model       | X                             | scores or logits        | odds, policies      |
| calibration | scores, outcomes (train only) | calibrated p_win        | odds, policies      |
| policy      | p_win, odds, race structure   | bets (runner, stake)    | labels              |
| evaluation  | bets, outcomes                | metrics, plots          | model internals     |

