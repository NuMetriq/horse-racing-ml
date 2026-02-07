\# Pipeline Contract



---



| Stage | Inputs | Outputs | Must NOT Know About |

| ----- | ------ | ------- | ------------------- |

| ingest | raw data sources | canonical race tables | labels, odds |

| features | canonical table | X, race\_ids, runner\_ids | outcomes |

| model | X | scores or logits | odds, policies |

| calibration | scores, outcomes (train only) | calibrated p\_win | odds, policies |

| policy | p\_win, odds, race structure | bets (runner, stake) | labels |

| evaluation | bets, outcomes | metrics, plots | model internals |

