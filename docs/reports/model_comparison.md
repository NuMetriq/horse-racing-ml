# Model Comparison: Win Probability vs Race Ranking



This report compares three modeling approaches for predicting horse race outcomes:



1. **Race Softmax (v1.1.0)** — winner-only probability model  

2. **Pairwise Learning-to-Rank** — comparison-based ranking model  

3. **Plackett–Luce (v1.2.0)** — race-level ranking model with calibrated win probabilities  



All results are evaluated **out of sample** using a fixed temporal split and are computed **per race**, not per runner.



The goal of this comparison is to understand the tradeoffs between:

- probabilistic calibration,

- race-level ordering quality,

- and modeling assumptions about race outcome structure.



---



## Evaluation Summary (Out of Sample)



| Model | Logloss | Brier | ECE | MRR | NDCG@3 | NDCG@5 | Mean Winner Rank |

|------|--------:|------:|----:|----:|-------:|-------:|-----------------:|

| Race Softmax (v1.1.0) | 0.0845 | 0.0246 | 0.0036 | 0.9031 | 0.9171 | 0.9239 | 1.304 |

| Pairwise Ranking | — | — | — | 0.8908 | 0.9036 | 0.9121 | 1.378 |

| **Plackett–Luce (calibrated)** | — | — | **0.0052** | 0.8994 | 0.9120 | 0.9186 | 1.346 |



**Notes:**

- Logloss and Brier are not reported for pairwise or PL models, as they do not directly optimize a binary probability target.

- Plackett–Luce win probabilities are calibrated post hoc using validation-based temperature scaling.

- PL metrics are computed on races with complete finish-order information.



---



## Model-by-Model Interpretation



### Race Softmax (v1.1.0)

The race-softmax model directly optimizes the probability that each horse wins, normalized within a race.



**Strengths**

- Excellent probabilistic calibration out of the box (ECE ≈ 0.0036)

- Strong winner-prediction performance

- Simple, stable, and interpretable



**Limitations**

- Optimizes only the win event

- Does not explicitly model the structure of full race outcomes

- Ranking quality emerges indirectly rather than by design



This model remains a strong baseline for win-probability estimation.



---



### Pairwise Learning-to-Rank

The pairwise model learns to prefer winners over non-winners via within-race comparisons.



**Strengths**

- Improves ordering quality relative to naive baselines

- Fast to train and simple to implement

- Directly targets ranking objectives



**Limitations**

- No probabilistic semantics

- Cannot produce calibrated win probabilities

- Ranking performance lags both softmax and PL models



Pairwise ranking is useful as a diagnostic and baseline, but not as a primary outcome model.



---



### Plackett–Luce Race Model (v1.2.0)

The Plackett–Luce (PL) model explicitly represents the race as an ordered outcome, optimizing a top-K ranking likelihood.



**Strengths**

- Explicitly models race outcome structure

- Competitive ranking performance (NDCG@5 ≈ 0.919)

- Strong MRR and mean winner rank

- Supports richer outputs (expected rank, place probabilities)



**Calibration**

- Raw PL scores are not probabilistic

- Validation-based temperature scaling restores calibration discipline

- Final calibrated ECE ≈ 0.0052, comparable to softmax



**Limitations**

- Requires complete finish-order data

- Slightly more complex training pipeline

- Ranking performance does not strictly dominate softmax in all metrics



---



## Key Takeaways



- **Winner-only softmax** remains the most direct and best-calibrated model for win probability estimation.

- **Pairwise ranking** improves ordering but lacks probabilistic meaning.

- **Plackett–Luce** provides the best balance between:

    - explicit race-level structure,

    - strong ranking quality,

    - and interpretable, calibrated probabilities (when paired with temperature scaling).



---



## Recommendation



For v1.2.0 and beyond:



- Use **Race Softmax** as a strong baseline for pure win-probability tasks.

- Use **Plackett–Luce (calibrated)** when race ordering, relative strength, or downstream rank-based analysis is important.

- Treat pairwise ranking as a fast diagnostic or benchmarking tool rather than a production model.



Plackett–Luce represents a meaningful conceptual upgrade: it models races as structured outcomes rather than independent win events, while preserving calibration discipline through explicit post-hoc correction.



---



## Non-Goals



These models are not designed for:

- betting strategy optimization,

- profit maximization,

- or market inefficiency exploitation.



The focus is on **predictive structure, calibration, and methodological clarity**.

