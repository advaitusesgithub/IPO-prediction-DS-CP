# S0 EDA findings and evaluation proposal

Development uses 149 IPOs with pre-2025 S0 cutoffs and mature pre-2025 labels.
Reserve 103 S0-in-2025 IPOs for final evaluation. 1 December-to-January
boundary IPO is kept separate because its label was not yet observable. Basic all-cohort target
summaries were already shown in data preparation; we do not claim this reserved period is unseen external data.

Development median listing gain is 16.77%, mean 27.84%, with a positive-gain
share of 78.5%. The target is right-skewed (sample skew 1.67); all
9 IQR-flagged extremes are retained. These are descriptive outcomes, not model scores.

Annual income and PAT are observed for 76/149 development IPOs.
Financial missingness is a source/extraction limitation, not evidence of zero activity. Definitions
of income and attributable PAT vary and must remain documented. Manager reputation is observed for
112/149 IPOs; the history starts with this cohort.

There are 2 groups of exactly identical development columns and 43
feature pairs with absolute Spearman correlation >= 0.90. Keep the original 52-feature dataset;
start later modelling with the proposed 12-variable core, with a seven-variable financial extension
available for a separate comparison. This proposal uses meaning, coverage and redundancy, not a
ranking by target correlation. Complete-case deletion would shrink the usable sample substantially.

Use four expanding validation quarters in 2024. Training counts are
59, 80, 94, 121 and validation counts are
22, 14, 26, 28. There are 2 training memberships
excluded for immature labels at quarter boundaries. Final training has 149 IPOs,
with a fit cutoff of 2025-01-06T09:00:00+05:30 and 1 immature pre-test row excluded.

For an S0 implementation, a training-median baseline, Ridge and a restrained Random Forest
provide a manageable proposed comparison. Use MAE in percentage points as primary metric; RMSE
and rank correlation are secondary. Fit imputation, missing indicators and scaling within each
training fold, then freeze settings before final 2025 evaluation. No models or preprocessing were fitted here.

Limitations: small sample, incomplete financials, limited manager history, pre-bid availability
assumptions inherited from preparation, and no verified anchor predictors. Pooled associations may
reflect issuer mix, market regimes and time trends. They are neither causal effects nor validated
predictive performance. Publication scope, statistical tests, extra model families and application scope can
be discussed later; the original project proposal is unchanged.
