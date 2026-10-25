# S0 input contract

## Unit and timing

One row per mainboard IPO; 253 IPOs first listed in 2023–2025. `ipo_id` is
the unique identifier. S0 is 09:00 Asia/Kolkata on the first bidding day.
IDs, names and timestamps are metadata, never model predictors.

## Inputs

`config/source_schema.json` preserves the original 52-predictor whitelist.
`config/features.json` specifies the ordered 12-variable core and seven optional
financial predictors. Use `get_xy(dataset, ids, feature_set)` to select X and y
explicitly. The preprocessing guard rejects extra, reordered, nonnumeric or
infinite inputs. Missing numeric values are permitted.

Predictors use announced issue terms, market observations before S0 and
historical IPO/manager outcomes available before S0. Manager history begins
with the collected cohort. Financial coverage is incomplete; keep missing
values and fit transformations inside each training fold. No verified anchor
predictor or subscription feature is admitted.

## Targets

- Regression: `listing_open_inr`, the recorded opening price on the original
  listing venue (NSE when available; BSE for BSE-only listings).
- Classification: `target_positive_listing_gain = int(listing_open_inr >
  final_issue_price_inr)`. Equality is a non-positive outcome.
- The existing decimal gain label is retained for audit and context.

Final issue price, listing prices, target columns, listing date and label
availability belong to labels/audit metadata. Do not feed them to an S0 model.
Predict opening price directly. Do not multiply a gain prediction by a future
final issue price to produce an S0 price prediction.

Apply/Skip represents predicted positive listing gain before fees and without
allotment probabilities. Regression and classification are separate models and
may disagree; the future interface must describe their separate definitions.

## Chronological contract

The development boundary is 1 January 2025. Preserve the existing 149 mature
development rows, 103 reserved 2025 rows and one excluded boundary row.
Final fitting cutoff: 6 January 2025 at 09:00 IST. Never randomly split.

| 2024 validation fold | Training IPOs | Validation IPOs | Unmatured rows purged |
|---|---:|---:|---:|
| Q1 | 59 | 22 | 0 |
| Q2 | 80 | 14 | 1 |
| Q3 | 94 | 26 | 1 |
| Q4 | 121 | 28 | 0 |

Training requires both an earlier S0 and `label_available_at < fit_cutoff`.
Model/preprocessing settings stay frozen within each validation quarter.
Freeze feature/model/threshold choices before evaluating 2025. Past observable
outcomes can update already-defined historical context features; this does not
authorize fitting a model or tuning on reserved outcomes.

## Validation and lineage

Loading checks cover unique IDs and exact cohort joins, matrix schema,
finite positive target prices, consistent gain/classification formulas,
timezone/timing and no infinite predictors. Split generation must reproduce
every previously frozen membership. `data/provenance/bundled_inputs.csv`
records SHA-256 checksums of the bundled input snapshots.

`reports/prior_preparation/` contains historical extraction audits and original
source hashes. Paths there are provenance references, not runtime dependencies.
Heavy original PDFs and the abandoned day-wise/SME sources are not required
to load the prepared inputs or reproduce Part 1 and later model runs.
Rebuilding the earlier document extraction from scratch would require those
original source archives; that extraction is outside this package's runtime.

## Limitations retained

Announced bands/lots and dated pre-bid RHPs have availability assumptions;
exact historic publication timestamps are absent. Financial extraction covers
only part of the cohort. Labels use a third-party snapshot of exchange
verification, including one explicitly weaker verification tier. Historical
all-cohort aggregate outcomes were seen in preparation, so the reserved period
is not described as completely unseen external data. These limits must appear
in the eventual report. Original source licensing still applies.
