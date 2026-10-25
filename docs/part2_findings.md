# Part 2 — Development modeling findings

## What was completed

Compared 10 regression and 9 classification candidates across four expanding 2024 folds: 76 fits, 90 unique validation IPOs. Fold training sizes are 59, 80, 94 and 121. All imputers/scalers were fitted on each fold's training rows only. The optional financial extension was compared using development data only. The Apply/Skip threshold stayed fixed at 0.5; no threshold search was performed.

## Opening-price regression

Selected: `regression_training_median_premium_core_baseline`. Pooled validation MAE: **INR 94.45**, RMSE: INR 158.02, MAPE: 17.97%.

Announced-cap baseline MAE: INR 108.48. Training-median-premium baseline MAE: INR 94.45. The premium baseline uses only training opening-price/announced-cap ratios. Final issue price is never an input.

Price error in rupees can be dominated by expensive IPOs; MAPE and the per-quarter results provide supporting context. Learned price predictions use a prespecified INR 0.01 physical floor.

## Positive-gain / Apply-Skip classification

Selected: `classification_random_forest_core_setting1`. Pooled validation accuracy: **77.78%**; balanced accuracy: 50.00%; Apply precision: 77.78%; Apply recall: 100.00%; F1: 0.8750.

Always-Apply validation accuracy: **77.78%** across these 90 rows. This differs from the earlier 78.5% rate across all 149 development rows. Best learned challenger: `classification_random_forest_core_setting1` (77.78% accuracy).

Selected-model confusion counts: TN=0, FP=20, FN=0, TP=70. Correctly identified 0 of 20 non-positive listings, while missing 0 of 70 positive listings. Predicted Apply: 90; Skip: 0.

The 90% accuracy aspiration was not reached on development validation. Raw model probabilities are not calibrated confidence estimates. Apply is a prediction of positive listing gain, without allotment or costs; it does not establish investment profitability.

## Selection and limitations

Regression was ranked by pooled MAE, then RMSE, fewer used predictors and candidate ID. Classification was ranked by pooled accuracy, then balanced accuracy, Brier score, fewer used predictors and candidate ID. Baselines were eligible to win. All candidates and results are retained; a stronger learned model was not assumed in advance.

These validation rows were used for model/feature selection, so selected scores are optimistic development estimates, not final performance. A quarter has only one non-positive listing; quarterly metrics are unstable. Sample size, missing financials, limited manager history and inherited point-in-time source assumptions remain limitations. Separate price/classification models can disagree because their targets differ.

## Part 3 handoff

`config/selected_models.json` freezes choices, threshold, input/code hashes and library versions. `models/validation/` contains selected and best-learned challenger fold pipelines with training/validation IDs; these are development models, not production models. Saved models were reloaded and reproduced their validation predictions. `outputs/part2/` holds full comparisons, per-fold metrics, per-IPO predictions, audits and error cases; `figures/part2/` holds five charts.

No model was fitted on the full 149-row development set yet. No 2025 reserved predictions or outcome metrics were computed. Part 3 will fit the frozen choices on 149 IPOs, evaluate once on the 103 reserved IPOs and build the local demonstration. The reserved period is not claimed as unseen external data because earlier preparation displayed full-cohort aggregate outcomes.
