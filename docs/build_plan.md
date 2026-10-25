# IPO Listing Prediction — Build Plan

Status: Parts 1–4 complete. Final models, reserved-period evaluation,
the local demonstration and submission materials are implemented.
The 90% accuracy aspiration was not achieved. Reserved accuracy is 68.0%, equal
to always Apply; the classifier does not demonstrate useful Skip detection.

## Aim

Use S0 information to predict listing opening price and classify positive
listing gain. Measure price error and classification quality honestly.
90% classification accuracy is an aspiration, not an acceptance gate or a
guaranteed result. The development always-Apply rate is approximately 78.5%.

## Part 1 — Inputs and setup

Bundle the prepared inputs and prior EDA evidence; define targets and feature
sets; verify fixed chronological memberships; install an isolated environment;
provide reusable loading/preprocessing code and an executed readiness notebook.
Completion requires passing data/split checks, leakage tests and a relocation
test. No predictive model or reserved-period scores are required here.

## Part 2 — Development modeling

Compare announced-cap and training-median-premium price baselines, Ridge and
a restrained Random Forest regressor. Define the median-premium baseline from
training opening-price/announced-cap ratios so it uses no unknown final price.
Compare always Apply, logistic regression and a restrained Random Forest
classifier. Use the four development folds for small, recorded comparisons
and any feature/threshold choices. Preprocessing must be fitted in each fold.

Primary regression metric: MAE in INR; secondary RMSE in INR and MAPE.
Classification: accuracy, precision, recall, F1 and confusion matrix.
Completion: an executed modeling notebook, validation comparisons and frozen
model choices. Do not access reserved metrics for model selection.

## Part 3 — Final evaluation and demo

Fit chosen models on the fixed 149-row development set and evaluate once on
the 103 reserved rows. Save pipelines and held-out predictions. Build a local
name-search interface with a historical IPO browser with predicted opening price, positive-gain
probability and Apply/Skip. Show actual outcomes separately. Demonstration
performance uses saved held-out predictions, never in-sample estimates.
Report achieved results even if below the aspiration.

## Part 4 — Delivery

Prepare the report, presentation, dependency lock, run instructions and folder
guide. Verify saved pipelines reload, demo output matches saved predictions
and required notebook outputs are present. Deliver after all four parts pass.

## Deferred extension

Day-wise stages, expanded anchor reputation, post-listing sell/hold,
allotment/cost backtests, publication-level inference and cloud/live-data
integration remain for the later full-scale extension.
