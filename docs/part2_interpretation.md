# What the development results establish

The completed comparison does not establish a 90%-accurate Apply/Skip system.
On the 90 development validation IPOs, 70 had positive listing gains. Always
predicting Apply therefore achieves 77.8% accuracy. The selected core Random
Forest makes exactly those same decisions: 70 correct positive predictions,
20 incorrect Apply predictions and no Skip predictions. Its better Brier score
breaks the prespecified selection tie, but gives no decision improvement over
always Apply. Its probabilities have not been calibrated.

The weighted core forest is a recorded alternative: it correctly identifies
11 of the 20 non-positive listings but misses 23 of the 70 positive listings,
with 64.4% accuracy and 61.1% balanced accuracy. This shows the trade-off in
identifying losers; it is not a reason to change the fixed selection criterion
after seeing results. The optional financial predictors did not improve the
winning models under the specified ranking.

For opening price, the training-median-premium baseline achieves MAE INR 94.45,
RMSE INR 158.02 and MAPE 17.97%. It beats announced cap alone (MAE INR 108.48)
and the learned regressor challengers on the primary criterion. Price bands
already contain much of the information about price level, so comparison
against a sensible cap-based baseline is essential.

These are development scores used for selection, not final test results.
The later report and demo must retain these distinctions, even if the final
evaluation changes the scores. Nothing here establishes net investment profit.
No additional dataset is required to complete the planned evaluation and demo.
