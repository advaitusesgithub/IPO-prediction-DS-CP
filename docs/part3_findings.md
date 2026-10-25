# Final chronological evaluation

The frozen choices were fitted on 149 development IPOs using labels available before 2025-01-06 09:00:00+05:30. Both fitted pipelines stayed fixed for all 103 reserved IPOs. Historical context inputs can reflect already observable past outcomes; no test-label fitting or tuning occurs.

Price MAE: INR 49.86; RMSE: INR 73.49; MAPE: 14.97%. Announced-cap baseline MAE: INR 44.95. The selected price method is the training-median-premium baseline, not a learned forest.

Classifier accuracy: 67.96%; balanced accuracy: 50.00%; Apply precision: 67.96%; recall: 100.00%; F1: 0.8092. Always-Apply accuracy: 67.96%. Confusion counts: TN=0, FP=33, FN=0, TP=70. Apply predictions: 103; Skip predictions: 0.

90% accuracy reached: False. Better accuracy than always Apply: False. Report these measured results without treating the class imbalance as useful Skip detection. Probabilities are uncalibrated. Apply means predicted positive listing gain before costs and without allotment; no investment-profit claim is supported.

The local historical demonstration uses saved reserved predictions, with actual outcomes in a separate evaluation view. Models were reloaded and reproduced the recorded predictions. Reruns reuse the evaluation receipt and do not fit or score again. Changes to recorded inputs/artifacts fail verification.

Limitations remain: small sample, incomplete financial coverage, cohort-limited manager history, assumptions about source publication availability, and one weaker label-verification tier. Prior data preparation displayed aggregate full-cohort outcomes, so this reserved period is not described as unseen external data. Separate price and classification outputs have different targets and can disagree. No model settings or threshold were changed after evaluation. Submission packaging remains Part 4.
