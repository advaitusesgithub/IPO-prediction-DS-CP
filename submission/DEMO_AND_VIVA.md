# Demonstration and explanation guide

## Suggested five-minute demonstration

1. State the goal: predict opening price and positive listing gain using S0
   information before bidding begins. S0 is 09:00 IST on bidding day 1.
2. Open the three executed notebooks. Point out missing-value handling,
   chronological membership and the simple baseline comparisons.
3. Launch the app. Enter Hyundai Motor India and press Predict. Show the results table,
   then use the historical list to select NSDL. Point out the prediction source labels.
4. Open About the model. Explain price MAE INR 49.86, accuracy 68.0% and
   baseline comparisons. Toggle the actual outcome and download the displayed
   prediction. Do not claim the saved historical results are live predictions.
5. Conclude that the workflow and demo work, but the classifier has not learned
   useful Skip detection. Future work needs stronger information and a fresh
   evaluation, not tuning to the existing reserved labels.

## Questions to be ready for

**Why not randomly split?** IPO conditions change over time. An earlier model
must never be trained using outcomes not yet available when a later IPO opens.

**Why is final issue price not a predictor?** It is not known at the pre-bid
cutoff. It is used only to define the historical gain target.

**Why did a baseline win price regression?** The announced price band already
contains much of the price-level information. The training-median premium
performed better than learned regressors on the prespecified validation MAE.

**Why keep it when announced cap wins final MAE?** Model choice was frozen
before reserved evaluation. Switching on the test result would invalidate the
claim that the test measured a previously selected method.

**Is MAPE 14.97% equivalent to 85.03% accuracy?** No. Percentage error is a
regression metric; subtracting it from 100 does not create classification
accuracy or a probability of correct prediction.

**What does 68.0% classifier accuracy mean here?** Seventy of 103 IPOs were
positive. The classifier predicted Apply for all 103, just like the majority
baseline: 70 true positives, 33 false positives, zero true negatives and zero
false negatives. Balanced accuracy is 50%; useful Skip detection is absent.

**Why was Random Forest selected over always Apply?** Decision accuracy and
balanced accuracy tied; its better probability Brier score broke the predefined
tie. This does not establish better decisions or calibrated confidence.

**Are actual outcomes in the demo leaked into prediction?** Predictions were
saved from the frozen pipelines. Outcomes are stored separately and loaded only on request. The app does no fitting.

**What is missing?** Day-wise stages, fully verified anchor reputation,
post-listing decisions, allotment/cost modeling and live integration are
deferred. Financial missingness and source publication-time assumptions remain.

**Was 90% achieved?** No. Report the measured result rather than changing labels,
splits or evaluation settings to meet an aspiration.
