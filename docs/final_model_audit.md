# Final machine-learning audit and refinement

## Audit outcome

The original pipeline's time ordering, label maturity and training-only preprocessing remain valid under its documented source assumptions. Two modeling weaknesses were identified: absolute price targets mix scale with listing premium, and accuracy-based classifier selection favors always Apply. No leakage-prone final subscription, final issue price, post-listing value or globally fitted transformation was added.

The second pass compared **48 candidates** (21 price / 27 classification) over **eight expanding 2024–2025 folds**, for **384 fits** and **194 out-of-fold validation IPOs**. Search families include Ridge, Huber, Random Forest, Extra Trees and histogram gradient boosting. Price models learn opening-price/announced-cap premium and use cap-proportional training weights. A cap-weighted median baseline is also checked. Both 8- and 12-variable shared input sets avoid unavailable external VIX, unverified manager and financial predictors.

## Development versus test

All **253 original 2023–2025 IPOs are now development data**. The already inspected 2025 period is not represented as a fresh test. Quarterly models use only earlier S0 dates and strictly mature labels. Thresholds and model settings are selected from development out-of-fold results only.

Initial unconstrained balanced-accuracy selection cut accuracy too severely. Before any external scoring, validation-only guardrails were added: accuracy no worse than majority accuracy minus five percentage points, positive-class recall at least 75%, and non-positive specificity at least 10%. That adjustment and the initial choice are preserved in `unconstrained_selection.json`; it was not made after examining new-test scores. Guarded selected forest: depth 3, minimum leaf 5, 200 trees, balanced subsample weighting, context predictors and threshold 0.45.

The actual new chronological test contains **17 previously unused early-2026 IPOs**. It uses small existing source snapshots, not fabricated or randomized samples. Of 94 verified mainboard 2026 records, 48 lack a matching profile, 28 are after the available market history and one lacks an explicit original band. Nifty history ends 30 March 2026. Included previous-session observations are at most seven days old. Past observable listing outcomes update history inputs strictly before each S0. Exact pre-bid publication times remain an assumption, and source verification is inherited from the existing snapshots.

## Same-new-data comparison

| Method | Price MAE (INR) | Classifier accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Original v1, trained on 149 IPOs | 68.97 | 29.41% | 50.00% |
| Same v1 architectures, refitted on 253 | 50.95 | 29.41% | 50.00% |
| Refined v2, trained on 253 | 50.95 | 29.41% | 50.00% |
| Announced-cap / always-Apply baselines | 30.67 | 29.41% | 50.00% |

The price improvement versus the original fit is **26.1%**, but it is explained by more training history; tuning did not find a better price method than the refitted median-premium baseline. Announced cap alone still wins MAE. Do not compare these 17-row errors directly with the older 103-row 2025 errors as though the test population were unchanged.

The refined classifier still predicts Apply for all 17 new IPOs: five true positives, twelve false positives, no true negatives and no false negatives. Its Brier score improves from 0.4315 to 0.2730, but probabilities remain uncalibrated and decision accuracy does not improve. **Useful Skip detection and 90% accuracy remain unproven.** A small source-covered early-2026 sample cannot establish full-year generalization. No further tuning was performed after these test results.

The original v1 classifier relies on VIX/manager features unavailable for this new sample; its fitted imputer handles them as missing. The v2 model avoids those unsupported inputs. This comparability limitation is explicit rather than silently using stale or future data. The always-Apply and announced-cap baselines provide common fully observed comparisons.

## Delivery and runtime behavior

Original models, 2025 predictions, frozen choices and evaluation receipt are unchanged. The 17 new 2026 audit predictions are available in the app under a separate **2026 audit prediction** badge. Historical v1 evaluation and validation examples retain their original outputs. The new branch uses its recorded 0.45 threshold rather than the old hardcoded 0.5. There is no live web retrieval or claim of reliable investment advice.

The tuned model artifacts are in `models/refined/`, their freeze record is `config/refined_selection.json`, full validation and external comparisons are in `outputs/refinement/`, and all 2026 eligibility/provenance tables are in `data/refinement/`. `04_model_audit_and_refinement.ipynb` contains executed explanations, code, tables and charts. A receipt prevents silent changes to the recorded new-data evaluation.

This is a completed, reproducible modeling audit—not a claim that model complexity solved a weak-signal problem. The proper next research step would require stronger point-in-time information and another untouched evaluation period, rather than more attempts to maximize a score on these known outcomes.
