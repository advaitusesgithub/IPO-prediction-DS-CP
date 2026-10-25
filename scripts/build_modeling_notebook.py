"""Create the reproducible development-modeling notebook; execute it separately."""
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parents[1]
cells = []
def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))
def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# S0 Development Modeling and Model Selection

Part 2 compares opening-price regressors and positive-listing-gain classifiers.
Only the fixed development population is supplied to fitting/scoring. The 2025
reserved set is not predicted or scored here. Re-running this notebook repeats
the same prespecified search; it does not authorize additional tuning.

The four 2024 folds contain 90 unique validation IPOs and training sizes of
59, 80, 94 and 121. Outcomes must be available before each fit cutoff.
Earlier data preparation exposed aggregate full-cohort outcomes, so the reserved
period is not claimed as completely unseen external data.""")
code("""from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown

root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "config/evaluation.json").exists() and (p / "ipo_prediction").is_dir())
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from ipo_prediction.train import run, development_inputs
from ipo_prediction.data import feature_names, get_xy
from ipo_prediction.models import positive_probability
plan = json.loads((root / "config/model_search.json").read_text())
print("Fixed classification threshold:", plan["classification_threshold"])
print("Selection uses pooled validation MAE for price and accuracy for classification.")
print("No reserved-period outcomes are scored.")""")
md("""## 1. Prespecified comparisons and leakage controls

Ten regression candidates: announced cap, a training-only median-premium
baseline, Ridge with two alpha values, and Random Forest with two restrained
settings. Learned families each compare 12 core predictors against a 19-variable
financial extension. Nine classification candidates: always Apply, logistic
regression with/without balanced weights and two Random Forest settings, again
comparing both feature sets. Settings are recorded in `config/model_search.json`.

The premium baseline predicts announced cap × median(training opening price /
training announced cap), never using a future final issue price. Learned prices
have a prespecified physical floor of INR 0.01. All learned preprocessing is
inside the fitted model pipeline. Extra outcome/subscription inputs are rejected.

Threshold 0.5 was fixed before comparisons. Regression ranking uses pooled MAE,
then RMSE and simplicity. Classification uses pooled accuracy, then balanced
accuracy, Brier score and simplicity. Baselines can win. These selection metrics
are not unbiased final performance estimates.""")
code("""summary = run()
out = root / "outputs/part2"
registry = json.loads((out / "candidate_registry.json").read_text())
display(pd.DataFrame(registry)[["candidate_id", "task", "feature_set", "params"]])
display(pd.Series(summary, name="Part 2 completion").to_frame())""")
md("""## 2. Opening-price regression

MAE and RMSE are measured in rupees; MAPE is a percentage error, not model
accuracy. Absolute rupee loss weights high-priced IPO mistakes more strongly.
Read MAPE and the quarter results alongside the primary MAE ranking.""")
code("""regression = pd.read_csv(out / "regression_comparison.csv")
display(regression[["rank", "candidate_id", "feature_set", "mae_inr", "rmse_inr", "mape_pct", "n"]].round(3))
display(Image(filename=str(root / "figures/part2/01_regression_comparison.png")))
display(Image(filename=str(root / "figures/part2/04_selected_price_diagnostics.png")))""")
md("""## 3. Positive-gain classification

Apply means predicted opening price exceeds final issue price, before costs and
without modeling allotment. A model can match the majority baseline while
identifying no losing listings. Therefore include balanced accuracy, Skip
specificity and full confusion counts alongside accuracy. Raw class probabilities
are not calibrated investment confidence estimates.""")
code("""classification = pd.read_csv(out / "classification_comparison.csv")
display(classification[["rank", "candidate_id", "accuracy", "balanced_accuracy", "precision_apply", "recall_apply", "f1_apply", "specificity_skip", "brier", "tn", "fp", "fn", "tp"]].round(4))
display(Image(filename=str(root / "figures/part2/02_classification_comparison.png")))
display(Image(filename=str(root / "figures/part2/05_selected_classification_diagnostics.png")))
chosen = classification.iloc[0]
baseline = classification.loc[classification.is_baseline].iloc[0]
display(Markdown(f"**Selected validation accuracy: {chosen.accuracy:.2%}; always-Apply baseline: {baseline.accuracy:.2%}.**"))
if int(chosen.predicted_skip) == 0:
    display(Markdown("**The selected classifier predicts Apply for every validation IPO. It has not demonstrated a useful Apply/Skip distinction.**"))
if chosen.accuracy < .9:
    print("The 90% accuracy aspiration was not reached in development validation.")""")
md("""## 4. Quarter stability and comparison limits

Each IPO appears in validation once, and each quarter uses a fresh model with
earlier mature training labels. Pooled ranking weights each validation IPO
equally. Quarterly scores show instability but are not extra independent tests;
one quarter contains only one non-positive listing.""")
code("""folds = pd.read_csv(out / "fold_metrics.csv")
selected = json.loads((root / "config/selected_models.json").read_text())
selected_ids = [selected["selection"][task]["candidate_id"] for task in ["regression", "classification"]]
display(folds.loc[folds.candidate_id.isin(selected_ids)].dropna(axis=1, how="all").round(4))
display(Image(filename=str(root / "figures/part2/03_fold_stability.png")))""")
md("""## 5. Audits and representative mistakes

Audit every fit rather than assuming time order from row order. Review large
price errors and incorrect classifications without dropping or relabeling them.
These errors remain part of validation. The source datasets are unchanged.""")
code("""audit = pd.read_csv(out / "training_audit.csv")
display(audit.groupby("fold_id")[["training_IPOs", "validation_IPOs", "reserved_rows_used"]].first())
display(pd.read_csv(out / "validation_checks.csv"))
display(pd.read_csv(out / "largest_price_errors.csv").head(5))
display(pd.read_csv(out / "classification_errors.csv").head(5))""")
md("""## 6. Saved development pipelines and reload verification

Save selected and best-learned challenger models for each development fold,
including training and validation IDs. These are not final 149-row models and
must not be presented as production models. Loading them should reproduce the
stored validation predictions without refitting.""")
code("""import joblib
artifacts = pd.read_csv(out / "model_artifacts.csv")
display(artifacts[["candidate_id", "fold_id", "reloaded_predictions_match"]])
classifier_id = selected["selection"]["classification"]["candidate_id"]
row = artifacts.loc[artifacts.candidate_id.eq(classifier_id)].iloc[0]
bundle = joblib.load(root / row.path)
dev, walk = development_inputs()
x, _ = get_xy(dev, bundle["validation_ids"], bundle["spec"]["feature_set"])
replayed = positive_probability(bundle["model"], x)
recorded = pd.read_parquet(out / "validation_predictions.parquet")
expected = recorded.loc[recorded.candidate_id.eq(classifier_id) & recorded.fold_id.eq(bundle["fold_id"])]
np.testing.assert_allclose(replayed, expected.prediction, rtol=1e-12, atol=1e-12)
print("Saved classifier reproduced its fold validation predictions.")
print("Final models fitted:", summary["final_models_fitted"])
print("Reserved outcome metrics computed:", summary["reserved_outcome_metrics_computed"])""")
md("""## 7. Results and handoff

The simple premium price baseline outperformed the learned price models on
the prespecified MAE criterion. The selected Random Forest classifier tied the
always-Apply decision accuracy and identified no non-positive listings at the
fixed threshold. It won the tie through its probability score; that does not
establish better Apply/Skip decisions. The weighted-forest comparisons show the
accuracy/specificity trade-off and remain recorded rather than substituted
after selection.

The small sample, missing financials, restricted manager history and source
availability assumptions constrain what can be concluded. Do not claim 90%
accuracy, investment profitability or independent final performance from this
notebook. The selected scores were used to choose the models.

Part 3 uses the frozen `config/selected_models.json` specifications, fits once on
the 149 development IPOs, evaluates on the 103 reserved IPOs, and builds the
historical local demo. No search expansion or threshold change is authorized
by an inconvenient validation result.""")
code("""display(Markdown((root / "docs/part2_findings.md").read_text()))
for task in ["regression", "classification"]:
    spec = selected["selection"][task]
    print(task, "selected:", spec["candidate_id"])
    print("Best learned challenger:", spec["best_learned_candidate_id"])
assert summary["reserved_outcome_metrics_computed"] is False
assert summary["final_models_fitted"] is False
print("Part 2 complete: choices frozen; reserved evaluation remains for Part 3.")""")
notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (local environment)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
destination = root / "notebooks/02_development_modeling.ipynb"
nbf.write(notebook, destination)
print("Created development modeling notebook.")
