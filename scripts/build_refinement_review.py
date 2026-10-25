"""Write the final modeling audit and its reproducible evidence notebook."""
from pathlib import Path
import json,os
import nbformat as nbf
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=ROOT/'outputs/refinement'
summary=json.loads((out/'summary.json').read_text())
metrics=pd.read_csv(out/'external_comparison.csv')
reg=metrics.loc[metrics.task.eq('regression')]
clf=metrics.loc[metrics.task.eq('classification')]
figures=ROOT/'figures/refinement';figures.mkdir(parents=True,exist_ok=True)
labels={'refined_v2':'Refined / refreshed v2','original_v1_149_train':'Original v1 (149 train)',
        'original_architecture_253_train':'Same v1 method (253 train)',
        'announced_cap_baseline':'Announced cap','always_apply_baseline':'Always Apply'}
fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].barh([labels[x] for x in reg.method],reg.mae_inr,color=['#277567','#8d9bad','#5b80a1','#abb59b'])
axes[0].set(xlabel='MAE (INR, lower is better)',title='Same 17 new 2026 IPOs: price')
axes[0].invert_yaxis()
axes[1].barh([labels[x] for x in clf.method],clf.accuracy,color='#5b80a1')
axes[1].set(xlim=(0,1),xlabel='Accuracy',title='Same 17 IPOs: no decision improvement')
axes[1].invert_yaxis()
fig.tight_layout();fig.savefig(figures/'before_after_same_new_data.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4))
ax.bar(['True negative','False positive','False negative','True positive'],[0,12,0,5],color=['#277567','#bd785c','#bd785c','#5b80a1'])
ax.set(ylabel='IPOs',title='Refined classifier: 2026 external confusion counts')
fig.tight_layout();fig.savefig(figures/'external_confusion_counts.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
rs=pd.read_csv(out/'regression_search.csv').head(8)
cs=pd.read_csv(out/'guarded_classification_search.csv').head(8)
axes[0].barh(rs.candidate_id,rs.mae_inr,color='#277567');axes[0].invert_yaxis();axes[0].tick_params(axis='y',labelsize=7)
axes[0].set(xlabel='Development MAE (INR)',title='Top price candidates: development only')
axes[1].scatter(cs.accuracy,cs.balanced_accuracy,color='#277567')
axes[1].set(xlabel='Development accuracy',ylabel='Balanced accuracy',title='Guarded classifier choices: development only')
fig.tight_layout();fig.savefig(figures/'development_comparisons.png',dpi=150);plt.close(fig)
old=reg.loc[reg.method.eq('original_v1_149_train')].iloc[0];new=reg.loc[reg.method.eq('refined_v2')].iloc[0]
reduction=(old.mae_inr-new.mae_inr)/old.mae_inr*100
text=f'''# Final machine-learning audit and refinement

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

The price improvement versus the original fit is **{reduction:.1f}%**, but it is explained by more training history; tuning did not find a better price method than the refitted median-premium baseline. Announced cap alone still wins MAE. Do not compare these 17-row errors directly with the older 103-row 2025 errors as though the test population were unchanged.

The refined classifier still predicts Apply for all 17 new IPOs: five true positives, twelve false positives, no true negatives and no false negatives. Its Brier score improves from 0.4315 to 0.2730, but probabilities remain uncalibrated and decision accuracy does not improve. **Useful Skip detection and 90% accuracy remain unproven.** A small source-covered early-2026 sample cannot establish full-year generalization. No further tuning was performed after these test results.

The original v1 classifier relies on VIX/manager features unavailable for this new sample; its fitted imputer handles them as missing. The v2 model avoids those unsupported inputs. This comparability limitation is explicit rather than silently using stale or future data. The always-Apply and announced-cap baselines provide common fully observed comparisons.

## Delivery and runtime behavior

Original models, 2025 predictions, frozen choices and evaluation receipt are unchanged. The 17 new 2026 audit predictions are available in the app under a separate **2026 audit prediction** badge. Historical v1 evaluation and validation examples retain their original outputs. The new branch uses its recorded 0.45 threshold rather than the old hardcoded 0.5. There is no live web retrieval or claim of reliable investment advice.

The tuned model artifacts are in `models/refined/`, their freeze record is `config/refined_selection.json`, full validation and external comparisons are in `outputs/refinement/`, and all 2026 eligibility/provenance tables are in `data/refinement/`. `04_model_audit_and_refinement.ipynb` contains executed explanations, code, tables and charts. A receipt prevents silent changes to the recorded new-data evaluation.

This is a completed, reproducible modeling audit—not a claim that model complexity solved a weak-signal problem. The proper next research step would require stronger point-in-time information and another untouched evaluation period, rather than more attempts to maximize a score on these known outcomes.
'''
(ROOT/'docs/final_model_audit.md').write_text(text)
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('# Final Model Audit and Refinement\n\nChronological hyperparameter comparison, unused 2026 data, baseline checks and honest before/after results. The original 2025 evaluation remains historical evidence; it is now exposed development data for this revision.')
code('''from pathlib import Path
import sys,json,inspect
import pandas as pd
from IPython.display import display,Image,Markdown
root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'config/refined_selection.json').exists())
if str(root) not in sys.path:sys.path.insert(0,str(root))
from ipo_prediction.refinement_models import RelativePriceRegressor,make_revision_model
from ipo_prediction.refine import chronological_folds,evaluate
from ipo_prediction.data import load_dataset
out=root/'outputs/refinement'
plan=json.loads((root/'config/refinement_plan.json').read_text())
display(pd.Series(plan,name='Refinement protocol').to_frame())''')
md('## 1. Correct temporal train / test split\n\nUse eight earlier-data folds for development and the first eligible 2026 S0 for the final training cutoff. Randomly splitting future IPOs into past training would invalidate this decision-time study.')
code('''old=load_dataset()
folds=chronological_folds(old)
display(pd.DataFrame([{'fold':f,'cutoff':c,'train':len(t),'validation':len(v)} for f,c,t,v in folds]))
new=pd.read_parquet(root/'data/refinement/external_2026.parquet')
assert set(old.index).isdisjoint(new.ipo_id)
assert old.label_available_at.lt(new.s0_cutoff_at.min()).all()
display(pd.read_csv(root/'data/refinement/external_eligibility.csv').status.value_counts().to_frame('IPOs'))
display(new[['company_name','s0_cutoff_at','market_source_date','market_source_age_days','latest_history_label']])''')
md('## 2. Actual estimator and target changes\n\nRelative premium separates price scale from listing performance. Cap-proportional sample weights make absolute-error fitting reflect rupee loss. Missing values and scaling remain fitted inside training folds. The original immutable estimator module is preserved.')
code('''print(inspect.getsource(RelativePriceRegressor))
registry=json.loads((out/'candidate_registry.json').read_text())
display(pd.DataFrame(registry))
audit=pd.read_csv(out/'training_audit.csv')
print('Candidates:',len(registry),'candidate/fold fits:',len(audit))
assert audit.external_rows_used.eq(0).all()
assert (pd.to_datetime(audit.latest_training_label)<pd.to_datetime(audit.cutoff)).all()''')
md('## 3. Development tuning and threshold trade-offs\n\nSelected scores use validation labels for selection and are optimistic. Practical guardrails were set on these development results before scoring the new 2026 data; the unconstrained attempt is retained. The new test was not used to set a threshold.')
code('''display(pd.read_csv(out/'regression_search.csv').head(10))
display(pd.read_csv(out/'classification_search.csv').head(10))
display(pd.read_csv(out/'guarded_classification_search.csv').head(10))
display(Image(filename=str(root/'figures/refinement/development_comparisons.png')))
selected=json.loads((root/'config/refined_selection.json').read_text())
display(pd.Series(selected['selection'],name='Frozen choices').to_frame())''')
md('## 4. Same-new-data comparison\n\nThe following call verifies and reuses the recorded external evaluation. It does not refit, tune or rescore models. Both the unchanged 149-row original and a 253-row refit are compared on exactly the same 17 new IPOs, separating the effect of more data from tuning.')
code('''summary=evaluate()
display(pd.read_csv(out/'external_comparison.csv').round(4))
display(Image(filename=str(root/'figures/refinement/before_after_same_new_data.png')))
display(Image(filename=str(root/'figures/refinement/external_confusion_counts.png')))
display(pd.read_parquet(out/'refined_2026_predictions.parquet'))''')
md('## 5. Audit conclusion and handoff\n\nNo model was retuned after seeing new-test results. Price MAE improves with more training history but still loses to announced cap. The classifier again fails to identify non-positive listings. Preserve the limitation rather than redefining the target or repeatedly changing the split.')
code('''display(Markdown((root/'docs/final_model_audit.md').read_text()))
print('Original artifacts preserved:',summary['original_artifacts_preserved'])
print('No external retuning:',summary['no_external_retuning'])
assert summary['external_rows']==17''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (local environment)','language':'python'},'language_info':{'name':'python'}})
nbf.write(nb,ROOT/'notebooks/04_model_audit_and_refinement.ipynb')
print('Wrote final audit, figures and notebook.')
