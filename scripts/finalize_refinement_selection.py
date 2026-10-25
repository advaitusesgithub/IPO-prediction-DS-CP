"""Apply practical validation-only decision guardrails before new-test scoring."""
from pathlib import Path
import hashlib,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ipo_prediction.refine import write

out=ROOT/'outputs/refinement'
assert not (out/'external_receipt.json').exists(), 'New test has already been evaluated'
path=ROOT/'config/refined_selection.json'
frozen=json.loads(path.read_text())
write(out/'unconstrained_selection.json',frozen)
scores=pd.read_csv(out/'classification_search.csv')
pred=pd.read_parquet(out/'development_predictions.parquet')
labels=pred.loc[pred.task.eq('classification')].drop_duplicates('ipo_id').actual
majority=max(labels.mean(),1-labels.mean())
eligible=scores.loc[scores.accuracy.ge(majority-.05)&scores.recall_apply.ge(.75)&scores.specificity_skip.ge(.10)]
assert len(eligible), 'No candidate meets practical guards; keep original classifier'
best=eligible.iloc[0]
registry=json.loads((out/'candidate_registry.json').read_text())
spec=next(s for s in registry if s['id']==best.candidate_id)
frozen['selection']['classification']={'spec':spec,'threshold':float(best.threshold),
    'development_metrics':{k:best[k] for k in ['accuracy','balanced_accuracy','precision_apply','recall_apply','specificity_skip','f1_apply','tn','fp','fn','tp']}}
plan_path=ROOT/'config/refinement_plan.json';plan=json.loads(plan_path.read_text())
plan['classification_guardrails']={'minimum_accuracy':'development majority accuracy minus 0.05','minimum_apply_recall':.75,
    'minimum_skip_specificity':.1,'reason':'Avoid sacrificing most winning IPOs just to improve balanced accuracy; guardrails applied to development outputs only before new-test scoring'}
write(plan_path,plan)
frozen['validation_guardrails']=plan['classification_guardrails']
frozen['sources_sha256']['config/refinement_plan.json']=hashlib.sha256(plan_path.read_bytes()).hexdigest()
write(path,frozen)
eligible.to_csv(out/'guarded_classification_search.csv',index=False)
print(json.dumps(frozen['selection'],indent=2,default=str))
