"""Audited second-pass tuning with a separate, unused 2026 chronological test."""
from pathlib import Path
import hashlib,json,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .data import load_dataset,feature_names
from .metrics import regression_metrics,classification_metrics
from .models import make_model,positive_probability
from .paths import project_root,read_json
from .refinement_models import COMPACT,CONTEXT,make_revision_model,specs


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')


def columns(spec):return COMPACT if spec['feature_set']=='compact' else CONTEXT


def chronological_folds(data):
    result=[]
    for year in [2024,2025]:
        for quarter in [1,2,3,4]:
            valid=data.loc[data.s0_cutoff_at.dt.year.eq(year)&data.s0_cutoff_at.dt.quarter.eq(quarter)]
            cutoff=valid.s0_cutoff_at.min()
            train=data.loc[data.s0_cutoff_at.lt(cutoff)&data.label_available_at.lt(cutoff)]
            assert set(train.index).isdisjoint(valid.index)
            result.append((f'{year}_Q{quarter}',cutoff,train.index.tolist(),valid.index.tolist()))
    return result


def tune():
    root=project_root();out=root/'outputs/refinement';out.mkdir(parents=True,exist_ok=True)
    data=load_dataset();folds=chronological_folds(data);candidates=specs()
    write(out/'candidate_registry.json',candidates)
    frame_rows=[];audits=[];warning_rows=[]
    with threadpool_limits(limits=1):
        for spec in candidates:
            for fid,cutoff,train_ids,valid_ids in folds:
                train=data.loc[train_ids];valid=data.loc[valid_ids];cols=columns(spec)
                x=train[cols].astype(float);xv=valid[cols].astype(float)
                target='listing_open_inr' if spec['task']=='regression' else 'target_positive_listing_gain'
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model=make_revision_model(spec).fit(x,train[target].astype(float if spec['task']=='regression' else int))
                for w in caught:
                    warning_rows.append({'candidate':spec['id'],'fold':fid,'category':w.category.__name__,'message':str(w.message)})
                predicted=model.predict(xv) if spec['task']=='regression' else positive_probability(model,xv)
                assert np.isfinite(predicted).all()
                for ipo_id,actual,pred in zip(valid_ids,valid[target],predicted):
                    frame_rows.append({'candidate_id':spec['id'],'task':spec['task'],'fold_id':fid,'ipo_id':ipo_id,
                        'actual':float(actual),'prediction':float(pred)})
                audits.append({'candidate_id':spec['id'],'fold_id':fid,'cutoff':cutoff,'training_rows':len(train),
                    'validation_rows':len(valid),'latest_training_label':train.label_available_at.max(),'external_rows_used':0})
            print('Validated',spec['id'],flush=True)
    predictions=pd.DataFrame(frame_rows)
    predictions.to_parquet(out/'development_predictions.parquet',index=False)
    predictions.to_csv(out/'development_predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'training_audit.csv',index=False)
    pd.DataFrame(warning_rows,columns=['candidate','fold','category','message']).to_csv(out/'warnings.csv',index=False)
    bad={w['candidate'] for w in warning_rows if w['category']=='ConvergenceWarning'}
    rows=[]
    for spec in candidates:
        source=predictions.loc[predictions.candidate_id.eq(spec['id'])]
        thresholds=[.5] if spec['task']=='regression' or spec['family']=='prior' else read_json('config/refinement_plan.json')['thresholds']
        for threshold in thresholds:
            m=regression_metrics(source.actual,source.prediction) if spec['task']=='regression' else classification_metrics(source.actual,source.prediction,threshold)
            rows.append({'candidate_id':spec['id'],'task':spec['task'],'family':spec['family'],'threshold':threshold,
                'features':len(columns(spec)),'converged':spec['id'] not in bad,**m})
    scores=pd.DataFrame(rows)
    regression=scores.loc[scores.task.eq('regression')&scores.converged].sort_values(['mae_inr','rmse_inr','features','candidate_id'])
    classification=scores.loc[scores.task.eq('classification')&scores.converged].sort_values(
        ['balanced_accuracy','accuracy','brier','features','candidate_id','threshold'],ascending=[False,False,True,True,True,True])
    regression.dropna(axis=1,how='all').to_csv(out/'regression_search.csv',index=False)
    classification.dropna(axis=1,how='all').to_csv(out/'classification_search.csv',index=False)
    byid={s['id']:s for s in candidates}
    selected={}
    for task,table in [('regression',regression),('classification',classification)]:
        winner=table.iloc[0]
        selected[task]={'spec':byid[winner.candidate_id],'threshold':float(winner.threshold),
            'development_metrics':{k:winner[k] for k in table.columns if k not in ['task','family','candidate_id','converged'] and pd.notna(winner[k])}}
    sources=['config/refinement_plan.json','ipo_prediction/refinement_models.py','ipo_prediction/refine.py',
             'data/refinement/external_2026.parquet','data/prepared/s0_model_ready.parquet','data/prepared/labels.parquet']
    frozen={'revision':'v2','selection':selected,'selected_without_external_scores':True,
        'sources_sha256':{s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},
        'validation_rows':int(predictions.ipo_id.nunique()),'development_rows':len(data),
        'note':'2025 is now development, not a new test. External 2026 results have not been used in this selection.'}
    target=root/'config/refined_selection.json'
    if target.exists():raise RuntimeError('Refinement choices already frozen; use recorded artifacts')
    write(target,frozen)
    print(json.dumps(selected,indent=2,default=str))


def evaluate():
    root=project_root();out=root/'outputs/refinement';model_dir=root/'models/refined';model_dir.mkdir(exist_ok=True)
    frozen=read_json('config/refined_selection.json')
    for name,expected in frozen['sources_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    if (out/'external_receipt.json').exists():
        receipt=read_json('outputs/refinement/external_receipt.json')
        for name,expected in receipt['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
        print('Reusing recorded refinement evaluation.');return read_json('outputs/refinement/summary.json')
    development=load_dataset()
    test=pd.read_parquet(root/'data/refinement/external_2026.parquet').set_index('ipo_id',drop=False)
    cutoff=test.s0_cutoff_at.min()
    train=development.loc[development.label_available_at.lt(cutoff)&development.s0_cutoff_at.lt(cutoff)]
    assert len(train)==253 and set(test.index).isdisjoint(train.index)
    metrics=[];predictions=[];saved=[]
    def record(name,task,pred,threshold=.5):
        actual=test.listing_open_inr if task=='regression' else test.target_positive_listing_gain
        m=regression_metrics(actual,pred) if task=='regression' else classification_metrics(actual,pred,threshold)
        metrics.append({'method':name,'task':task,'threshold':threshold,**m})
        for ipo_id,value in zip(test.index,pred):predictions.append({'method':name,'task':task,'ipo_id':ipo_id,'prediction':float(value)})
    with threadpool_limits(limits=1):
        for task in ['regression','classification']:
            choice=frozen['selection'][task];spec=choice['spec'];cols=columns(spec)
            target='listing_open_inr' if task=='regression' else 'target_positive_listing_gain'
            model=make_revision_model(spec).fit(train[cols].astype(float),train[target])
            pred=model.predict(test[cols].astype(float)) if task=='regression' else positive_probability(model,test[cols].astype(float))
            record('refined_v2',task,pred,choice['threshold'])
            path=model_dir/f'{task}.joblib'
            joblib.dump({'revision':'v2','model':model,'spec':spec,'input_features':cols,'threshold':choice['threshold'],
                         'training_ids':train.index.tolist(),'fit_cutoff':str(cutoff)},path,compress=3)
            loaded=joblib.load(path)['model']
            replay=loaded.predict(test[cols].astype(float)) if task=='regression' else positive_probability(loaded,test[cols].astype(float))
            np.testing.assert_allclose(pred,replay,rtol=1e-12,atol=1e-12)
            saved.append(str(path.relative_to(root)))
            # Exactly reproduce v1 feature interfaces. Missing external VIX/manager
            # fields are explicitly NaN, never copied from future or stale dates.
            oldbundle=joblib.load(root/f'models/final/{task}.joblib')
            oldcols=oldbundle['input_features']
            oldx=test.reindex(columns=oldcols).astype(float)
            pred=(oldbundle['model'].predict(oldx) if task=='regression' else positive_probability(oldbundle['model'],oldx))
            record('original_v1_149_train',task,pred,oldbundle['threshold'])
            oldmodel=make_model(oldbundle['spec']).fit(train[oldcols].astype(float),train[target])
            pred=oldmodel.predict(oldx) if task=='regression' else positive_probability(oldmodel,oldx)
            record('original_architecture_253_train',task,pred,oldbundle['threshold'])
    record('announced_cap_baseline','regression',test.band_high_inr.to_numpy())
    record('always_apply_baseline','classification',np.ones(len(test)))
    table=pd.DataFrame(metrics);table.to_csv(out/'external_comparison.csv',index=False)
    predicted=pd.DataFrame(predictions);predicted.to_parquet(out/'external_predictions.parquet',index=False)
    predicted.to_csv(out/'external_predictions.csv',index=False)
    view=test[['ipo_id','company_name','symbol','s0_cutoff_at','listing_date','band_low_inr','band_high_inr']].reset_index(drop=True)
    for task,column in [('regression','predicted_open_inr'),('classification','positive_gain_probability')]:
        p=predicted.loc[predicted.method.eq('refined_v2')&predicted.task.eq(task)].set_index('ipo_id').prediction
        view[column]=view.ipo_id.map(p)
    threshold=frozen['selection']['classification']['threshold']
    view['predicted_class']=(view.positive_gain_probability>=threshold).astype(int)
    view['signal']=np.where(view.predicted_class.eq(1),'Apply','Skip')
    view.to_parquet(out/'refined_2026_predictions.parquet',index=False)
    view.to_csv(out/'refined_2026_predictions.csv',index=False)
    summary={'revision':'v2','development_rows':len(train),'validation_rows':frozen['validation_rows'],
        'external_rows':len(test),'external_first_s0':str(cutoff),'external_last_s0':str(test.s0_cutoff_at.max()),
        'external_positive':int(test.target_positive_listing_gain.sum()),'threshold':threshold,'metrics':metrics,
        'original_artifacts_preserved':True,'no_external_retuning':True,
        'comparability_note':'v1 classifiers require VIX/manager predictors unavailable in new sources, imputed using fitted training statistics; v2 uses only shared, observed feature definitions.',
        'sample_note':'Small early-2026 source-covered sample; not a full-year or publication-grade external benchmark.'}
    write(out/'summary.json',summary)
    paths=[out/'external_comparison.csv',out/'external_predictions.parquet',out/'external_predictions.csv',
           out/'refined_2026_predictions.parquet',out/'refined_2026_predictions.csv',out/'summary.json',
           root/'config/refined_selection.json',*[root/p for p in saved]]
    write(out/'external_receipt.json',{'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    print(table.dropna(axis=1,how='all').to_string(index=False))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tune','evaluate'])
    args=parser.parse_args()
    tune() if args.action=='tune' else evaluate()
