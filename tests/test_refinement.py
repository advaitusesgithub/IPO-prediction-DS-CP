import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ipo_prediction.data import load_dataset
from ipo_prediction.paths import project_root,read_json
from ipo_prediction.refine import chronological_folds,columns
from ipo_prediction.refinement_models import RelativePriceRegressor,RevisionGuard,COMPACT
from ipo_prediction.models import positive_probability
from ipo_prediction.inference import lookup


def test_refinement_folds_are_label_mature_and_time_ordered():
    data=load_dataset();all_valid=[]
    for name,cutoff,train,valid in chronological_folds(data):
        assert not set(train)&set(valid)
        assert data.loc[train,'label_available_at'].lt(cutoff).all()
        assert data.loc[train,'s0_cutoff_at'].lt(cutoff).all()
        all_valid.extend(valid)
    assert len(all_valid)==len(set(all_valid))==194


def test_cap_relative_weighted_baseline_uses_training_only():
    x=pd.DataFrame({'band_high_inr':[100.,500.,1000.]})
    model=RelativePriceRegressor(baseline='weighted_median').fit(x,[100.,550.,1200.])
    assert model.premium_==pytest.approx(.2)
    np.testing.assert_allclose(model.predict(pd.DataFrame({'band_high_inr':[200.]})),[240.])


def test_refinement_feature_guard_rejects_outcomes():
    x=pd.DataFrame(1.,index=range(3),columns=COMPACT)
    x['final_issue_price_inr']=10
    with pytest.raises(ValueError,match='whitelist'):RevisionGuard(tuple(COMPACT)).fit(x)


def test_external_inputs_are_unused_fresh_and_point_in_time():
    root=project_root();external=pd.read_parquet(root/'data/refinement/external_2026.parquet')
    old=load_dataset()
    assert len(external)==17
    assert set(external.ipo_id).isdisjoint(old.index)
    assert external.market_source_age_days.between(1,7).all()
    assert external.latest_history_label.lt(external.s0_cutoff_at).all()
    assert old.label_available_at.lt(external.s0_cutoff_at.min()).all()


def test_refinement_model_reload_reproduces_new_predictions():
    root=project_root();data=pd.read_parquet(root/'data/refinement/external_2026.parquet')
    recorded=pd.read_parquet(root/'outputs/refinement/refined_2026_predictions.parquet')
    for task,column in [('regression','predicted_open_inr'),('classification','positive_gain_probability')]:
        bundle=joblib.load(root/f'models/refined/{task}.joblib')
        assert len(bundle['training_ids'])==253 and set(bundle['training_ids']).isdisjoint(data.ipo_id)
        x=data[bundle['input_features']].astype(float)
        pred=bundle['model'].predict(x) if task=='regression' else positive_probability(bundle['model'],x)
        np.testing.assert_allclose(pred,recorded[column],rtol=1e-12,atol=1e-12)


def test_refinement_receipt_and_original_evidence_unchanged():
    root=project_root()
    for path,group in [('outputs/refinement/external_receipt.json','sha256'),('outputs/part3/evaluation_receipt.json','artifact_sha256')]:
        receipt=read_json(path)
        for name,expected in receipt[group].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected


def test_external_ui_uses_recorded_prediction_and_its_threshold():
    root=project_root();recorded=pd.read_parquet(root/'outputs/refinement/refined_2026_predictions.parquet')
    row=recorded.iloc[0]
    data,code=lookup(row.symbol)
    assert code==200 and data['source']['kind']=='external'
    assert data['prediction']['threshold']==.45
    assert data['prediction']['opening_price']==row.predicted_open_inr
    assert data['prediction']['positive_probability']==row.positive_gain_probability
    assert data['prediction']['signal']==row.signal
