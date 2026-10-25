import pandas as pd
from starlette.testclient import TestClient
from app.server import app
from ipo_prediction.paths import project_root

client=TestClient(app)


def test_new_frontend_and_catalog():
    page=client.get('/')
    assert page.status_code==200
    assert 'prediction-form' in page.text and 'See historical' in page.text
    assert client.get('/assets/styles.css').status_code==200
    catalog=client.get('/api/ipos').json()
    extra=0
    if (project_root()/'data/refinement/external_2026.parquet').exists():
        extra=len(pd.read_parquet(project_root()/'data/refinement/external_2026.parquet'))
    assert len(catalog['ipos'])==253+extra and catalog['supported']==252+extra


def test_exact_short_symbol_and_ambiguous_queries():
    for name in ['Hyundai Motor India','HYUNDAI','hyundai motor india limited','NSDL']:
        response=client.post('/api/predict',json={'name':name})
        assert response.status_code==200,name
        assert response.json()['prediction']['opening_price']>0
    ambiguous=client.post('/api/predict',json={'name':'India'})
    assert ambiguous.status_code==409 and len(ambiguous.json()['suggestions'])>1


def test_invalid_unknown_and_excluded_names():
    for body in [{},{'name':''},{'name':42},[],{'name':'IPO'}]:
        assert client.post('/api/predict',json=body).status_code==400
    assert client.post('/api/predict',json={'name':'Not A Real Issuer QZXV 999'}).status_code==404
    assert client.post('/api/predict',json={'name':'Indo Farm'}).status_code==422


def test_development_predictions_match_saved_validation():
    source=pd.read_csv(project_root()/'outputs/part2/selected_validation_predictions.csv')
    ipo_id=source.ipo_id.iloc[0]
    response=client.post('/api/predict',json={'name':ipo_id}).json()
    assert response['source']['kind']=='validation'
    saved=source.loc[source.ipo_id.eq(ipo_id)]
    assert response['prediction']['opening_price']==float(saved.loc[saved.task.eq('regression'),'prediction'].iloc[0])
    assert response['prediction']['positive_probability']==float(saved.loc[saved.task.eq('classification'),'prediction'].iloc[0])


def test_training_illustration_explicitly_labeled():
    result=client.post('/api/predict',json={'name':'Radiant Cash'}).json()
    assert result['source']['kind']=='training'
    assert 'not an independent' in result['source']['description']
    assert result['prediction']['opening_price']>0
    assert not {'actual','final_issue_price_inr','listing_open_inr'} & set(result['prediction'])
