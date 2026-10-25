import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from starlette.testclient import TestClient
from app.server import app
from ipo_prediction.data import load_dataset, get_xy
from ipo_prediction.evaluate import run
from ipo_prediction.metrics import regression_metrics, classification_metrics
from ipo_prediction.models import positive_probability
from ipo_prediction.paths import project_root, read_json


def test_final_models_train_on_exact_development_ids_and_replay():
    root = project_root()
    data = load_dataset()
    splits = pd.read_csv(root / "data/splits/final_membership.csv")
    train = set(splits.loc[splits.role.eq("final_train"), "ipo_id"])
    pred = pd.read_parquet(root / "outputs/part3/heldout_predictions.parquet")
    assert set(pred.ipo_id) == set(splits.loc[splits.role.eq("reserved_test"), "ipo_id"])
    frozen = read_json("config/selected_models.json")
    for task, column in [("regression", "predicted_open_inr"), ("classification", "positive_gain_probability")]:
        bundle = joblib.load(root / f"models/final/{task}.joblib")
        assert set(bundle["training_ids"]) == train
        assert not train & set(pred.ipo_id)
        assert bundle["spec"] == frozen["selection"][task]
        assert data.loc[bundle["training_ids"], "label_available_at"].lt(pd.Timestamp(bundle["fit_cutoff"])).all()
        x, _ = get_xy(data, pred.ipo_id, bundle["spec"]["feature_set"])
        result = bundle["model"].predict(x) if task == "regression" else positive_probability(bundle["model"], x)
        np.testing.assert_allclose(result, pred[column], atol=1e-12, rtol=1e-12)


def test_final_metrics_recompute_and_outcomes_are_separate():
    root = project_root()
    pred = pd.read_parquet(root / "outputs/part3/heldout_predictions.parquet")
    actual = pd.read_parquet(root / "outputs/part3/evaluation_outcomes.parquet")
    assert not {"listing_open_inr", "final_issue_price_inr", "target_positive_listing_gain"} & set(pred.columns)
    assert list(pred.ipo_id) == list(actual.ipo_id)
    summary = read_json("outputs/part3/completion_summary.json")
    assert regression_metrics(actual.listing_open_inr, pred.predicted_open_inr) == {k:v for k,v in summary["regression"].items() if k not in {"task", "model"}}
    assert classification_metrics(actual.target_positive_listing_gain, pred.positive_gain_probability, .5) == {k:v for k,v in summary["classification"].items() if k not in {"task", "model"}}


def test_evaluation_rerun_reuses_saved_artifacts():
    root = project_root()
    receipt = root / "outputs/part3/evaluation_receipt.json"
    before = receipt.read_bytes()
    stamps = [(p, p.stat().st_mtime_ns) for p in (root / "models/final").glob("*.joblib")]
    run()
    assert receipt.read_bytes() == before
    assert all(p.stat().st_mtime_ns == stamp for p,stamp in stamps)


def test_app_displays_saved_predictions_and_actual_outcomes():
    root = project_root()
    client = TestClient(app)
    pred = pd.read_parquet(root / "outputs/part3/heldout_predictions.parquet").set_index("ipo_id")
    for ipo_id in [pred.index[0], pred.index[-1]]:
        response = client.post('/api/predict',json={'name':ipo_id})
        assert response.status_code==200
        result=response.json()
        row = pred.loc[ipo_id]
        assert result['prediction']['opening_price']==row.predicted_open_inr
        assert result['prediction']['positive_probability']==row.positive_gain_probability
        assert result['prediction']['signal']==row.signal
        assert result['source']['kind']=='reserved'
    response=client.get('/api/outcome/'+pred.index[-1])
    assert response.status_code==200 and response.json()['opening_price']>0


def test_recorded_evaluation_hashes():
    root = project_root()
    receipt = read_json("outputs/part3/evaluation_receipt.json")
    for group in ["input_sha256", "artifact_sha256"]:
        for name, expected in receipt[group].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
