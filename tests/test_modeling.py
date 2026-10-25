"""Independent metric calculations, leakage controls and saved-model replay."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from ipo_prediction.data import feature_names, get_xy
from ipo_prediction.metrics import classification_metrics, regression_metrics
from ipo_prediction.models import AnnouncedCapRegressor, candidate_specs, make_model, positive_probability
from ipo_prediction.paths import project_root, read_json
from ipo_prediction.train import development_inputs, evaluate_candidates, rank_candidates


def test_metrics_against_hand_calculations():
    prices = regression_metrics([100, 200], [110, 180])
    assert prices["mae_inr"] == 15
    assert prices["rmse_inr"] == pytest.approx(np.sqrt(250))
    assert prices["mape_pct"] == 10
    classification = classification_metrics([0, 0, 1, 1], [.1, .8, .4, .9])
    assert classification["accuracy"] == .5
    assert classification["balanced_accuracy"] == .5
    assert (classification["tn"], classification["fp"], classification["fn"], classification["tp"]) == (1, 1, 1, 1)
    assert classification["brier"] == pytest.approx((.01 + .64 + .36 + .01) / 4)


def test_cap_baseline_uses_training_ratios_not_final_issue_price():
    x = pd.DataFrame({"band_high_inr": [100, 200, 400]})
    model = AnnouncedCapRegressor(use_training_median=True).fit(x, [120, 200, 800])
    assert model.multiplier_ == 1.2
    prediction = model.predict(pd.DataFrame({"band_high_inr": [300]}))
    np.testing.assert_allclose(prediction, [360])


def test_prespecified_candidates_and_threshold():
    plan = read_json("config/model_search.json")
    specs = candidate_specs(plan)
    assert len(specs) == len({s["candidate_id"] for s in specs}) == 19
    assert sum(s["task"] == "regression" for s in specs) == 10
    assert sum(s["task"] == "classification" for s in specs) == 9
    assert plan["classification_threshold"] == .5


def test_training_refuses_non_development_dates():
    dev, walk = development_inputs()
    poisoned = dev.copy()
    poisoned.loc[poisoned.index[0], "s0_cutoff_at"] = pd.Timestamp("2025-01-07 09:00:00", tz="Asia/Kolkata")
    with pytest.raises(ValueError, match="only the 149 development"):
        evaluate_candidates(poisoned, walk, [], .5)


def test_selection_does_not_depend_on_candidate_order():
    root = project_root()
    if not (root / "outputs/part2/validation_predictions.parquet").exists():
        pytest.skip("Run Part 2 to create comparison artifacts")
    predictions = pd.read_parquet(root / "outputs/part2/validation_predictions.parquet")
    specs = candidate_specs(read_json("config/model_search.json"))
    a = rank_candidates(predictions, specs, .5)
    b = rank_candidates(predictions.sample(frac=1, random_state=7), list(reversed(specs)), .5)
    for left, right in zip(a, b):
        assert list(left.candidate_id) == list(right.candidate_id)


def test_saved_models_replay_and_never_train_on_reserved_rows():
    root = project_root()
    path = root / "outputs/part2/model_artifacts.csv"
    if not path.exists():
        pytest.skip("Run Part 2 to create fitted fold models")
    artifacts = pd.read_csv(path)
    predictions = pd.read_parquet(root / "outputs/part2/validation_predictions.parquet")
    dev, walk = development_inputs()
    for row in artifacts.itertuples():
        bundle = joblib.load(root / row.path)
        assert bundle["role"] == "development_validation_only"
        assert set(bundle["training_ids"]) <= set(dev.index)
        assert not set(bundle["training_ids"]) & set(bundle["validation_ids"])
        assert len(bundle["training_ids"]) <= 121
        x, _ = get_xy(dev, bundle["validation_ids"], bundle["spec"]["feature_set"])
        model = bundle["model"]
        actual = (model.predict(x) if bundle["spec"]["task"] == "regression" else positive_probability(model, x))
        expected = predictions.loc[predictions.candidate_id.eq(row.candidate_id) & predictions.fold_id.eq(row.fold_id)].prediction
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_validation_predictions_exclude_reserved_population():
    root = project_root()
    path = root / "outputs/part2/validation_predictions.parquet"
    if not path.exists():
        pytest.skip("Run Part 2 to create predictions")
    predictions = pd.read_parquet(path)
    final = pd.read_csv(root / "data/splits/final_membership.csv")
    reserved = set(final.loc[final.role.eq("reserved_test"), "ipo_id"])
    assert not set(predictions.ipo_id) & reserved
    assert predictions.groupby("candidate_id").ipo_id.nunique().eq(90).all()
    assert len(predictions) == 19 * 90


def test_saved_score_tables_recompute_from_predictions():
    root = project_root()
    path = root / "outputs/part2/validation_predictions.parquet"
    if not path.exists():
        pytest.skip("Run Part 2 to create predictions")
    predictions = pd.read_parquet(path)
    regression, classification = rank_candidates(predictions, candidate_specs(read_json("config/model_search.json")), .5)
    for task, recomputed in [("regression", regression), ("classification", classification)]:
        saved = pd.read_csv(root / f"outputs/part2/{task}_comparison.csv")
        pd.testing.assert_frame_equal(recomputed, saved, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_selected_specs_are_the_best_ranked_and_frozen():
    root = project_root()
    path = root / "config/selected_models.json"
    if not path.exists():
        pytest.skip("Run Part 2 to freeze selection")
    selected = json.loads(path.read_text())
    for task in ["regression", "classification"]:
        scores = pd.read_csv(root / f"outputs/part2/{task}_comparison.csv")
        assert scores.iloc[0].candidate_id == selected["selection"][task]["candidate_id"]
        assert selected["selection"][task]["feature_set"] in ["core", "extended"]
    assert selected["classification_threshold"] == .5
