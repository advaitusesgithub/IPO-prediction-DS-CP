"""Checks for leakage, frozen membership and portable data loading."""
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from ipo_prediction.data import load_dataset, feature_names, get_xy
from ipo_prediction.paths import project_root
from ipo_prediction.preprocessing import make_preprocessor
from ipo_prediction.splits import make_splits, fold_ids


@pytest.fixture(scope="module")
def data():
    return load_dataset()


def synthetic():
    frame = pd.DataFrame(1.0, index=range(3), columns=feature_names())
    frame["nifty_return_21d"] = [1.0, np.nan, 3.0]
    return frame


def test_frozen_splits_and_label_maturity(data):
    final, walk = make_splits(data)
    assert final.role.value_counts().to_dict() == {"final_train": 149, "reserved_test": 103, "purged_label_unavailable": 1}
    validation = []
    for fid, rows in walk.groupby("fold_id"):
        train = fold_ids(walk, fid, "train")
        valid = fold_ids(walk, fid, "validation")
        assert not set(train) & set(valid)
        cutoff = data.loc[valid, "s0_cutoff_at"].min()
        assert data.loc[train, "label_available_at"].lt(cutoff).all()
        assert data.loc[train, "s0_cutoff_at"].lt(cutoff).all()
        validation.extend(valid)
    assert len(validation) == len(set(validation)) == 90
    assert not set(validation) & set(final.loc[final.role.eq("reserved_test"), "ipo_id"])


def test_xy_has_only_announced_features(data):
    ids = data.index[:5]
    x, y = get_xy(data, ids)
    assert list(x.columns) == feature_names()
    assert list(y.columns) == ["listing_open_inr", "target_positive_listing_gain"]
    assert x.index.equals(y.index)
    assert "final_issue_price_inr" not in x


@pytest.mark.parametrize("bad_column", ["final_issue_price_inr", "listing_open_inr", "target_positive_listing_gain", "subscription_total"])
def test_outcome_and_subscription_columns_rejected(bad_column):
    x = synthetic()
    x[bad_column] = 1
    with pytest.raises(ValueError, match="whitelist"):
        make_preprocessor().fit(x)


def test_training_only_imputation():
    train = synthetic()
    validation = train.iloc[:1].copy()
    validation["nifty_return_21d"] = np.nan
    pipeline = make_preprocessor()
    pipeline.fit(train)
    assert pipeline.transform(validation).iloc[0]["nifty_return_21d"] == 2.0
    validation["nifty_return_21d"] = 1e9
    pipeline.transform(validation)
    assert pipeline.named_steps["imputer"].statistics_[4] == 2.0


def test_scaler_fits_training_only():
    pipeline = make_preprocessor(scale=True).fit(synthetic())
    mean = pipeline.named_steps["scaler"].mean_.copy()
    pipeline.transform(synthetic() * 1e6)
    np.testing.assert_array_equal(mean, pipeline.named_steps["scaler"].mean_)


def test_all_missing_training_feature_retained():
    train = synthetic()
    train["manager_prior_median_gain"] = np.nan
    pipeline = make_preprocessor().fit(train)
    result = pipeline.transform(train)
    assert "manager_prior_median_gain" in result.columns
    assert result.manager_prior_median_gain.eq(0).all()
    assert result.missingindicator_manager_prior_median_gain.eq(1).all()
    assert np.isfinite(result.to_numpy()).all()


def test_factory_is_unfitted():
    with pytest.raises(NotFittedError):
        make_preprocessor().transform(synthetic())


def test_infinite_and_changed_order_rejected():
    pipeline = make_preprocessor().fit(synthetic())
    with pytest.raises(ValueError):
        pipeline.transform(synthetic().iloc[:, ::-1])
    x = synthetic()
    x.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        pipeline.transform(x)


def test_portable_under_different_directory_name(tmp_path):
    root = project_root()
    renamed = tmp_path / "renamed_workspace"
    for name in ["ipo_prediction", "config", "data"]:
        shutil.copytree(root / name, renamed / name, ignore=shutil.ignore_patterns("__pycache__"))
    # CWD and package path now point only at the relocated bundle.
    code = "from ipo_prediction.data import load_dataset; from ipo_prediction.splits import make_splits; d=load_dataset(); f,w=make_splits(d); assert len(d)==253; print(len(f),len(w))"
    completed = subprocess.run([sys.executable, "-c", code], cwd=renamed, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "253 446"
