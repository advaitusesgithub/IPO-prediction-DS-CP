"""Validate prepared inputs and expose explicitly separated X and y."""
import numpy as np
import pandas as pd
from .paths import project_root, read_json


def feature_names(feature_set="core"):
    config = read_json("config/features.json")
    if feature_set == "core":
        return config["core_predictors"]
    if feature_set == "extended":
        return config["core_predictors"] + config["optional_financial_predictors"]
    raise ValueError("feature_set must be core or extended")


def load_dataset():
    """One row per IPO, with outcomes attached exclusively for targets/auditing."""
    base = project_root() / "data/prepared"
    matrix = pd.read_parquet(base / "s0_model_ready.parquet")
    labels = pd.read_parquet(base / "labels.parquet")
    master = pd.read_parquet(base / "ipo_master.parquet")
    schema = read_json("config/source_schema.json")
    expected = set(schema["predictor_columns"]) | set(schema["must_not_use_as_predictors"])
    if set(matrix.columns) != expected:
        raise ValueError("Prepared matrix schema changed")
    for name, frame in [("matrix", matrix), ("labels", labels), ("master", master)]:
        if frame.ipo_id.isna().any() or frame.ipo_id.duplicated().any():
            raise ValueError(f"Invalid or duplicate IPO identifiers in {name}")
        if set(frame.ipo_id) != set(matrix.ipo_id):
            raise ValueError(f"IPO coverage mismatch in {name}")
    if len(matrix) != 253 or not matrix.s0_eligible.eq(True).all():
        raise ValueError("Expected the fixed 253-row eligible cohort")
    outcome = labels.set_index("ipo_id").loc[matrix.ipo_id].reset_index()
    for col in ["target_listing_gain_open", "target_positive_listing_gain"]:
        if not np.allclose(matrix[col].to_numpy(dtype=float), outcome[col].to_numpy(dtype=float)):
            raise ValueError(f"Conflicting label: {col}")
    for col in ["label_available_at", "label_verified"]:
        if not matrix[col].equals(outcome[col]):
            raise ValueError(f"Conflicting label metadata: {col}")
    for col in ["listing_open_inr", "final_issue_price_inr"]:
        if not np.isfinite(outcome[col]).all() or not outcome[col].gt(0).all():
            raise ValueError(f"Invalid target price: {col}")
    if not np.allclose(outcome.listing_open_inr / outcome.final_issue_price_inr - 1,
                       matrix.target_listing_gain_open, atol=1e-10):
        raise ValueError("Gain label formula mismatch")
    if not np.array_equal((outcome.listing_open_inr > outcome.final_issue_price_inr).astype(int),
                          matrix.target_positive_listing_gain.astype(int)):
        raise ValueError("Classification label formula mismatch")
    if not matrix.label_available_at.gt(matrix.s0_cutoff_at).all():
        raise ValueError("Outcome maturity must follow S0")
    if str(matrix.s0_cutoff_at.dt.tz) != "Asia/Kolkata":
        raise ValueError("S0 timezone must be Asia/Kolkata")
    if not (matrix.s0_cutoff_at.dt.hour.eq(9) & matrix.s0_cutoff_at.dt.minute.eq(0)).all():
        raise ValueError("Unexpected S0 time")
    numeric = matrix[schema["predictor_columns"]].to_numpy(dtype=float)
    if np.isinf(numeric).any():
        raise ValueError("Infinite predictor value")
    if not matrix.band_high_inr.gt(0).all():
        raise ValueError("Announced price cap must be positive")
    result = matrix.merge(outcome[["ipo_id", "listing_open_inr", "final_issue_price_inr",
                                  "label_verification_tier"]], on="ipo_id", validate="one_to_one")
    result = result.merge(master[["ipo_id", "company_name", "listing_date"]],
                          on="ipo_id", validate="one_to_one")
    return result.sort_values(["s0_cutoff_at", "ipo_id"]).set_index("ipo_id", drop=False)


def get_xy(dataset, ids, feature_set="core"):
    """IDs must be supplied explicitly; targets are never inferred as features."""
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate requested IDs")
    selected = dataset.loc[ids]
    x = selected[feature_names(feature_set)].astype(float).copy()
    y = selected[["listing_open_inr", "target_positive_listing_gain"]].copy()
    y["target_positive_listing_gain"] = y["target_positive_listing_gain"].astype(int)
    return x, y
