"""Serializable estimators and fixed candidate factories for S0 modeling."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from .preprocessing import S0FeatureGuard, make_preprocessor


class AnnouncedCapRegressor(RegressorMixin, BaseEstimator):
    """Predict announced cap, optionally times a training-only median ratio."""
    def __init__(self, use_training_median=False):
        self.use_training_median = use_training_median

    def fit(self, x, y):
        caps = x["band_high_inr"].to_numpy(dtype=float)
        prices = np.asarray(y, dtype=float)
        if not np.isfinite(caps).all() or not (caps > 0).all():
            raise ValueError("Announced caps must be finite and positive")
        if not np.isfinite(prices).all() or not (prices > 0).all():
            raise ValueError("Training prices must be finite and positive")
        self.multiplier_ = float(np.median(prices / caps)) if self.use_training_median else 1.0
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        check_is_fitted(self)
        caps = x["band_high_inr"].to_numpy(dtype=float)
        if not np.isfinite(caps).all() or not (caps > 0).all():
            raise ValueError("Announced caps must be finite and positive")
        return caps * self.multiplier_


class PositivePriceRegressor(RegressorMixin, BaseEstimator):
    """Prespecified physical price floor, retained in saved estimator behavior."""
    def __init__(self, estimator, minimum_price=0.01):
        self.estimator = estimator
        self.minimum_price = minimum_price

    def fit(self, x, y):
        self.estimator_ = clone(self.estimator).fit(x, y)
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x):
        check_is_fitted(self)
        return np.maximum(self.minimum_price, self.estimator_.predict(x))


def candidate_specs(plan):
    result = []
    def add(task, family, features, params, suffix, baseline=False):
        result.append({"candidate_id": f"{task}_{family}_{features}_{suffix}", "task": task,
                       "family": family, "feature_set": features, "params": params,
                       "is_baseline": baseline, "random_state": plan["random_state"]})
    for name in plan["regression"]["baselines"]:
        add("regression", name, "core", {}, "baseline", True)
    add("classification", "always_apply", "core", {}, "baseline", True)
    for features in plan["feature_sets"]:
        for alpha in plan["regression"]["ridge_alphas"]:
            add("regression", "ridge", features, {"alpha": alpha}, f"alpha{alpha:g}")
        for i, params in enumerate(plan["regression"]["random_forest_settings"], 1):
            add("regression", "random_forest", features, params, f"setting{i}")
        for i, params in enumerate(plan["classification"]["logistic_settings"], 1):
            add("classification", "logistic", features, params, f"setting{i}")
        for i, params in enumerate(plan["classification"]["random_forest_settings"], 1):
            add("classification", "random_forest", features, params, f"setting{i}")
    return result


def make_model(spec):
    """Fresh unfitted model; train only on the fold-specific X and y."""
    features, family, task = spec["feature_set"], spec["family"], spec["task"]
    params, seed = spec["params"], spec["random_state"]
    if family in {"announced_cap", "training_median_premium"}:
        return Pipeline([("guard", S0FeatureGuard(features)),
                         ("model", AnnouncedCapRegressor(family == "training_median_premium"))])
    if family == "always_apply":
        return Pipeline([("guard", S0FeatureGuard(features)),
                         ("model", DummyClassifier(strategy="constant", constant=1))])
    scale = family in {"ridge", "logistic"}
    if family == "ridge":
        estimator = PositivePriceRegressor(Ridge(**params))
    elif family == "logistic":
        estimator = LogisticRegression(**params, max_iter=5000, solver="lbfgs", random_state=seed)
    elif family == "random_forest" and task == "regression":
        estimator = PositivePriceRegressor(RandomForestRegressor(**params, random_state=seed, n_jobs=1))
    elif family == "random_forest" and task == "classification":
        estimator = RandomForestClassifier(**params, random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown model spec: {spec}")
    return Pipeline([("preprocessing", make_preprocessor(features, scale=scale)), ("model", estimator)])


def positive_probability(model, x):
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("Fitted classifier must include the positive class")
    probabilities = model.predict_proba(x)[:, matches[0]]
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid positive-class probabilities")
    return probabilities
