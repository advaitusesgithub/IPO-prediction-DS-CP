"""Unfitted preprocessing factories for later fold-specific model training."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from .data import feature_names


class S0FeatureGuard(TransformerMixin, BaseEstimator):
    """Fail closed on extra columns, changed order, nonnumeric or infinite inputs."""
    def __init__(self, feature_set="core"):
        self.feature_set = feature_set

    def _validate(self, x):
        expected = feature_names(self.feature_set)
        if not isinstance(x, pd.DataFrame) or list(x.columns) != expected:
            raise ValueError("Input must contain only the ordered S0 feature whitelist")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in x.dtypes):
            raise ValueError("S0 inputs must be numeric")
        if np.isinf(x.to_numpy(dtype=float)).any():
            raise ValueError("Infinite S0 input")
        return x.astype(float)

    def fit(self, x, y=None):
        self._validate(x)
        self.feature_names_in_ = np.array(x.columns, dtype=object)
        self.n_features_in_ = len(x.columns)
        return self

    def transform(self, x):
        check_is_fitted(self)
        return self._validate(x)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return self.feature_names_in_.copy()


def make_preprocessor(feature_set="core", scale=False):
    """Return an UNFITTED pipeline; caller fits it on one training fold only.

    Missing-indicator columns are learned from training missingness. Entirely
    missing training columns remain present and are filled with zero, with an
    indicator. The same fitted transformation is reused on validation/test.
    """
    steps = [("guard", S0FeatureGuard(feature_set)),
             ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                       keep_empty_features=True))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")
