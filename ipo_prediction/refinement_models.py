"""Second-pass estimators: cap-relative price targets and non-degenerate decisions."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted

COMPACT = ['band_high_inr','band_width_fraction','lot_application_at_cap_inr',
           'log1p_planned_offer_upper_cr','nifty_return_21d','nifty_volatility_21d',
           'recent_listed_ipo_count_90d','recent_median_listing_gain_90d']
CONTEXT = COMPACT + ['nifty_return_5d','nifty_return_63d',
                     'recent_listed_ipo_count_30d','recent_median_listing_gain_30d']


class RevisionGuard(TransformerMixin, BaseEstimator):
    def __init__(self, columns): self.columns = columns
    def fit(self, x, y=None):
        self.transform(x)
        self.feature_names_in_ = np.asarray(self.columns, dtype=object)
        return self
    def transform(self, x):
        if list(x.columns) != list(self.columns):
            raise ValueError('Use only the ordered refinement predictor whitelist')
        values = x.to_numpy(dtype=float)
        if np.isinf(values).any(): raise ValueError('Infinite predictor')
        return values


class RelativePriceRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimator=None, baseline='learned'):
        self.estimator = estimator
        self.baseline = baseline
    def fit(self, x, y):
        cap = x.band_high_inr.to_numpy(dtype=float)
        if not np.isfinite(cap).all() or not (cap > 0).all(): raise ValueError('Invalid cap')
        ratio = np.asarray(y, dtype=float) / cap - 1
        if self.baseline == 'cap': self.premium_ = 0.
        elif self.baseline == 'median': self.premium_ = float(np.median(ratio))
        elif self.baseline == 'weighted_median':
            order = np.argsort(ratio)
            self.premium_ = float(ratio[order][np.searchsorted(np.cumsum(cap[order]),cap.sum()/2)])
        else:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(x, ratio, model__sample_weight=cap/cap.mean())
        self.n_features_in_ = x.shape[1]
        return self
    def predict(self, x):
        check_is_fitted(self)
        premium = self.estimator_.predict(x) if self.baseline == 'learned' else self.premium_
        return np.maximum(.01,x.band_high_inr.to_numpy(dtype=float)*(1+premium))


def make_revision_model(spec):
    task, family, params = spec['task'],spec['family'],spec['params']
    columns = COMPACT if spec['feature_set']=='compact' else CONTEXT
    pre = [('guard', RevisionGuard(tuple(columns))),
           ('imputer',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True))]
    if family in ['ridge','huber','logistic']: pre.append(('scaler',StandardScaler()))
    if task=='regression' and family in ['cap','median','weighted_median']:
        return RelativePriceRegressor(baseline=family)
    if family=='ridge': model=Ridge(**params)
    elif family=='huber': model=HuberRegressor(max_iter=3000,**params)
    elif family=='logistic': model=LogisticRegression(max_iter=5000,random_state=42,**params)
    elif family=='prior': model=DummyClassifier(strategy='prior')
    elif family=='forest':
        cls=RandomForestRegressor if task=='regression' else RandomForestClassifier
        model=cls(n_estimators=200,random_state=42,n_jobs=1,**params)
    elif family=='extra':
        cls=ExtraTreesRegressor if task=='regression' else ExtraTreesClassifier
        model=cls(n_estimators=200,random_state=42,n_jobs=1,**params)
    elif family=='hist':
        cls=HistGradientBoostingRegressor if task=='regression' else HistGradientBoostingClassifier
        model=cls(max_iter=120,learning_rate=.05,early_stopping=False,random_state=42,**params)
    else: raise ValueError(family)
    pipeline=Pipeline(pre+[('model',model)])
    return RelativePriceRegressor(pipeline) if task=='regression' else pipeline


def specs():
    items=[]
    def add(task,family,feature_set,params):
        count=sum(s['task']==task and s['family']==family and s['feature_set']==feature_set for s in items)+1
        items.append({'id':f'{task}_{family}_{feature_set}_{count}','task':task,'family':family,
                      'feature_set':feature_set,'params':params})
    for family in ['cap','median','weighted_median']: add('regression',family,'compact',{})
    add('classification','prior','compact',{})
    for feature_set in ['compact','context']:
        for alpha in [1.,10.,100.]: add('regression','ridge',feature_set,{'alpha':alpha})
        add('regression','huber',feature_set,{'alpha':.1,'epsilon':1.35})
        for depth,leaf in [(3,5),(None,10)]:
            add('regression','forest',feature_set,{'max_depth':depth,'min_samples_leaf':leaf,'criterion':'absolute_error'})
        add('regression','extra',feature_set,{'max_depth':None,'min_samples_leaf':5,'criterion':'absolute_error'})
        for nodes,leaf in [(3,10),(7,15)]:
            add('regression','hist',feature_set,{'loss':'absolute_error','max_leaf_nodes':nodes,'min_samples_leaf':leaf,'l2_regularization':1.})
        for c in [.1,1.,10.]:
            for weight in [None,'balanced']: add('classification','logistic',feature_set,{'C':c,'class_weight':weight})
        for depth,leaf in [(3,5),(6,10)]:
            for weight in [None,'balanced_subsample']:
                add('classification','forest',feature_set,{'max_depth':depth,'min_samples_leaf':leaf,'class_weight':weight,'max_features':'sqrt'})
        add('classification','extra',feature_set,{'max_depth':None,'min_samples_leaf':5,'class_weight':'balanced','max_features':'sqrt'})
        for nodes,leaf in [(3,10),(7,15)]:
            add('classification','hist',feature_set,{'max_leaf_nodes':nodes,'min_samples_leaf':leaf,'class_weight':'balanced','l2_regularization':1.})
    return items
