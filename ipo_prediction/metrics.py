"""Metrics use decimal proportions, except explicitly labeled INR/percent values."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score)


def regression_metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not np.isfinite(actual).all() or not (actual > 0).all() or not np.isfinite(predicted).all():
        raise ValueError("Invalid price data for metrics")
    errors = predicted - actual
    return {"mae_inr": float(np.mean(np.abs(errors))),
            "rmse_inr": float(np.sqrt(np.mean(errors ** 2))),
            "mape_pct": float(np.mean(np.abs(errors) / actual) * 100),
            "n": len(actual)}


def classification_metrics(actual, probability, threshold=0.5):
    actual, probability = np.asarray(actual, dtype=int), np.asarray(probability, dtype=float)
    if not np.isin(actual, [0, 1]).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid classification data")
    if not ((probability >= 0) & (probability <= 1)).all():
        raise ValueError("Probabilities must be in [0, 1]")
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "precision_apply": float(precision_score(actual, predicted, zero_division=0)),
            "recall_apply": float(recall_score(actual, predicted, zero_division=0)),
            "f1_apply": float(f1_score(actual, predicted, zero_division=0)),
            "specificity_skip": float(tn / (tn + fp)) if tn + fp else 0.0,
            "brier": float(brier_score_loss(actual, probability)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "predicted_apply": int(predicted.sum()), "predicted_skip": int((1 - predicted).sum()),
            "n": len(actual)}
