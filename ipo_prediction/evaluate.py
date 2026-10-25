"""Fit frozen choices and record one chronological evaluation; reuse on reruns."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .data import load_dataset, get_xy, feature_names
from .metrics import regression_metrics, classification_metrics
from .models import make_model, positive_probability
from .paths import project_root, read_json
from .splits import make_splits


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def run():
    root = project_root()
    out = root / "outputs/part3"
    out.mkdir(parents=True, exist_ok=True)
    frozen = read_json("config/selected_models.json")
    for name, expected in frozen["input_and_code_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen input/code changed: {name}")
    for name, expected in frozen["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Frozen library version changed: {name}")
    inputs = {**frozen["input_and_code_sha256"],
              "config/selected_models.json": digest(root / "config/selected_models.json"),
              "ipo_prediction/evaluate.py": digest(Path(__file__))}
    receipt_path = out / "evaluation_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["input_sha256"] != inputs:
            raise ValueError("Evaluation already recorded against different inputs; no silent reevaluation")
        for name, expected in receipt["artifact_sha256"].items():
            if not (root / name).exists() or digest(root / name) != expected:
                raise ValueError(f"Recorded evaluation artifact changed: {name}")
        print("Reusing recorded evaluation; no fitting or scoring repeated.")
        return read_json("outputs/part3/completion_summary.json")
    data = load_dataset()
    final, _ = make_splits(data)
    train_ids = final.loc[final.role.eq("final_train"), "ipo_id"].tolist()
    test_ids = final.loc[final.role.eq("reserved_test"), "ipo_id"].tolist()
    cutoff = pd.Timestamp(read_json("config/evaluation.json")["final_fit_cutoff"])
    if len(train_ids) != 149 or len(test_ids) != 103 or set(train_ids) & set(test_ids):
        raise ValueError("Unexpected final partitions")
    if not data.loc[train_ids, "label_available_at"].lt(cutoff).all():
        raise ValueError("Training label unavailable at final fit cutoff")
    if not data.loc[train_ids, "s0_cutoff_at"].lt(cutoff).all():
        raise ValueError("Future training row")
    if data.loc[test_ids, "s0_cutoff_at"].min() != cutoff:
        raise ValueError("Final cutoff differs from first test S0")
    predictions = data.loc[test_ids, ["ipo_id", "company_name", "s0_cutoff_at"]].copy().reset_index(drop=True)
    model_dir = root / "models/final"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for task in ["regression", "classification"]:
        spec = frozen["selection"][task]
        x_train, y_train = get_xy(data, train_ids, spec["feature_set"])
        x_test, _ = get_xy(data, test_ids, spec["feature_set"])
        target = frozen[f"{task}_target"]
        model = make_model(spec).fit(x_train, y_train[target])
        predicted = model.predict(x_test) if task == "regression" else positive_probability(model, x_test)
        column = "predicted_open_inr" if task == "regression" else "positive_gain_probability"
        predictions[column] = predicted
        path = model_dir / f"{task}.joblib"
        joblib.dump({"role": "final_development_fit", "spec": spec, "model": model,
                     "training_ids": train_ids, "fit_cutoff": str(cutoff),
                     "input_features": feature_names(spec["feature_set"]),
                     "threshold": frozen["classification_threshold"]}, path, compress=3)
        reloaded = joblib.load(path)["model"]
        replay = reloaded.predict(x_test) if task == "regression" else positive_probability(reloaded, x_test)
        np.testing.assert_allclose(replay, predicted, rtol=1e-12, atol=1e-12)
        artifacts.append(str(path.relative_to(root)))
    threshold = frozen["classification_threshold"]
    predictions["predicted_class"] = (predictions.positive_gain_probability >= threshold).astype(int)
    predictions["signal"] = np.where(predictions.predicted_class.eq(1), "Apply", "Skip")
    actual = data.loc[test_ids, ["ipo_id", "listing_open_inr", "final_issue_price_inr",
        "target_positive_listing_gain", "target_listing_gain_open", "label_verification_tier"]].reset_index(drop=True)
    scores = []
    for name, values in [("selected_price", predictions.predicted_open_inr),
                         ("announced_cap_baseline", data.loc[test_ids, "band_high_inr"])]:
        scores.append({"task": "regression", "model": name, **regression_metrics(actual.listing_open_inr, values)})
    for name, values in [("selected_classifier", predictions.positive_gain_probability),
                         ("always_apply_baseline", np.ones(len(test_ids)))]:
        scores.append({"task": "classification", "model": name,
                       **classification_metrics(actual.target_positive_listing_gain, values, threshold)})
    metrics = pd.DataFrame(scores)
    metrics.to_csv(out / "metrics.csv", index=False)
    for name, frame in [("heldout_predictions", predictions), ("evaluation_outcomes", actual),
                        ("demo_inputs", data.loc[test_ids, ["ipo_id", *feature_names()]].reset_index(drop=True))]:
        frame.to_parquet(out / f"{name}.parquet", index=False)
        frame.to_csv(out / f"{name}.csv", index=False)
    joined = predictions.merge(actual, on="ipo_id", validate="one_to_one")
    joined["absolute_error_inr"] = abs(joined.predicted_open_inr - joined.listing_open_inr)
    joined.nlargest(10, "absolute_error_inr").to_csv(out / "largest_price_errors.csv", index=False)
    joined.loc[joined.predicted_class.ne(joined.target_positive_listing_gain)].to_csv(out / "classification_errors.csv", index=False)
    generate_figures(joined, root / "figures/part3")
    clf = scores[2]
    baseline = scores[3]
    checks = pd.DataFrame([
        ("Frozen inputs/code and versions verified", True),
        ("149 mature training labels before first test cutoff", True),
        ("103 disjoint reserved IPO predictions", len(predictions) == 103),
        ("Both saved pipelines reproduce held-out predictions", True),
        ("Only frozen settings used; threshold remains 0.5", threshold == .5),
        ("Positive finite price predictions", bool(np.isfinite(predictions.predicted_open_inr).all() and predictions.predicted_open_inr.gt(0).all())),
        ("Probability values in [0,1]", bool(predictions.positive_gain_probability.between(0, 1).all())),
    ], columns=["check", "passed"])
    if not checks.passed.all():
        raise ValueError("Final evaluation validation failed")
    checks.to_csv(out / "validation_checks.csv", index=False)
    summary = {"part": 3, "evaluation_status": "complete", "training_IPOs": 149,
        "reserved_IPOs": 103, "fit_cutoff": str(cutoff), "classification_threshold": threshold,
        "regression": scores[0], "cap_baseline": scores[1], "classification": clf,
        "always_apply_baseline": baseline,
        "classification_beats_always_apply": clf["accuracy"] > baseline["accuracy"],
        "accuracy_90_percent_reached": clf["accuracy"] >= .9,
        "validation_checks_passed": len(checks), "models": artifacts,
        "probability_note": "Raw, uncalibrated classifier probabilities",
        "test_note": "Reserved chronological evaluation; prior preparation exposed full-cohort aggregate outcomes"}
    write_json(out / "completion_summary.json", summary)
    (root / "docs/part3_findings.md").write_text(f'''# Final chronological evaluation

The frozen choices were fitted on 149 development IPOs using labels available before {cutoff}. Both fitted pipelines stayed fixed for all 103 reserved IPOs. Historical context inputs can reflect already observable past outcomes; no test-label fitting or tuning occurs.

Price MAE: INR {scores[0]['mae_inr']:.2f}; RMSE: INR {scores[0]['rmse_inr']:.2f}; MAPE: {scores[0]['mape_pct']:.2f}%. Announced-cap baseline MAE: INR {scores[1]['mae_inr']:.2f}. The selected price method is the training-median-premium baseline, not a learned forest.

Classifier accuracy: {clf['accuracy']:.2%}; balanced accuracy: {clf['balanced_accuracy']:.2%}; Apply precision: {clf['precision_apply']:.2%}; recall: {clf['recall_apply']:.2%}; F1: {clf['f1_apply']:.4f}. Always-Apply accuracy: {baseline['accuracy']:.2%}. Confusion counts: TN={clf['tn']}, FP={clf['fp']}, FN={clf['fn']}, TP={clf['tp']}. Apply predictions: {clf['predicted_apply']}; Skip predictions: {clf['predicted_skip']}.

90% accuracy reached: {clf['accuracy'] >= .9}. Better accuracy than always Apply: {clf['accuracy'] > baseline['accuracy']}. Report these measured results without treating the class imbalance as useful Skip detection. Probabilities are uncalibrated. Apply means predicted positive listing gain before costs and without allotment; no investment-profit claim is supported.

The local historical demonstration uses saved reserved predictions, with actual outcomes in a separate evaluation view. Models were reloaded and reproduced the recorded predictions. Reruns reuse the evaluation receipt and do not fit or score again. Changes to recorded inputs/artifacts fail verification.

Limitations remain: small sample, incomplete financial coverage, cohort-limited manager history, assumptions about source publication availability, and one weaker label-verification tier. Prior data preparation displayed aggregate full-cohort outcomes, so this reserved period is not described as unseen external data. Separate price and classification outputs have different targets and can disagree. No model settings or threshold were changed after evaluation. Submission packaging remains Part 4.
''')
    # Seal evaluation outputs and fitted pipelines; tests/docs/app can be added independently.
    paths = [p for p in out.iterdir() if p.is_file() and p.suffix in {".csv", ".parquet", ".json"}]
    paths += [root / p for p in artifacts]
    paths += list((root / "figures/part3").glob("*.png"))
    write_json(receipt_path, {"input_sha256": inputs,
        "artifact_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        "policy": "Reuse recorded evaluation; never tune on these results"})
    print(json.dumps(summary, indent=2))
    return summary


def generate_figures(frame, destination):
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(project_root() / ".runtime/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(frame.listing_open_inr, frame.predicted_open_inr, alpha=.7)
    top = max(frame.listing_open_inr.max(), frame.predicted_open_inr.max()) * 1.05
    ax.plot([0, top], [0, top], "--", color="gray")
    ax.set(xlabel="Actual opening price (INR)", ylabel="Predicted opening price (INR)", title="Reserved 2025 predictions — 103 IPOs")
    fig.tight_layout(); fig.savefig(destination / "price_predictions.png", dpi=150); plt.close(fig)
    matrix = confusion_matrix(frame.target_positive_listing_gain, frame.predicted_class, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=20,
                    color="white" if matrix[i, j] > matrix.max()/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Skip", "Apply"], yticklabels=["Non-positive", "Positive"],
           xlabel="Predicted", ylabel="Actual", title="Reserved classification results")
    fig.tight_layout(); fig.savefig(destination / "confusion_matrix.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    run()
