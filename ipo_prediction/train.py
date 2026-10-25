"""Part 2: development-only walk-forward comparisons and frozen selection."""
import hashlib
import importlib.metadata
import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from .data import feature_names, get_xy, load_dataset
from .metrics import classification_metrics, regression_metrics
from .models import candidate_specs, make_model, positive_probability
from .paths import project_root, read_json
from .splits import fold_ids, make_splits


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def development_inputs():
    """Separate development rows before any candidate fitting or scoring."""
    all_data = load_dataset()  # Existing schema/label integrity checks, not test scoring.
    final, walk = make_splits(all_data)
    ids = final.loc[final.role.eq("final_train"), "ipo_id"].tolist()
    development = all_data.loc[ids].copy()
    if len(development) != 149:
        raise ValueError("Expected fixed development population")
    if not set(walk.ipo_id).issubset(development.index):
        raise ValueError("Non-development member entered walk-forward data")
    return development, walk


def evaluate_candidates(development, walk, specs, threshold):
    """Accept only development data; no reserved rows are supplied to this loop."""
    plan = read_json("config/evaluation.json")
    boundary = pd.Timestamp(plan["development_cutoff_exclusive"])
    if len(development) != 149 or not development.s0_cutoff_at.lt(boundary).all():
        raise ValueError("Training function accepts only the 149 development IPOs")
    if not development.label_available_at.lt(boundary).all():
        raise ValueError("Development includes unavailable labels")
    rows, fold_scores, audits, models, warning_rows = [], [], [], {}, []
    for spec in specs:
        cid = spec["candidate_id"]
        for fid in sorted(walk.fold_id.unique()):
            train_ids, valid_ids = fold_ids(walk, fid, "train"), fold_ids(walk, fid, "validation")
            if set(train_ids) & set(valid_ids):
                raise ValueError("Train/validation overlap")
            fit_cutoff = development.loc[valid_ids, "s0_cutoff_at"].min()
            if not development.loc[train_ids, "label_available_at"].lt(fit_cutoff).all():
                raise ValueError("Immature label in candidate fitting")
            if not development.loc[train_ids, "s0_cutoff_at"].lt(fit_cutoff).all():
                raise ValueError("Future feature row in training")
            x_train, y_train = get_xy(development, train_ids, spec["feature_set"])
            x_valid, y_valid = get_xy(development, valid_ids, spec["feature_set"])
            target = "listing_open_inr" if spec["task"] == "regression" else "target_positive_listing_gain"
            model = make_model(spec)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x_train, y_train[target])
            for warning in caught:
                warning_rows.append({"candidate_id": cid, "fold_id": fid,
                    "category": warning.category.__name__, "message": str(warning.message)})
                if issubclass(warning.category, ConvergenceWarning):
                    raise RuntimeError(f"Unconverged candidate: {cid} {fid}")
            imputer = (model.named_steps["preprocessing"].named_steps["imputer"]
                       if "preprocessing" in model.named_steps else None)
            before = imputer.statistics_.copy() if imputer is not None else None
            if spec["task"] == "regression":
                predictions = model.predict(x_valid)
                if not np.isfinite(predictions).all() or not (predictions > 0).all():
                    raise ValueError("Invalid predicted listing price")
                metrics = regression_metrics(y_valid[target], predictions)
            else:
                predictions = positive_probability(model, x_valid)
                metrics = classification_metrics(y_valid[target], predictions, threshold)
            if imputer is not None and not np.array_equal(before, imputer.statistics_, equal_nan=True):
                raise ValueError("Prediction changed training-fitted preprocessing")
            for position, ipo_id in enumerate(valid_ids):
                rows.append({"candidate_id": cid, "task": spec["task"], "fold_id": fid,
                    "ipo_id": ipo_id, "company_name": development.loc[ipo_id, "company_name"],
                    "s0_cutoff_at": development.loc[ipo_id, "s0_cutoff_at"],
                    "fit_cutoff": fit_cutoff, "actual": float(y_valid.loc[ipo_id, target]),
                    "prediction": float(predictions[position])})
            fold_scores.append({"candidate_id": cid, "task": spec["task"], "fold_id": fid, **metrics})
            audits.append({"candidate_id": cid, "fold_id": fid, "fit_cutoff": fit_cutoff,
                "training_IPOs": len(train_ids), "validation_IPOs": len(valid_ids),
                "latest_training_label": development.loc[train_ids, "label_available_at"].max(),
                "input_features": len(x_train.columns),
                "reserved_rows_used": 0, "preprocessing_unchanged_by_prediction": True})
            models[(cid, fid)] = model
        print(f"Validated {cid}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(fold_scores), pd.DataFrame(audits), models, warning_rows


def rank_candidates(predictions, specs, threshold):
    rows = []
    for spec in specs:
        frame = predictions.loc[predictions.candidate_id.eq(spec["candidate_id"])]
        if len(frame) != 90 or frame.ipo_id.nunique() != 90:
            raise ValueError("Each candidate must cover the same 90 unique validation IPOs")
        metrics = (regression_metrics(frame.actual, frame.prediction) if spec["task"] == "regression"
                   else classification_metrics(frame.actual, frame.prediction, threshold))
        used = (0 if spec["family"] == "always_apply" else 1) if spec["is_baseline"] else len(feature_names(spec["feature_set"]))
        rows.append({"candidate_id": spec["candidate_id"], "task": spec["task"],
                     "family": spec["family"], "feature_set": spec["feature_set"],
                     "is_baseline": spec["is_baseline"], "predictors_used": used, **metrics})
    table = pd.DataFrame(rows)
    regression = table.loc[table.task.eq("regression")].sort_values(
        ["mae_inr", "rmse_inr", "predictors_used", "candidate_id"]).reset_index(drop=True)
    classification = table.loc[table.task.eq("classification")].sort_values(
        ["accuracy", "balanced_accuracy", "brier", "predictors_used", "candidate_id"],
        ascending=[False, False, True, True, True]).reset_index(drop=True)
    for frame in [regression, classification]:
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return regression.dropna(axis=1, how="all"), classification.dropna(axis=1, how="all")


def make_figures(regression, classification, predictions, folds, selection, threshold, destination):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(project_root() / ".runtime/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(destination / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        names.append(name)
    def short(cid):
        return cid.replace("regression_", "").replace("classification_", "").replace("_", " ")
    for table, task, column, title, xlabel in [
        (regression, "regression", "mae_inr", "Opening-price validation error — 90 IPOs", "MAE (INR; lower is better)"),
        (classification, "classification", "accuracy", "Apply/Skip validation — 90 IPOs", "Accuracy / balanced accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = np.arange(len(table))
        if task == "regression":
            ax.barh(positions, table[column], color=["#9aabb6" if b else "#347ca5" for b in table.is_baseline])
        else:
            ax.barh(positions - .17, table.accuracy, height=.32, label="Accuracy", color="#347ca5")
            ax.barh(positions + .17, table.balanced_accuracy, height=.32, label="Balanced accuracy", color="#d29337")
            ax.axvline(.90, color="#777777", linestyle=":", label="90% aspiration")
            ax.set_xlim(0, 1)
            ax.legend(loc="lower right", fontsize=8)
        ax.set_yticks(positions, [short(cid) for cid in table.candidate_id], fontsize=8)
        ax.invert_yaxis()
        ax.set(title=title, xlabel=xlabel)
        ax.grid(axis="x", alpha=.2)
        save(fig, f"01_{task}_comparison.png" if task == "regression" else "02_classification_comparison.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, task, metric in [(axes[0], "regression", "mae_inr"), (axes[1], "classification", "accuracy")]:
        ids = list(dict.fromkeys([selection[task]["candidate_id"], selection[task]["best_learned_candidate_id"],
               "regression_training_median_premium_core_baseline" if task == "regression" else "classification_always_apply_core_baseline"]))
        for cid in ids:
            sub = folds.loc[folds.candidate_id.eq(cid)].sort_values("fold_id")
            ax.plot(sub.fold_id, sub[metric], marker="o", label=short(cid))
        ax.set(title=f"{task.title()} by quarter", ylabel="MAE (INR)" if task == "regression" else "Accuracy")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    save(fig, "03_fold_stability.png")
    reg = predictions.loc[predictions.candidate_id.eq(selection["regression"]["candidate_id"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(reg.actual, reg.prediction, alpha=.7)
    high = max(reg.actual.max(), reg.prediction.max()) * 1.05
    axes[0].plot([0, high], [0, high], "--", color="gray")
    axes[0].set(xlabel="Actual opening price (INR)", ylabel="Predicted opening price (INR)", title="Selected model: development predictions")
    axes[1].scatter(reg.actual, reg.prediction - reg.actual, alpha=.7)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set(xlabel="Actual opening price (INR)", ylabel="Prediction minus actual (INR)", title="Price residuals")
    save(fig, "04_selected_price_diagnostics.png")
    clf = classification.loc[classification.candidate_id.eq(selection["classification"]["candidate_id"])].iloc[0]
    matrix = np.array([[clf.tn, clf.fp], [clf.fn, clf.tp]], dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18,
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Skip", "Apply"], yticklabels=["Non-positive", "Positive"],
                xlabel="Prediction", ylabel="Actual listing outcome", title="Selected classifier: validation")
    selected = predictions.loc[predictions.candidate_id.eq(selection["classification"]["candidate_id"])]
    for label, color in [(0, "#c7793d"), (1, "#347ca5")]:
        axes[1].hist(selected.loc[selected.actual.eq(label), "prediction"], bins=np.linspace(0, 1, 11),
                     alpha=.6, label="Positive" if label else "Non-positive", color=color)
    axes[1].axvline(threshold, color="black", linestyle="--", label="Fixed threshold")
    axes[1].set(xlabel="Raw positive-class model probability", ylabel="Validation IPOs", title="Probabilities are not calibrated")
    axes[1].legend(fontsize=8)
    save(fig, "05_selected_classification_diagnostics.png")
    return names


def run():
    root = project_root()
    output, model_dir, figures = root / "outputs/part2", root / "models/validation", root / "figures/part2"
    for path in [output, model_dir, figures]:
        path.mkdir(parents=True, exist_ok=True)
    plan = read_json("config/model_search.json")
    threshold = plan["classification_threshold"]
    specs = candidate_specs(plan)
    # Verify immutable bundled inputs before fitting anything.
    original = pd.read_csv(root / "data/provenance/bundled_inputs.csv")
    for row in original.itertuples():
        if hashlib.sha256((root / row.local_path).read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Bundled input changed: {row.local_path}")
    tracked = [*original.local_path, "config/features.json", "config/evaluation.json", "config/model_search.json",
               "ipo_prediction/models.py", "ipo_prediction/metrics.py", "ipo_prediction/train.py",
               "ipo_prediction/preprocessing.py", "ipo_prediction/data.py", "ipo_prediction/splits.py"]
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in tracked}
    development, walk = development_inputs()
    write_json(output / "candidate_registry.json", specs)
    predictions, folds, audit, fitted, warning_rows = evaluate_candidates(development, walk, specs, threshold)
    regression, classification = rank_candidates(predictions, specs, threshold)
    by_id = {spec["candidate_id"]: spec for spec in specs}
    selection = {}
    for task, table in [("regression", regression), ("classification", classification)]:
        winner = table.iloc[0]
        best_learned = table.loc[~table.is_baseline].iloc[0]
        selection[task] = {**by_id[winner.candidate_id],
            "best_learned_candidate_id": best_learned.candidate_id,
            "validation_metrics": {k: winner[k] for k in table.columns if k not in ["candidate_id", "task", "family", "feature_set", "is_baseline", "rank"]}}
    versions = {name: importlib.metadata.version(name) for name in ["scikit-learn", "pandas", "numpy", "joblib"]}
    frozen = {"stage": "S0", "status": "choices frozen after development validation; final fitting/evaluation deferred",
        "classification_threshold": threshold, "regression_target": "listing_open_inr",
        "classification_target": "target_positive_listing_gain", "selection": selection,
        "development_IPOs": 149, "validation_IPOs": 90, "reserved_IPOs": 103,
        "input_and_code_sha256": hashes, "versions": versions,
        "metric_interpretation": "Pooled development validation used for selection; not independent final-test performance",
        "probability_note": "Raw classifier probabilities; no calibration or guaranteed investment confidence"}
    freeze_path = root / "config/selected_models.json"
    if freeze_path.exists() and json.loads(freeze_path.read_text()) != json.loads(json.dumps(frozen, default=str)):
        raise RuntimeError("Frozen selection differs. Review explicitly instead of silently overwriting choices.")
    write_json(freeze_path, frozen)
    predictions.to_csv(output / "validation_predictions.csv", index=False)
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    folds.to_csv(output / "fold_metrics.csv", index=False)
    audit.to_csv(output / "training_audit.csv", index=False)
    pd.DataFrame(warning_rows, columns=["candidate_id", "fold_id", "category", "message"]).to_csv(output / "training_warnings.csv", index=False)
    regression.to_csv(output / "regression_comparison.csv", index=False)
    classification.to_csv(output / "classification_comparison.csv", index=False)
    selected_ids = {item["candidate_id"] for item in selection.values()}
    selected_predictions = predictions.loc[predictions.candidate_id.isin(selected_ids)].copy()
    selected_predictions["absolute_error_inr"] = np.where(selected_predictions.task.eq("regression"),
        np.abs(selected_predictions.prediction - selected_predictions.actual), np.nan)
    selected_predictions["predicted_class"] = np.where(selected_predictions.task.eq("classification"),
        (selected_predictions.prediction >= threshold).astype(int), np.nan)
    selected_predictions.to_csv(output / "selected_validation_predictions.csv", index=False)
    price_errors = selected_predictions.loc[selected_predictions.task.eq("regression")].sort_values("absolute_error_inr", ascending=False)
    price_errors.head(10).to_csv(output / "largest_price_errors.csv", index=False)
    class_errors = selected_predictions.loc[selected_predictions.task.eq("classification") &
        selected_predictions.actual.ne(selected_predictions.predicted_class)]
    class_errors.to_csv(output / "classification_errors.csv", index=False)
    # Persist only selected and best-learned challenger fold models, never final-fit models.
    saved = []
    keep = selected_ids | {item["best_learned_candidate_id"] for item in selection.values()}
    for cid, fid in sorted(fitted):
        if cid not in keep:
            continue
        path = model_dir / f"{cid}__{fid}.joblib"
        train_ids, valid_ids = fold_ids(walk, fid, "train"), fold_ids(walk, fid, "validation")
        bundle = {"role": "development_validation_only", "spec": by_id[cid], "fold_id": fid,
            "training_ids": train_ids, "validation_ids": valid_ids, "threshold": threshold,
            "input_features": feature_names(by_id[cid]["feature_set"]), "model": fitted[(cid, fid)]}
        joblib.dump(bundle, path, compress=3)
        reloaded = joblib.load(path)
        x_valid, _ = get_xy(development, valid_ids, by_id[cid]["feature_set"])
        replay = (reloaded["model"].predict(x_valid) if by_id[cid]["task"] == "regression"
                  else positive_probability(reloaded["model"], x_valid))
        expected = predictions.loc[predictions.candidate_id.eq(cid) & predictions.fold_id.eq(fid)].prediction.to_numpy()
        np.testing.assert_allclose(replay, expected, rtol=1e-12, atol=1e-12)
        saved.append({"path": str(path.relative_to(root)), "candidate_id": cid, "fold_id": fid,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reloaded_predictions_match": True})
    pd.DataFrame(saved).to_csv(output / "model_artifacts.csv", index=False)
    names = make_figures(regression, classification, predictions, folds, selection, threshold, figures)
    checks = [
        ("bundled source data unchanged", True),
        ("only 149 development rows supplied to modeling", len(development) == 149),
        ("same 90 development validation IPOs for every candidate", predictions.groupby("candidate_id").ipo_id.nunique().eq(90).all()),
        ("all 19 prespecified candidates evaluated", len(regression) == 10 and len(classification) == 9),
        ("all 76 candidate/fold fits audited", len(audit) == 76),
        ("training labels mature before fold cutoff", (pd.to_datetime(audit.latest_training_label) < pd.to_datetime(audit.fit_cutoff)).all()),
        ("reserved rows not used for fitting or scoring", audit.reserved_rows_used.eq(0).all()),
        ("no training convergence warnings", not any(w["category"] == "ConvergenceWarning" for w in warning_rows)),
        ("saved fold models reproduce validation predictions", all(item["reloaded_predictions_match"] for item in saved)),
        ("model choices and threshold frozen", freeze_path.exists()),
        ("prepared data and splits remain unchanged after modeling", all(hashlib.sha256((root / n).read_bytes()).hexdigest() == value for n, value in hashes.items())),
    ]
    check_frame = pd.DataFrame(checks, columns=["check", "passed"])
    check_frame.to_csv(output / "validation_checks.csv", index=False)
    if not check_frame.passed.all():
        raise RuntimeError("Part 2 validation checks failed")
    summary = {"part": 2, "status": "complete", "development_IPOs": 149, "validation_IPOs": 90,
        "regression_candidates": len(regression), "classification_candidates": len(classification),
        "fold_fits": len(audit), "selected_regression": selection["regression"]["candidate_id"],
        "selected_classification": selection["classification"]["candidate_id"],
        "validation_regression_mae_inr": float(regression.iloc[0].mae_inr),
        "validation_classification_accuracy": float(classification.iloc[0].accuracy),
        "saved_development_models": len(saved), "figures": names,
        "checks_passed": len(checks), "reserved_outcome_metrics_computed": False,
        "final_models_fitted": False, "selection_scores_are_final_performance": False}
    write_json(output / "completion_summary.json", summary)
    write_findings(root, summary, regression, classification, selection)
    print(json.dumps(summary, indent=2))
    return summary


def write_findings(root, summary, regression, classification, selection):
    reg, clf = regression.iloc[0], classification.iloc[0]
    cap = regression.loc[regression.family.eq("announced_cap")].iloc[0]
    premium = regression.loc[regression.family.eq("training_median_premium")].iloc[0]
    always = classification.loc[classification.is_baseline].iloc[0]
    learned = classification.loc[~classification.is_baseline].iloc[0]
    text = f"""# Part 2 — Development modeling findings

## What was completed

Compared {len(regression)} regression and {len(classification)} classification candidates across four expanding 2024 folds: 76 fits, 90 unique validation IPOs. Fold training sizes are 59, 80, 94 and 121. All imputers/scalers were fitted on each fold's training rows only. The optional financial extension was compared using development data only. The Apply/Skip threshold stayed fixed at 0.5; no threshold search was performed.

## Opening-price regression

Selected: `{reg.candidate_id}`. Pooled validation MAE: **INR {reg.mae_inr:.2f}**, RMSE: INR {reg.rmse_inr:.2f}, MAPE: {reg.mape_pct:.2f}%.

Announced-cap baseline MAE: INR {cap.mae_inr:.2f}. Training-median-premium baseline MAE: INR {premium.mae_inr:.2f}. The premium baseline uses only training opening-price/announced-cap ratios. Final issue price is never an input.

Price error in rupees can be dominated by expensive IPOs; MAPE and the per-quarter results provide supporting context. Learned price predictions use a prespecified INR 0.01 physical floor.

## Positive-gain / Apply-Skip classification

Selected: `{clf.candidate_id}`. Pooled validation accuracy: **{clf.accuracy:.2%}**; balanced accuracy: {clf.balanced_accuracy:.2%}; Apply precision: {clf.precision_apply:.2%}; Apply recall: {clf.recall_apply:.2%}; F1: {clf.f1_apply:.4f}.

Always-Apply validation accuracy: **{always.accuracy:.2%}** across these 90 rows. This differs from the earlier 78.5% rate across all 149 development rows. Best learned challenger: `{learned.candidate_id}` ({learned.accuracy:.2%} accuracy).

Selected-model confusion counts: TN={int(clf.tn)}, FP={int(clf.fp)}, FN={int(clf.fn)}, TP={int(clf.tp)}. Correctly identified {int(clf.tn)} of {int(clf.tn + clf.fp)} non-positive listings, while missing {int(clf.fn)} of {int(clf.fn + clf.tp)} positive listings. Predicted Apply: {int(clf.predicted_apply)}; Skip: {int(clf.predicted_skip)}.

The 90% accuracy aspiration {'was reached on selection validation, but still needs independent final evaluation' if clf.accuracy >= .9 else 'was not reached on development validation'}. Raw model probabilities are not calibrated confidence estimates. Apply is a prediction of positive listing gain, without allotment or costs; it does not establish investment profitability.

## Selection and limitations

Regression was ranked by pooled MAE, then RMSE, fewer used predictors and candidate ID. Classification was ranked by pooled accuracy, then balanced accuracy, Brier score, fewer used predictors and candidate ID. Baselines were eligible to win. All candidates and results are retained; a stronger learned model was not assumed in advance.

These validation rows were used for model/feature selection, so selected scores are optimistic development estimates, not final performance. A quarter has only one non-positive listing; quarterly metrics are unstable. Sample size, missing financials, limited manager history and inherited point-in-time source assumptions remain limitations. Separate price/classification models can disagree because their targets differ.

## Part 3 handoff

`config/selected_models.json` freezes choices, threshold, input/code hashes and library versions. `models/validation/` contains selected and best-learned challenger fold pipelines with training/validation IDs; these are development models, not production models. Saved models were reloaded and reproduced their validation predictions. `outputs/part2/` holds full comparisons, per-fold metrics, per-IPO predictions, audits and error cases; `figures/part2/` holds five charts.

No model was fitted on the full 149-row development set yet. No 2025 reserved predictions or outcome metrics were computed. Part 3 will fit the frozen choices on 149 IPOs, evaluate once on the 103 reserved IPOs and build the local demonstration. The reserved period is not claimed as unseen external data because earlier preparation displayed full-cohort aggregate outcomes.
"""
    (root / "docs/part2_findings.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run()
