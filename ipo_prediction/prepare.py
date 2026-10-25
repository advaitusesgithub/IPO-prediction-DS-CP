"""Run Part 1 without fitting a predictive model or evaluating reserved outcomes."""
import hashlib
import importlib.metadata
import json
import platform
import numpy as np
import pandas as pd
from .data import load_dataset, get_xy
from .paths import project_root, read_json
from .preprocessing import make_preprocessor
from .splits import make_splits, fold_ids


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def run():
    root = project_root()
    output = root / "outputs/part1"
    output.mkdir(parents=True, exist_ok=True)
    # These hashes protect local prepared inputs; source audit hashes refer to
    # earlier extraction, whose large original documents are not runtime inputs.
    manifest = pd.read_csv(root / "data/provenance/bundled_inputs.csv")
    for row in manifest.itertuples():
        if hashlib.sha256((root / row.local_path).read_bytes()).hexdigest() != row.sha256:
            raise ValueError(f"Bundled input hash changed: {row.local_path}")
    data = load_dataset()
    final, walk = make_splits(data)
    final.to_csv(output / "final_membership.csv", index=False)
    walk.to_csv(output / "walk_forward_membership.csv", index=False)
    fold_summary = walk.groupby(["fold_id", "role"]).size().unstack(fill_value=0)
    fold_summary.to_csv(output / "fold_summary.csv")
    train_ids = final.loc[final.role.eq("final_train"), "ipo_id"].tolist()
    train = data.loc[train_ids]
    profile = pd.DataFrame({
        "non_missing": train[read_json("config/features.json")["core_predictors"]].notna().sum(),
        "missing": train[read_json("config/features.json")["core_predictors"]].isna().sum(),
    }).rename_axis("feature").reset_index()
    profile.to_csv(output / "development_feature_coverage.csv", index=False)
    # Demonstrate only the earliest development fold's training/validation
    # preprocessing. These diagnostic matrices are not final model artifacts.
    first = "2024_Q1"
    training_ids, validation_ids = fold_ids(walk, first, "train"), fold_ids(walk, first, "validation")
    checks = [{"check": "bundled input hashes", "passed": True},
              {"check": "input identities, labels and S0 schema", "passed": True},
              {"check": "exact frozen chronological memberships", "passed": True}]
    preprocessing_rows = []
    for features in ["core", "extended"]:
        x_train, _ = get_xy(data, training_ids, features)
        x_valid, _ = get_xy(data, validation_ids, features)
        for scale in [False, True]:
            pipeline = make_preprocessor(features, scale=scale)
            transformed_train = pipeline.fit_transform(x_train)
            before = pipeline.named_steps["imputer"].statistics_.copy()
            transformed_valid = pipeline.transform(x_valid)
            assert np.array_equal(before, pipeline.named_steps["imputer"].statistics_, equal_nan=True)
            assert np.isfinite(transformed_train.to_numpy()).all()
            assert np.isfinite(transformed_valid.to_numpy()).all()
            assert list(transformed_train.columns) == list(transformed_valid.columns)
            label = f"{features}_{'scaled' if scale else 'unscaled'}"
            transformed_train.to_parquet(output / f"diagnostic_{first}_{label}_train.parquet")
            transformed_valid.to_parquet(output / f"diagnostic_{first}_{label}_validation.parquet")
            preprocessing_rows.append({"feature_set": features, "scaled": scale,
                "training_rows": len(x_train), "validation_rows": len(x_valid),
                "input_features": x_train.shape[1], "output_features": transformed_train.shape[1]})
            checks.append({"check": f"finite, aligned, frozen development preprocessing: {label}", "passed": True})
    pd.DataFrame(preprocessing_rows).to_csv(output / "preprocessing_readiness.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "validation_checks.csv", index=False)
    packages = ["pandas", "numpy", "pyarrow", "scikit-learn", "scipy", "matplotlib", "joblib",
                "streamlit", "nbformat", "nbclient", "ipykernel", "pytest"]
    write_json(output / "environment.json", {"python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages}})
    summary = {"part": 1, "status": "complete", "cohort": len(data),
        "source_predictors_preserved": 52, "active_core_features": 12,
        "optional_financial_features": 7,
        "final_memberships": final.role.value_counts().to_dict(),
        "validation_folds": int(walk.fold_id.nunique()), "checks_passed": len(checks),
        "predictive_models_trained": False, "reserved_outcome_metrics_computed": False,
        "preprocessing_demo": "2024 Q1 development training only; not a saved final pipeline"}
    write_json(output / "readiness_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
