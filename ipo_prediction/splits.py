"""Rebuild and verify time-ordered, label-mature evaluation memberships."""
import pandas as pd
from .paths import project_root, read_json


def make_splits(dataset, verify_frozen=True):
    plan = read_json("config/evaluation.json")
    dev_boundary = pd.Timestamp(plan["development_cutoff_exclusive"])
    test_fit = pd.Timestamp(plan["final_fit_cutoff"])
    before = dataset.s0_cutoff_at.lt(dev_boundary)
    dev = before & dataset.label_available_at.lt(dev_boundary)
    test = ~before
    pending = before & ~dev
    # Preserve the existing conservative development contract: boundary rows stay out.
    if not dataset.loc[dev, "label_available_at"].lt(test_fit).all():
        raise ValueError("Final training labels unavailable at cutoff")
    final = dataset[["ipo_id"]].copy()
    final["role"] = "purged_label_unavailable"
    final.loc[dev, "role"] = "final_train"
    final.loc[test, "role"] = "reserved_test"
    if (int(dev.sum()), int(test.sum()), int(pending.sum())) != (149, 103, 1):
        raise ValueError("Frozen final partition counts changed")
    parts = []
    for fold in plan["validation_folds"]:
        fid = fold["fold_id"]
        quarter = int(fid[-1])
        valid = dev & dataset.s0_cutoff_at.dt.year.eq(2024) & dataset.s0_cutoff_at.dt.quarter.eq(quarter)
        cutoff = dataset.loc[valid, "s0_cutoff_at"].min()
        if cutoff != pd.Timestamp(fold["fit_cutoff"]):
            raise ValueError(f"Changed cutoff in {fid}")
        earlier = dev & dataset.s0_cutoff_at.lt(cutoff)
        train = earlier & dataset.label_available_at.lt(cutoff)
        purged = earlier & ~train
        if (int(train.sum()), int(valid.sum()), int(purged.sum())) != (
            int(fold["training_IPOs"]), int(fold["validation_IPOs"]), int(fold["purged_unmatured_labels"])):
            raise ValueError(f"Changed membership counts in {fid}")
        for role, mask in [("train", train), ("validation", valid), ("purged_unmatured", purged)]:
            rows = dataset.loc[mask, ["ipo_id"]].copy()
            rows["fold_id"], rows["role"], rows["fit_cutoff"] = fid, role, cutoff
            parts.append(rows)
    walk = pd.concat(parts, ignore_index=True)
    if verify_frozen:
        for generated, file, columns in [
            (final, "final_membership.csv", ["ipo_id", "role"]),
            (walk, "walk_forward_membership.csv", ["fold_id", "ipo_id", "role"]),
        ]:
            frozen = pd.read_csv(project_root() / "data/splits" / file)
            left = set(map(tuple, generated[columns].to_numpy()))
            right = set(map(tuple, frozen[columns].to_numpy()))
            if left != right or len(generated) != len(frozen):
                raise ValueError(f"Frozen split mismatch: {file}")
    return final.reset_index(drop=True), walk


def fold_ids(walk, fold_id, role):
    return walk.loc[walk.fold_id.eq(fold_id) & walk.role.eq(role), "ipo_id"].tolist()
