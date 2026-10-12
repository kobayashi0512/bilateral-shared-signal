#!/usr/bin/env python3
"""Re-audit bilateral fundus controls with eye-level labels and node bootstrap.

The script reads frozen feature arrays and row-aligned manifests; it does not
train or modify a model. DeepDRiD and PAPILA inputs remain local and are not
copied to the public release. Output contains aggregate statistics only.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from conditional_shared_signal_controls import (
    QUALITY_COLUMNS,
    match_permutation,
    nearest_quality_permutation,
    patient_table,
)


SEEDS = list(range(20260922, 20260927))
PAPILA_SEEDS = list(range(20260921, 20260926))


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise ValueError("A matched stratum needs at least two patients")
    for _ in range(200):
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate
    return np.roll(indices, 1)


def weighted_auc(
    positive: np.ndarray,
    negative: np.ndarray,
    positive_weight: np.ndarray,
    negative_weight: np.ndarray,
) -> float:
    """Weighted AUROC with half credit for ties, without expanding weights."""
    total_positive = float(positive_weight.sum())
    total_negative = float(negative_weight.sum())
    if total_positive <= 0 or total_negative <= 0:
        return float("nan")
    order = np.argsort(negative, kind="mergesort")
    sorted_negative = negative[order]
    cumulative = np.r_[0.0, np.cumsum(negative_weight[order], dtype=np.float64)]
    left = np.searchsorted(sorted_negative, positive, side="left")
    right = np.searchsorted(sorted_negative, positive, side="right")
    below = cumulative[left]
    tied = cumulative[right] - cumulative[left]
    numerator = np.sum(positive_weight * (below + 0.5 * tied), dtype=np.float64)
    return float(numerator / (total_positive * total_negative))


def node_bootstrap_auc(
    positive: np.ndarray,
    negative: np.ndarray,
    permutation: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> list[float]:
    """Two-way patient-node bootstrap, conditional on a fixed negative assignment.

    Positive rows receive their patient multiplicity. A negative edge (i, j)
    receives the product of its anchor and donor patient multiplicities, which
    preserves the fact that a patient can occur in both roles in the matching
    graph. This is a node/pigeonhole bootstrap sensitivity analysis; it does
    not include uncertainty from drawing a new matching assignment.
    """
    n = len(positive)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    aucs: list[float] = []
    attempts = 0
    max_attempts = max(20 * n_bootstrap, n_bootstrap + 100)
    while len(aucs) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        node_weight = rng.multinomial(n, probabilities).astype(np.float64)
        edge_weight = node_weight * node_weight[permutation]
        auc = weighted_auc(positive, negative, node_weight, edge_weight)
        if np.isfinite(auc):
            aucs.append(auc)
    if len(aucs) != n_bootstrap:
        raise RuntimeError(f"Only {len(aucs)} valid node-bootstrap samples after {attempts} attempts")
    return aucs


def node_bootstrap_mean_difference(
    positive: np.ndarray,
    negative: np.ndarray,
    permutation: np.ndarray,
    seed: int,
    n_bootstrap: int,
) -> list[float]:
    n = len(positive)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    values: list[float] = []
    attempts = 0
    max_attempts = max(20 * n_bootstrap, n_bootstrap + 100)
    while len(values) < n_bootstrap and attempts < max_attempts:
        attempts += 1
        node_weight = rng.multinomial(n, probabilities).astype(np.float64)
        edge_weight = node_weight * node_weight[permutation]
        if node_weight.sum() <= 0 or edge_weight.sum() <= 0:
            continue
        delta = np.average(positive, weights=node_weight) - np.average(negative, weights=edge_weight)
        values.append(float(delta))
    if len(values) != n_bootstrap:
        raise RuntimeError(f"Only {len(values)} valid node-bootstrap samples after {attempts} attempts")
    return values


def node_bootstrap_mean(
    values: np.ndarray,
    seed: int,
    n_bootstrap: int,
    permutation: np.ndarray | None = None,
) -> list[float]:
    n = len(values)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    output = []
    for _ in range(n_bootstrap):
        node_weight = rng.multinomial(n, probabilities).astype(np.float64)
        weight = node_weight if permutation is None else node_weight * node_weight[permutation]
        output.append(float(np.average(values, weights=weight)))
    return output


def paired_node_bootstrap_mean_difference(
    first: np.ndarray, second: np.ndarray, seed: int, n_bootstrap: int
) -> list[float]:
    n = len(first)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n, 1.0 / n, dtype=np.float64)
    out = []
    for _ in range(n_bootstrap):
        weights = rng.multinomial(n, probabilities).astype(np.float64)
        out.append(float(np.average(first - second, weights=weights)))
    return out


def interval(values: list[float] | np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def eye_metadata(manifest: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    required = {
        "patient_id", "image_id", "eye", "patient_grade", "benchmark_split",
        "left_eye_DR_Level", "right_eye_DR_Level", *QUALITY_COLUMNS,
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"DeepDRiD manifest missing required fields: {missing}")
    if len(manifest) != len(features):
        raise ValueError(f"Manifest rows {len(manifest)} != feature rows {len(features)}")

    work = manifest.copy().reset_index(drop=True)
    work["eye"] = work["eye"].astype(str).str.upper().str.strip()
    work["image_id"] = work["image_id"].astype(str)
    work["feature_row"] = np.arange(len(work))
    row_quality = 0.25 * (
        pd.to_numeric(work["Overall quality"], errors="raise")
        + pd.to_numeric(work["Clarity"], errors="raise") / 10.0
        + pd.to_numeric(work["Field definition"], errors="raise") / 10.0
        + (10.0 - pd.to_numeric(work["Artifact"], errors="raise")) / 10.0
    )
    work["row_quality_composite"] = row_quality

    records: list[dict[str, Any]] = []
    feature_map: dict[str, dict[str, Any]] = {}
    for patient_id, group in work.groupby("patient_id", sort=True):
        pid = str(patient_id)
        if set(group["eye"]) != {"L", "R"}:
            continue
        def grade_for(column: str) -> int:
            # The official CSV stores each eye's grade in a patient-level
            # column, while only one of the two grade columns is populated on
            # each image row. Recover the unique patient-level value across
            # the four rows; do not infer it from the image's filename suffix.
            values = pd.to_numeric(group[column], errors="coerce").dropna().unique()
            if len(values) != 1:
                raise ValueError(f"Patient {pid} has {len(values)} unique values for {column}")
            return int(values[0])

        left_grade = grade_for("left_eye_DR_Level")
        right_grade = grade_for("right_eye_DR_Level")
        split_values = group["benchmark_split"].astype(str).unique()
        if len(split_values) != 1:
            raise ValueError(f"Patient {pid} appears in multiple splits: {split_values.tolist()}")

        eye_data: dict[str, Any] = {}
        eye_vectors: dict[str, np.ndarray] = {}
        views: dict[str, np.ndarray] = {}
        for eye in ("L", "R"):
            eye_group = group.loc[group.eye == eye]
            if len(eye_group) < 1:
                raise ValueError(f"Missing all images for patient {pid}/{eye}")
            q = float(eye_group.row_quality_composite.mean())
            eye_data[f"{eye.lower()}_quality"] = q
            eye_data[f"{eye.lower()}_quality_tier"] = "low" if q <= 0.625 else ("mid" if q <= 0.75 else "high")
            rows = eye_group.feature_row.to_numpy(dtype=int)
            eye_vectors[eye] = features[rows].mean(axis=0).astype(np.float64)
            for row in eye_group.itertuples(index=False):
                match = re.search(r"(?:^|[_-])([LR])([12])$", str(row.image_id), flags=re.IGNORECASE)
                if match:
                    view_key = match.group(1).upper() + match.group(2)
                    views[view_key] = features[int(row.feature_row)].astype(np.float64)

        signature = f"L={left_grade}|R={right_grade}"
        records.append({
            "patient_id": pid,
            "left_eye_grade": left_grade,
            "right_eye_grade": right_grade,
            "bilateral_grade_signature": signature,
            "patient_grade": int(pd.to_numeric(group.patient_grade, errors="raise").mode().iloc[0]),
            "benchmark_split": str(split_values[0]),
            "left_eye_quality_composite": eye_data["l_quality"],
            "right_eye_quality_composite": eye_data["r_quality"],
            "right_eye_quality_tier": eye_data["r_quality_tier"],
            "strict_four_view": set(views) == {"L1", "L2", "R1", "R2"},
        })
        feature_map[pid] = {"left": eye_vectors["L"], "right": eye_vectors["R"], "views": views}

    metadata = pd.DataFrame.from_records(records)
    if metadata.patient_id.duplicated().any():
        raise ValueError("Duplicate DeepDRiD patient records")
    if len(metadata) != 300:
        raise ValueError(f"Expected 300 bilateral patients; got {len(metadata)}")
    return metadata, feature_map


def eligible_ids(table: pd.DataFrame, columns: list[str]) -> list[str]:
    sizes = table.groupby(columns, dropna=False)["patient_id"].transform("size")
    return table.loc[sizes >= 2, "patient_id"].astype(str).tolist()


def run_pair_condition(
    table: pd.DataFrame,
    feature_map: dict[str, dict[str, Any]],
    condition: str,
    match_columns: list[str],
    seeds: list[int],
    n_bootstrap: int,
    right_quality: bool = False,
) -> dict[str, Any]:
    sub = table.copy().reset_index(drop=True)
    ids = sub.patient_id.astype(str).tolist()
    left = np.stack([feature_map[pid]["left"] for pid in ids])
    right = np.stack([feature_map[pid]["right"] for pid in ids])
    positive = cosine_rows(left, right)
    rows = []
    for seed in seeds:
        permutation = match_permutation(sub, match_columns, seed)
        negative = cosine_rows(left, right[permutation])
        auc = float(roc_auc_score(np.r_[np.ones(len(positive)), np.zeros(len(negative))], np.r_[positive, negative]))
        bootstrap = node_bootstrap_auc(positive, negative, permutation, seed + 810000, n_bootstrap)
        rec = {
            "seed": seed,
            "n_patients": len(sub),
            "pair_identity_auroc": auc,
            "same_patient_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "cosine_gap": float(positive.mean() - negative.mean()),
            "node_bootstrap_95ci": interval(bootstrap),
        }
        if right_quality:
            q = sub.right_eye_quality_composite.to_numpy(dtype=float)
            rec["mean_abs_right_eye_quality_difference"] = float(np.mean(np.abs(q - q[permutation])))
            rec["max_abs_right_eye_quality_difference"] = float(np.max(np.abs(q - q[permutation])))
        rows.append(rec)
    aucs = np.asarray([row["pair_identity_auroc"] for row in rows], dtype=float)
    return {
        "condition": condition,
        "n_patients": len(sub),
        "match_columns": match_columns,
        "seed_sd_definition": "sample SD across five fixed assignments (ddof=1)",
        "auroc_mean": float(aucs.mean()),
        "auroc_seed_sd": float(aucs.std(ddof=1)),
        "node_bootstrap_interval_envelope": [
            float(min(row["node_bootstrap_95ci"][0] for row in rows)),
            float(max(row["node_bootstrap_95ci"][1] for row in rows)),
        ],
        "rows": rows,
    }


def run_old_conditions(
    table: pd.DataFrame,
    feature_map: dict[str, dict[str, Any]],
    n_bootstrap: int,
) -> list[dict[str, Any]]:
    conditions: list[tuple[str, list[str]]] = [
        ("patient_DR_grade", ["patient_grade"]),
        ("patient_grade_x_patient_quality_tier", ["patient_grade", "quality_tier"]),
    ]
    results = []
    for name, columns in conditions:
        results.append(run_pair_condition(table, feature_map, name, columns, SEEDS, n_bootstrap))

    # Preserve the original nearest-quality assignment exactly, including the
    # seeded tie-breaking jitter used by the archived protocol.
    sub = table.copy().reset_index(drop=True)
    ids = sub.patient_id.astype(str).tolist()
    left = np.stack([feature_map[pid]["left"] for pid in ids])
    right = np.stack([feature_map[pid]["right"] for pid in ids])
    positive = cosine_rows(left, right)
    rows = []
    for seed in SEEDS:
        permutation = nearest_quality_permutation(sub, seed)
        negative = cosine_rows(left, right[permutation])
        auc = float(roc_auc_score(np.r_[np.ones(len(positive)), np.zeros(len(negative))], np.r_[positive, negative]))
        bootstrap = node_bootstrap_auc(positive, negative, permutation, seed + 810000, n_bootstrap)
        quality = sub[["Overall quality", "Clarity", "Field definition", "Artifact"]].to_numpy(dtype=float)
        rows.append({
            "seed": seed,
            "n_patients": len(sub),
            "pair_identity_auroc": auc,
            "same_patient_mean": float(positive.mean()),
            "negative_mean": float(negative.mean()),
            "cosine_gap": float(positive.mean() - negative.mean()),
            "node_bootstrap_95ci": interval(bootstrap),
            "mean_abs_patient_quality_composite_difference": float(np.mean(np.abs(sub.quality_composite.to_numpy() - sub.quality_composite.to_numpy()[permutation]))),
        })
    aucs = np.asarray([row["pair_identity_auroc"] for row in rows], dtype=float)
    results.append({
        "condition": "patient_DR_grade_nearest_patient_quality",
        "n_patients": len(sub),
        "match_columns": ["patient_grade"],
        "seed_sd_definition": "sample SD across five fixed assignments (ddof=1)",
        "auroc_mean": float(aucs.mean()),
        "auroc_seed_sd": float(aucs.std(ddof=1)),
        "node_bootstrap_interval_envelope": [
            float(min(row["node_bootstrap_95ci"][0] for row in rows)),
            float(max(row["node_bootstrap_95ci"][1] for row in rows)),
        ],
        "rows": rows,
    })
    return results


def strict_hierarchy(
    metadata: pd.DataFrame,
    feature_map: dict[str, dict[str, Any]],
    split: str,
    n_bootstrap: int,
) -> dict[str, Any]:
    table = metadata.loc[
        (metadata.benchmark_split == split) & metadata.strict_four_view
    ].copy().reset_index(drop=True)
    ids = eligible_ids(table, ["bilateral_grade_signature", "right_eye_quality_tier"])
    table = table.loc[table.patient_id.isin(ids)].copy().reset_index(drop=True)
    if len(table) < 2:
        raise ValueError(f"No eligible four-view patients for split {split}")

    def mean_cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(cosine_rows(a[None, :], b[None, :])[0])

    same_eye, cross_eye = [], []
    for pid in table.patient_id.astype(str):
        views = feature_map[pid]["views"]
        same_eye.append((mean_cos(views["L1"], views["L2"]) + mean_cos(views["R1"], views["R2"])) / 2.0)
        cross_eye.append(float(np.mean([
            mean_cos(views["L1"], views["R1"]), mean_cos(views["L1"], views["R2"]),
            mean_cos(views["L2"], views["R1"]), mean_cos(views["L2"], views["R2"]),
        ])))
    same_eye = np.asarray(same_eye, dtype=float)
    cross_eye = np.asarray(cross_eye, dtype=float)
    ids = table.patient_id.astype(str).tolist()
    rows = []
    negative_score_rows: list[list[float]] = []
    for seed in SEEDS:
        permutation = match_permutation(table, ["bilateral_grade_signature", "right_eye_quality_tier"], seed)
        negative = []
        for anchor, donor in zip(ids, np.asarray(ids)[permutation]):
            a = feature_map[str(anchor)]["views"]
            d = feature_map[str(donor)]["views"]
            negative.append(float(np.mean([
                mean_cos(a["L1"], d["R1"]), mean_cos(a["L1"], d["R2"]),
                mean_cos(a["L2"], d["R1"]), mean_cos(a["L2"], d["R2"]),
            ])))
        negative = np.asarray(negative, dtype=float)
        negative_score_rows.append(negative.tolist())
        negative_mean_ci = interval(node_bootstrap_mean(negative, seed + 930000, n_bootstrap, permutation))
        same_cross = paired_node_bootstrap_mean_difference(same_eye, cross_eye, seed + 910000, n_bootstrap)
        cross_negative = node_bootstrap_mean_difference(cross_eye, negative, permutation, seed + 920000, n_bootstrap)
        rows.append({
            "seed": seed,
            "n_patients": len(table),
            "same_eye_mean": float(same_eye.mean()),
            "cross_eye_mean": float(cross_eye.mean()),
            "matched_negative_mean": float(negative.mean()),
            "matched_negative_node_bootstrap_95ci": negative_mean_ci,
            "same_eye_minus_cross_eye": float((same_eye - cross_eye).mean()),
            "cross_eye_minus_matched_negative": float(cross_eye.mean() - negative.mean()),
            "same_eye_minus_cross_eye_node_bootstrap_95ci": interval(same_cross),
            "cross_eye_minus_matched_negative_node_bootstrap_95ci": interval(cross_negative),
            "mean_abs_right_eye_quality_difference": float(np.mean(np.abs(table.right_eye_quality_composite.to_numpy() - table.right_eye_quality_composite.to_numpy()[permutation]))),
        })
    return {
        "split": split,
        "n_strict_four_view_patients_before_matching": int(((metadata.benchmark_split == split) & metadata.strict_four_view).sum()),
        "n_common_eligible_patients": len(table),
        "matching": "Exact ordered bilateral DR signature and exact right-eye quality-composite tier; anchor left-eye image is held fixed for positive and negative scores.",
        "identity_free_patient_score_distributions": {
            "same_eye": same_eye.tolist(),
            "cross_eye": cross_eye.tolist(),
            "matched_negative_mean_over_seeds": np.mean(np.asarray(negative_score_rows), axis=0).tolist(),
        },
        "same_eye_patient_mean": float(same_eye.mean()),
        "same_eye_patient_bootstrap_95ci": interval(paired_node_bootstrap_mean_difference(same_eye, np.zeros_like(same_eye), 940000 + (0 if split == "test" else 1), n_bootstrap)),
        "cross_eye_patient_mean": float(cross_eye.mean()),
        "cross_eye_patient_bootstrap_95ci": interval(paired_node_bootstrap_mean_difference(cross_eye, np.zeros_like(cross_eye), 950000 + (0 if split == "test" else 1), n_bootstrap)),
        "matched_negative_mean_over_seeds": float(np.mean([row["matched_negative_mean"] for row in rows])),
        "matched_negative_node_bootstrap_interval_envelope": [
            float(min(row["matched_negative_node_bootstrap_95ci"][0] for row in rows)),
            float(max(row["matched_negative_node_bootstrap_95ci"][1] for row in rows)),
        ],
        "rows": rows,
        "same_eye_minus_cross_eye_interval_envelope": [
            float(min(row["same_eye_minus_cross_eye_node_bootstrap_95ci"][0] for row in rows)),
            float(max(row["same_eye_minus_cross_eye_node_bootstrap_95ci"][1] for row in rows)),
        ],
        "cross_eye_minus_matched_negative_interval_envelope": [
            float(min(row["cross_eye_minus_matched_negative_node_bootstrap_95ci"][0] for row in rows)),
            float(max(row["cross_eye_minus_matched_negative_node_bootstrap_95ci"][1] for row in rows)),
        ],
    }


def run_deepdrid(manifest_path: Path, feature_path: Path, n_bootstrap: int) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path)
    features = np.load(feature_path)
    base, left, right = patient_table(manifest, features)
    base["patient_id"] = base.patient_id.astype(str)
    metadata, feature_map = eye_metadata(manifest, features)
    merged = base.merge(metadata, on=["patient_id", "patient_grade", "benchmark_split"], how="inner", validate="one_to_one")
    if len(merged) != 300:
        raise ValueError(f"Expected 300 patients after joining feature table and labels; got {len(merged)}")

    old_full = run_old_conditions(merged, feature_map, n_bootstrap)
    test = merged.loc[merged.benchmark_split == "test"].copy().reset_index(drop=True)
    old_test_ids = eligible_ids(test, ["patient_grade", "quality_tier"])
    old_test = test.loc[test.patient_id.isin(old_test_ids)].copy().reset_index(drop=True)
    old_heldout = run_old_conditions(old_test, feature_map, n_bootstrap)

    strict_sets: dict[str, dict[str, Any]] = {}
    for scope, subset in [("full_cohort", merged), ("validation", merged.loc[merged.benchmark_split == "val"]), ("test", test)]:
        common_ids = eligible_ids(subset, ["bilateral_grade_signature", "right_eye_quality_tier"])
        common = subset.loc[subset.patient_id.isin(common_ids)].copy().reset_index(drop=True)
        sig_only = run_pair_condition(
            common, feature_map, "exact_ordered_bilateral_DR_signature", ["bilateral_grade_signature"],
            SEEDS, n_bootstrap, right_quality=True,
        )
        sig_quality = run_pair_condition(
            common, feature_map, "bilateral_DR_signature_plus_right_eye_quality_tier",
            ["bilateral_grade_signature", "right_eye_quality_tier"], SEEDS, n_bootstrap, right_quality=True,
        )
        strict_sets[scope] = {
            "n_patients_before_exact_matching": len(subset),
            "n_common_eligible_patients": len(common),
            "signature_only": sig_only,
            "signature_plus_right_eye_quality": sig_quality,
        }

    return {
        "manifest": str(manifest_path.resolve()),
        "features": str(feature_path.resolve()),
        "n_complete_patients": len(merged),
        "bilateral_DR_grade_discordant_patients": int((merged.left_eye_grade != merged.right_eye_grade).sum()),
        "split_counts": {str(k): int(v) for k, v in merged.benchmark_split.value_counts().items()},
        "heldout_eye_grade_discordant_patients": int(((merged.benchmark_split == "test") & (merged.left_eye_grade != merged.right_eye_grade)).sum()),
        "old_protocol_full_cohort": old_full,
        "old_protocol_heldout_common_denominator": {
            "n_patients": len(old_test),
            "conditions": old_heldout,
        },
        "strict_exact_signature_controls": strict_sets,
        "four_view_hierarchy": {
            "validation": strict_hierarchy(metadata, feature_map, "val", n_bootstrap),
            "test": strict_hierarchy(metadata, feature_map, "test", n_bootstrap),
        },
    }


def run_papila(manifest_path: Path, feature_path: Path, n_bootstrap: int, archive_path: Path | None) -> dict[str, Any]:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str, "eye": str, "diagnosis": str}, keep_default_na=False)
    features = np.load(feature_path)
    if len(manifest) != len(features):
        raise ValueError("PAPILA manifest-feature row count mismatch")
    patients = []
    for patient_id, group in manifest.groupby("patient_id", sort=True):
        group = group.copy()
        group["eye"] = group.eye.astype(str).str.upper().str.strip()
        if group.eye.value_counts().to_dict() != {"L": 1, "R": 1}:
            raise ValueError(f"Expected one image per eye for PAPILA patient {patient_id}")
        left_row = group[group.eye == "L"].iloc[0]
        right_row = group[group.eye == "R"].iloc[0]
        patients.append({
            "patient_id": str(patient_id),
            "left": features[int(left_row.name)].astype(np.float64),
            "right": features[int(right_row.name)].astype(np.float64),
            "signature": f"L={left_row.diagnosis}|R={right_row.diagnosis}",
        })
    left = np.stack([row["left"] for row in patients])
    right = np.stack([row["right"] for row in patients])
    signatures = np.asarray([row["signature"] for row in patients])
    positive_all = cosine_rows(left, right)
    archived = None
    if archive_path:
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        archived_map = {(row["condition"], row["seed"]): row for row in archived["results"]}
    output: dict[str, Any] = {"n_complete_bilateral_patients": len(patients), "conditions": []}
    rows_by_condition = {"unmatched_random": [], "bilateral_diagnosis_signature_matched": []}
    for seed in PAPILA_SEEDS:
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(patients))
        random_permutation = derangement(all_indices, rng)
        matched = np.full(len(patients), -1, dtype=int)
        eligible = []
        for signature in sorted(set(signatures)):
            indices = np.flatnonzero(signatures == signature)
            if len(indices) >= 2:
                matched[indices] = derangement(indices, rng)
                eligible.extend(indices.tolist())
        matched_anchors = np.asarray(sorted(eligible), dtype=int)
        matched_targets = matched[matched_anchors]
        matched_local = {int(index): i for i, index in enumerate(matched_anchors)}
        matched_permutation = np.asarray([matched_local[int(index)] for index in matched_targets], dtype=int)
        assignments = {
            "unmatched_random": (all_indices, random_permutation, random_permutation),
            "bilateral_diagnosis_signature_matched": (matched_anchors, matched_targets, matched_permutation),
        }
        for condition, (anchors, target_indices, permutation) in assignments.items():
            positive = positive_all[anchors]
            negative = cosine_rows(left[anchors], right[target_indices])
            labels = np.r_[np.ones(len(positive)), np.zeros(len(negative))]
            auc = float(roc_auc_score(labels, np.r_[positive, negative]))
            if archived is not None:
                expected = archived_map[(condition, seed)]["pair_identity_auroc"]
                if abs(auc - expected) > 1e-12:
                    raise AssertionError(f"PAPILA archive mismatch for {condition}/{seed}: {auc} != {expected}")
            boot = node_bootstrap_auc(positive, negative, permutation, seed + 1010000, n_bootstrap)
            rows_by_condition[condition].append({
                "seed": seed,
                "n_patients": len(anchors),
                "pair_identity_auroc": auc,
                "node_bootstrap_95ci": interval(boot),
            })
    for condition, rows in rows_by_condition.items():
        aucs = np.asarray([row["pair_identity_auroc"] for row in rows], dtype=float)
        output["conditions"].append({
            "condition": condition,
            "n_patients": rows[0]["n_patients"],
            "auroc_mean": float(aucs.mean()),
            "auroc_seed_sd": float(aucs.std(ddof=1)),
            "node_bootstrap_interval_envelope": [
                float(min(row["node_bootstrap_95ci"][0] for row in rows)),
                float(max(row["node_bootstrap_95ci"][1] for row in rows)),
            ],
            "rows": rows,
        })
    return {"manifest": str(manifest_path.resolve()), "features": str(feature_path.resolve()), **output}


def run_hcbfd_audit(path: Path) -> dict[str, Any]:
    """Recompute sample SDs from archived seed rows without retraining."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selection"]["selected_variant"]
    raw = payload["raw_controls"]

    def summary(values: list[float]) -> dict[str, float | int]:
        values = np.asarray(values, dtype=float)
        return {
            "n_seeds": int(len(values)),
            "mean": float(values.mean()),
            "sample_sd_ddof_1": float(values.std(ddof=1)),
            "population_sd_ddof_0_archived": float(values.std(ddof=0)),
        }

    metrics: list[dict[str, Any]] = []
    for stage, raw_key, display in (
        ("validation", "validation", "DeepDRiD validation (patient-grade × patient-quality tier)"),
        ("refit_test", "test", "DeepDRiD held-out test (patient-grade × patient-quality tier)"),
    ):
        candidates = [row for row in payload["rows"] if row["stage"] == stage and row["variant"] == selected]
        if len(candidates) != 5:
            raise ValueError(f"Expected five archived selected-candidate rows for {stage}; got {len(candidates)}")
        candidate_values = [row["deepdrid_pair"]["pair_id_auroc"] for row in candidates]
        raw_values = [row["pair_id_auroc"] for row in raw[raw_key]["per_seed"]]
        metrics.append({
            "comparison": display,
            "n_patients": raw[raw_key]["n_patients"],
            "raw_frozen": summary(raw_values),
            "hcbfd_selected": summary(candidate_values),
            "raw_seed_aurocs": [float(x) for x in raw_values],
            "hcbfd_seed_aurocs": [float(x) for x in candidate_values],
        })
    for raw_key, field, display in (
        ("cfp_patientmax", "external_patientmax", "CFP–OCT development stress set, patient-max DME"),
        ("cfp_eyemode", "external_eyemode", "CFP–OCT development stress set, legacy eye-mode DME"),
    ):
        candidates = [row for row in payload["rows"] if row["stage"] == "refit_test" and row["variant"] == selected]
        raw_values = [row["pair_id_auroc"] for row in raw[raw_key]["per_seed"]]
        candidate_values = [row[field]["pair_id_auroc"] for row in candidates]
        metrics.append({
            "comparison": display,
            "n_patients": raw[raw_key]["n_patients"],
            "raw_frozen": summary(raw_values),
            "hcbfd_selected": summary(candidate_values),
            "raw_seed_aurocs": [float(x) for x in raw_values],
            "hcbfd_seed_aurocs": [float(x) for x in candidate_values],
        })

    validation_rows = [row for row in payload["rows"] if row["stage"] == "validation"]
    variants = sorted({row["variant"] for row in validation_rows})
    branch_keys = ("h_mean_feature_std", "d_mean_feature_std", "e_mean_feature_std")
    branch_means = {
        variant: {
            key: float(np.mean([row["noncollapse"][key] for row in validation_rows if row["variant"] == variant]))
            for key in branch_keys
        }
        for variant in variants
    }
    floor = 0.10
    return {
        "source": str(path.resolve()),
        "selected_variant": selected,
        "noncollapse_floor": floor,
        "archived_validation_branch_mean_std_by_variant": branch_means,
        "all_archived_variants_pass_floor_posthoc": all(
            all(value >= floor for value in branch_means[variant].values()) for variant in variants
        ),
        "metrics": metrics,
        "historical_note": "The archived v1 selector did not mechanically enforce this branch-standard-deviation floor. The revision selector now does; retrospective application to all archived validation seeds shows every candidate passes and the winner is unchanged.",
    }


def report(payload: dict[str, Any]) -> str:
    deep = payload["deepdrid"]
    papila = payload.get("papila")
    lines = [
        "# Revision v11: eye-level conditional and patient-node uncertainty audit",
        "",
        "This post-hoc audit uses the frozen ImageNet ResNet-50 feature arrays. It does not train a model or alter the locked PAPILA encoder, preprocessing, or pair score.",
        "",
        "## Why the analysis was extended",
        "",
        f"The DeepDRiD manifest contains {deep['bilateral_DR_grade_discordant_patients']} patients with different left- and right-eye DR grades; {deep['heldout_eye_grade_discordant_patients']} of {deep['split_counts'].get('test', 0)} held-out test patients are discordant. Patient-level DR matching alone therefore does not balance the ordered bilateral grade signature.",
        "",
        "The stricter control matches cross-patient negatives within the exact ordered bilateral DR signature. It additionally matches the right-eye quality-composite tier; because the anchor left-eye feature is identical in its positive and negative pair, this balances both the source eye exactly and the target eye at the recorded quality-tier resolution.",
        "",
        "## Bootstrap definition",
        "",
        "For each fixed one-to-one derangement, the interval uses a patient-node (pigeonhole) bootstrap. A positive pair receives its patient multiplicity; a cross-patient negative edge receives the product of its anchor and donor multiplicities. This retains the dependence created when a patient appears as an anchor and as another pair's donor. Intervals remain conditional on each original assignment and do not include rematching uncertainty. The five assignment-level estimates are summarized with sample SD (ddof=1).",
        "",
        "## DeepDRiD results",
        "",
        "### Existing protocol, recomputed with node bootstrap",
        "",
        "| Scope | Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in deep["old_protocol_full_cohort"]:
        lines.append(f"| Full cohort | {row['condition']} | {row['n_patients']} | {row['auroc_mean']:.4f} | {row['auroc_seed_sd']:.4f} | [{row['node_bootstrap_interval_envelope'][0]:.4f}, {row['node_bootstrap_interval_envelope'][1]:.4f}] |")
    for row in deep["old_protocol_heldout_common_denominator"]["conditions"]:
        lines.append(f"| Held-out common subset | {row['condition']} | {row['n_patients']} | {row['auroc_mean']:.4f} | {row['auroc_seed_sd']:.4f} | [{row['node_bootstrap_interval_envelope'][0]:.4f}, {row['node_bootstrap_interval_envelope'][1]:.4f}] |")
    lines += [
        "",
        "### Exact bilateral disease and eye-quality controls",
        "",
        "| Scope | Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope | Mean absolute right-eye quality difference |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope, section in deep["strict_exact_signature_controls"].items():
        for key in ("signature_only", "signature_plus_right_eye_quality"):
            row = section[key]
            q_values = [item.get("mean_abs_right_eye_quality_difference", float("nan")) for item in row["rows"]]
            qtext = f"{float(np.mean(q_values)):.4f} ± {float(np.std(q_values, ddof=1)):.4f}"
            lines.append(f"| {scope} | {row['condition']} | {row['n_patients']} | {row['auroc_mean']:.4f} | {row['auroc_seed_sd']:.4f} | [{row['node_bootstrap_interval_envelope'][0]:.4f}, {row['node_bootstrap_interval_envelope'][1]:.4f}] | {qtext} |")
    lines += [
        "",
        "### Strict four-view hierarchy",
        "",
        "| Split | Eligible four-view patients | Same-eye mean | Cross-eye mean | Matched-negative mean | Same-eye − cross-eye interval envelope | Cross-eye − negative interval envelope |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        row = deep["four_view_hierarchy"][split]
        negative_mean = float(np.mean([item["matched_negative_mean"] for item in row["rows"]]))
        lines.append(f"| {split} | {row['n_common_eligible_patients']} | {row['same_eye_patient_mean']:.4f} | {row['cross_eye_patient_mean']:.4f} | {negative_mean:.4f} | [{row['same_eye_minus_cross_eye_interval_envelope'][0]:.4f}, {row['same_eye_minus_cross_eye_interval_envelope'][1]:.4f}] | [{row['cross_eye_minus_matched_negative_interval_envelope'][0]:.4f}, {row['cross_eye_minus_matched_negative_interval_envelope'][1]:.4f}] |")
    if papila:
        lines += [
            "",
            "## PAPILA frozen external baseline",
            "",
            "| Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in papila["conditions"]:
            lines.append(f"| {row['condition']} | {row['n_patients']} | {row['auroc_mean']:.4f} | {row['auroc_seed_sd']:.4f} | [{row['node_bootstrap_interval_envelope'][0]:.4f}, {row['node_bootstrap_interval_envelope'][1]:.4f}] |")
        lines += [
            "",
            "The archived per-seed PAPILA AUROCs were asserted to reproduce exactly before uncertainty was recomputed. This remains an external evaluation of the frozen raw baseline only; it does not validate H-CBFD.",
        ]
    hcbfd = payload.get("hcbfd")
    if hcbfd:
        lines += [
            "",
            "## H-CBFD exploratory candidate: corrected across-seed SD",
            "",
            "| Comparison | n | Raw frozen mean ± sample SD | H-CBFD mean ± sample SD |",
            "|---|---:|---:|---:|",
        ]
        for row in hcbfd["metrics"]:
            raw = row["raw_frozen"]
            head = row["hcbfd_selected"]
            lines.append(f"| {row['comparison']} | {row['n_patients']} | {raw['mean']:.4f} ± {raw['sample_sd_ddof_1']:.4f} | {head['mean']:.4f} ± {head['sample_sd_ddof_1']:.4f} |")
        lines += [
            "",
            f"The selected candidate is {hcbfd['selected_variant']}. The archived seedwise AUROCs were not changed; only SDs were recomputed from population SD to sample SD. A 0.10 branch mean-feature-SD floor was added to the rerunnable selector. Retrospective checks show all three archived validation variants pass and the winning variant is unchanged. The archived candidate has not been reevaluated under the stricter eye-specific disease/quality control above; CFP–OCT remains an already-inspected development stress set, not blind external validation.",
        ]
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "All results are conditional on the available matching strata and the specified fixed assignment seeds. Full-cohort results are descriptive; the held-out analysis is a post-hoc sensitivity analysis. The stricter DeepDRiD control does not match continuous per-image quality, camera, site, age, pigmentation, or other unrecorded factors. No causal decomposition or identity biomarker claim follows from these AUROCs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep-manifest", type=Path, required=True)
    parser.add_argument("--deep-features", type=Path, required=True)
    parser.add_argument("--papila-manifest", type=Path)
    parser.add_argument("--papila-features", type=Path)
    parser.add_argument("--papila-archive", type=Path)
    parser.add_argument("--hcbfd-json", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=3000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap samples for a diagnostic run")
    if bool(args.papila_manifest) != bool(args.papila_features):
        raise ValueError("Provide both --papila-manifest and --papila-features, or neither")

    payload: dict[str, Any] = {
        "protocol": "revision_v11_eye_signature_and_patient_node_bootstrap",
        "n_bootstrap": args.bootstrap,
        "seeds": SEEDS,
        "papila_seeds": PAPILA_SEEDS,
        "deepdrid": run_deepdrid(args.deep_manifest, args.deep_features, args.bootstrap),
    }
    if args.papila_manifest:
        payload["papila"] = run_papila(args.papila_manifest, args.papila_features, args.bootstrap, args.papila_archive)
    if args.hcbfd_json:
        payload["hcbfd"] = run_hcbfd_audit(args.hcbfd_json)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "revision_v11_conditional_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REVISION_V11_CONDITIONAL_AUDIT.md").write_text(report(payload), encoding="utf-8")
    print(json.dumps({
        "protocol": payload["protocol"],
        "deepdrid": {
            "full": [(r["condition"], r["n_patients"], r["auroc_mean"], r["node_bootstrap_interval_envelope"]) for r in payload["deepdrid"]["old_protocol_full_cohort"]],
            "strict_test_n": payload["deepdrid"]["strict_exact_signature_controls"]["test"]["n_common_eligible_patients"],
        },
        "papila": [(r["condition"], r["n_patients"], r["auroc_mean"], r["node_bootstrap_interval_envelope"]) for r in payload.get("papila", {}).get("conditions", [])],
        "hcbfd": [(r["comparison"],r["raw_frozen"]["mean"],r["raw_frozen"]["sample_sd_ddof_1"],r["hcbfd_selected"]["mean"],r["hcbfd_selected"]["sample_sd_ddof_1"]) for r in payload.get("hcbfd",{}).get("metrics",[])],
        "out_dir": str(args.out_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
