#!/usr/bin/env python3
"""Run the pre-specified PAPILA final-external pair-correspondence probe.

This script assumes that ``normalize_papila_official.py`` has already produced
the manifest and that image features were extracted once with the frozen
ImageNet ResNet-50 baseline.  It trains nothing and has no tunable parameters.
PAPILA must not be used to choose a feature extractor, pooling rule, threshold,
or negative-pair rule before running this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def _derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) < 2:
        raise ValueError("A matching stratum needs at least two patients")
    for _ in range(100):
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate
    return np.roll(indices, 1)


def _result(condition: str, positive: np.ndarray, negative: np.ndarray, seed: int, eligible: int) -> dict[str, float | int | str]:
    scores = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    return {
        "condition": condition,
        "seed": seed,
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "n_eligible_patients": eligible,
        "same_patient_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "mean_gap": float(positive.mean() - negative.mean()),
        "pair_identity_auroc": float(roc_auc_score(labels, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260921, 20260922, 20260923, 20260924, 20260925])
    args = parser.parse_args()

    manifest = pd.read_csv(
        args.manifest,
        dtype={"patient_id": str, "eye": str, "diagnosis": str},
        keep_default_na=False,
    )
    features = np.load(args.features)
    required = {"patient_id", "eye", "diagnosis", "image_id", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Missing canonical PAPILA fields: {missing}")
    if len(manifest) != len(features):
        raise ValueError(f"Feature rows {len(features)} != manifest rows {len(manifest)}")
    if not bool(manifest["image_path"].map(lambda value: Path(value).is_file()).all()):
        raise FileNotFoundError("The frozen external manifest contains a missing image path")

    manifest = manifest.copy()
    manifest["eye"] = manifest["eye"].str.upper().str.strip()
    if set(manifest["eye"]) != {"L", "R"}:
        raise ValueError("PAPILA manifest must contain only explicit L/R rows")
    manifest["feature_row"] = np.arange(len(manifest))

    patient_rows: list[dict[str, object]] = []
    for patient, group in manifest.groupby("patient_id", sort=True):
        eye_counts = group["eye"].value_counts().to_dict()
        if eye_counts != {"L": 1, "R": 1}:
            raise ValueError(f"Expected exactly one L and one R image for PAPILA patient {patient}; got {eye_counts}")
        left_row = group[group["eye"] == "L"].iloc[0]
        right_row = group[group["eye"] == "R"].iloc[0]
        if not left_row.diagnosis or not right_row.diagnosis:
            raise ValueError(f"Missing diagnosis for PAPILA patient {patient}")
        patient_rows.append({
            "patient_id": patient,
            "left": features[int(left_row.feature_row)],
            "right": features[int(right_row.feature_row)],
            # Ordered bilateral signature prevents a disease-label shortcut in the sensitivity control.
            "diagnosis_signature": f"L={left_row.diagnosis}|R={right_row.diagnosis}",
        })
    if len(patient_rows) < 2:
        raise ValueError("Need at least two complete PAPILA patients")

    left = np.stack([row["left"] for row in patient_rows])
    right = np.stack([row["right"] for row in patient_rows])
    signature = np.asarray([row["diagnosis_signature"] for row in patient_rows])
    positive_all = _cosine(left, right)
    results: list[dict[str, float | int | str]] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        all_indices = np.arange(len(patient_rows))
        random_permutation = _derangement(all_indices, rng)
        results.append(_result("unmatched_random", positive_all, _cosine(left, right[random_permutation]), seed, len(patient_rows)))

        matched_indices: list[int] = []
        matched_permutation = np.full(len(patient_rows), -1, dtype=int)
        for value in sorted(set(signature)):
            indices = np.flatnonzero(signature == value)
            if len(indices) >= 2:
                matched_permutation[indices] = _derangement(indices, rng)
                matched_indices.extend(indices.tolist())
        if matched_indices:
            indices = np.asarray(sorted(matched_indices), dtype=int)
            results.append(_result(
                "bilateral_diagnosis_signature_matched",
                positive_all[indices],
                _cosine(left[indices], right[matched_permutation[indices]]),
                seed,
                len(indices),
            ))

    frame = pd.DataFrame(results)
    summary = {
        "protocol": "papila_frozen_external_pair_correspondence_v1",
        "features": str(args.features.resolve()),
        "manifest": str(args.manifest.resolve()),
        "n_feature_rows": int(len(manifest)),
        "n_complete_bilateral_patients": len(patient_rows),
        "seeds": args.seeds,
        "results": results,
        "aggregate": frame.groupby("condition")["pair_identity_auroc"].agg(["mean", "std", "min", "max"]).reset_index().to_dict(orient="records"),
        "external_lock": "PAPILA labels and scores were not used for architecture, threshold, preprocessing, or negative-pair selection. This script performs no fitting.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame.to_csv(args.out.with_suffix(".csv"), index=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
