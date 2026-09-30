#!/usr/bin/env python3
"""Post-hoc held-out and PAPILA uncertainty audits under frozen v6 protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from conditional_shared_signal_controls import (
    bootstrap_pair_metrics,
    cosine_rows,
    match_permutation,
    metrics,
    nearest_quality_permutation,
    patient_table,
)
from papila_frozen_external_probe import _cosine, _derangement


ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "audit_outputs/phase2_public"
SEEDS = list(range(20260922, 20260927))
PAPILA_SEEDS = list(range(20260921, 20260926))
N_BOOT = 3000


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        vals = np.asarray([r["pair_identity_auroc"] for r in subset])
        out.append({
            "condition": condition,
            "n_patients": subset[0]["n_patients"],
            "auroc_mean": float(vals.mean()),
            "auroc_seed_sd": float(vals.std(ddof=1)),
            "conditional_bootstrap_interval_envelope": [
                float(min(r["bootstrap_auroc_95ci"][0] for r in subset)),
                float(max(r["bootstrap_auroc_95ci"][1] for r in subset)),
            ],
        })
    return out


def deepdrid() -> dict:
    manifest = pd.read_csv(AUDIT / "splits_stratified/DeepDRiD_patient_disjoint_feature_manifest.csv")
    features = np.load(AUDIT / "deepdrid_resnet50_features_full.npy")
    if len(manifest) != len(features):
        raise AssertionError("DeepDRiD manifest-feature length mismatch")
    table, left, right = patient_table(manifest, features)
    original_n = int((table.benchmark_split == "test").sum())
    indices = np.flatnonzero(table.benchmark_split.to_numpy() == "test")
    test_table = table.iloc[indices].copy().reset_index(drop=True)
    sizes = test_table.groupby(["patient_grade", "quality_tier"], observed=False)["patient_id"].transform("size")
    keep = np.flatnonzero(sizes.to_numpy() >= 2)
    excluded = test_table.iloc[np.flatnonzero(sizes.to_numpy() < 2)]["patient_id"].tolist()
    eligible = test_table.iloc[keep].copy().reset_index(drop=True)
    left = left[indices][keep]
    right = right[indices][keep]
    positive = cosine_rows(left, right)
    if original_n != 44 or len(eligible) != 42 or len(excluded) != 2:
        raise AssertionError(f"Unexpected eligible test count: {original_n}/{len(eligible)}")
    conditions = {
        "grade_matched": lambda seed: match_permutation(eligible, ["patient_grade"], seed),
        "grade_quality_tier_matched": lambda seed: match_permutation(eligible, ["patient_grade", "quality_tier"], seed),
        "grade_nearest_quality": lambda seed: nearest_quality_permutation(eligible, seed),
    }
    rows = []
    for condition, assignment in conditions.items():
        for seed in SEEDS:
            perm = assignment(seed)
            negative = cosine_rows(left, right[perm])
            record = {
                "condition": condition,
                "seed": seed,
                "n_patients": len(eligible),
                **metrics(positive, negative),
            }
            boot = bootstrap_pair_metrics(positive, negative, np.random.default_rng(seed + 600000), N_BOOT)
            record["bootstrap_auroc_95ci"] = boot["bootstrap_pair_id_auroc_95ci"]
            rows.append(record)
    return {
        "source": "archived frozen ResNet-50 features and patient-disjoint manifest",
        "population": "held-out test, common eligible subset for all three conditions",
        "original_test_patients": original_n,
        "eligible_test_patients": len(eligible),
        "excluded_singleton_stratum_patients": excluded,
        "seeds": SEEDS,
        "n_bootstrap_per_seed": N_BOOT,
        "rows": rows,
        "aggregate": summarize(rows),
    }


def papila() -> dict:
    manifest = pd.read_csv(AUDIT / "papila/papila_external_frozen_manifest.csv", dtype=str, keep_default_na=False)
    features = np.load(AUDIT / "papila/papila_resnet50_features.npy")
    archived = json.loads((AUDIT / "papila/papila_frozen_external_probe_v1.json").read_text())
    if len(manifest) != 488 or len(features) != 488:
        raise AssertionError("Unexpected PAPILA image count")
    manifest["feature_row"] = np.arange(len(manifest))
    patients = []
    for pid, group in manifest.groupby("patient_id", sort=True):
        group = group.copy()
        group["eye"] = group["eye"].str.upper().str.strip()
        if group.eye.value_counts().to_dict() != {"L": 1, "R": 1}:
            raise AssertionError(f"PAPILA laterality error for {pid}")
        l = group[group.eye == "L"].iloc[0]
        r = group[group.eye == "R"].iloc[0]
        patients.append((
            features[int(l.feature_row)], features[int(r.feature_row)],
            f"L={l.diagnosis}|R={r.diagnosis}",
        ))
    left = np.stack([x[0] for x in patients])
    right = np.stack([x[1] for x in patients])
    signature = np.asarray([x[2] for x in patients])
    pos_all = _cosine(left, right)
    archive_map = {(row["condition"], row["seed"]): row for row in archived["results"]}
    rows = []
    for seed in PAPILA_SEEDS:
        rng = np.random.default_rng(seed)
        all_idx = np.arange(len(patients))
        random_perm = _derangement(all_idx, rng)
        assignments = [("unmatched_random", all_idx, random_perm)]
        matched_perm = np.full(len(patients), -1, dtype=int)
        matched_idx = []
        for value in sorted(set(signature)):
            idx = np.flatnonzero(signature == value)
            if len(idx) >= 2:
                matched_perm[idx] = _derangement(idx, rng)
                matched_idx.extend(idx.tolist())
        idx = np.asarray(sorted(matched_idx), dtype=int)
        assignments.append(("bilateral_diagnosis_signature_matched", idx, matched_perm[idx]))
        for condition, anchors, targets in assignments:
            positive = pos_all[anchors]
            negative = _cosine(left[anchors], right[targets])
            actual = metrics(positive, negative)["pair_identity_auroc"]
            expected = archive_map[(condition, seed)]["pair_identity_auroc"]
            if abs(actual - expected) > 1e-12:
                raise AssertionError(f"Archived PAPILA result not reproduced: {condition}/{seed}: {actual} vs {expected}")
            boot = bootstrap_pair_metrics(positive, negative, np.random.default_rng(seed + 700000), N_BOOT)
            rows.append({
                "condition": condition,
                "seed": seed,
                "n_patients": len(anchors),
                "pair_identity_auroc": actual,
                "bootstrap_auroc_95ci": boot["bootstrap_pair_id_auroc_95ci"],
            })
    return {
        "source": "locked PAPILA frozen external baseline; archived AUROC checked for every seed",
        "seeds": PAPILA_SEEDS,
        "n_bootstrap_per_seed": N_BOOT,
        "uncertainty_scope": "paired-patient bootstrap conditional on each original fixed negative assignment; envelope is not a pooled 95% CI and does not represent rematching uncertainty",
        "rows": rows,
        "aggregate": summarize(rows),
    }


def main() -> None:
    payload = {
        "protocol": "revision_v6_posthoc_uncertainty_audit",
        "deepdrid": deepdrid(),
        "papila": papila(),
    }
    out = AUDIT / "revision_v6_uncertainty_audit.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key]["aggregate"] for key in ("deepdrid", "papila")}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
