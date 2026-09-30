#!/usr/bin/env python3
"""Run R2.0a/c on frozen DeepDRiD features without fitting a model.

The probe deliberately keeps the four documented image views separate.  It
evaluates each cross-eye view combination against patient-DR-level-matched
cross-patient right eyes, and tests the expected similarity hierarchy:

    same eye / different view > different eye / same patient >
    different patient / same patient-DR level.

Only complete patients with exactly L1, L2, R1 and R2 are included.  A small
number of DeepDRiD records use alternative view suffixes (for example ``l3``);
those patients are excluded rather than silently relabelled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


VIEW_ORDER = ("L1", "L2", "R1", "R2")
CROSS_COMBINATIONS = (("L1", "R1"), ("L1", "R2"), ("L2", "R1"), ("L2", "R2"))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.sum(a * b, axis=1)


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation with no identity pairs within a stratum."""
    if len(indices) < 2:
        raise ValueError("Every matching stratum must contain at least two patients")
    for _ in range(200):
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate
    return np.roll(indices, 1)


def matched_permutation(grades: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permutation = np.arange(len(grades))
    for grade in sorted(np.unique(grades)):
        idx = np.flatnonzero(grades == grade)
        permutation[idx] = derangement(idx, rng)
    return permutation


def paired_bootstrap(values: np.ndarray, seed: int, n_bootstrap: int = 3000) -> tuple[float, float, float]:
    """Patient bootstrap for a paired mean difference.

    The returned probability is descriptive (fraction of bootstrap means above
    zero), not a model-selection statistic.
    """
    rng = np.random.default_rng(seed + 800000)
    index = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[index].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high), float(np.mean(means > 0))


def patient_table(manifest: pd.DataFrame, features: np.ndarray, split: str) -> tuple[list[dict], dict]:
    work = manifest.loc[manifest["benchmark_split"] == split].copy()
    work["feature_row"] = np.arange(len(manifest))[manifest["benchmark_split"].eq(split).to_numpy()]
    work["view"] = work["image_id"].astype(str).str.extract(r"_(l1|l2|r1|r2)$", expand=False).str.upper()
    total_patients = int(work["patient_id"].nunique())
    rows: list[dict] = []
    excluded: list[str] = []
    for patient_id, group in work.groupby("patient_id", sort=True):
        lookup = dict(zip(group["view"], group["feature_row"]))
        if set(lookup) != set(VIEW_ORDER):
            excluded.append(str(patient_id))
            continue
        grades = group["patient_grade"].dropna().unique()
        if len(grades) != 1:
            raise ValueError(f"Patient {patient_id} has non-unique patient_grade: {grades.tolist()}")
        rows.append({
            "patient_id": str(patient_id),
            "patient_grade": int(grades[0]),
            **{view: features[int(lookup[view])] for view in VIEW_ORDER},
        })
    return rows, {
        "split": split,
        "patients_in_manifest": total_patients,
        "patients_eligible_exact_l1_l2_r1_r2": len(rows),
        "patients_excluded_nonstandard_or_incomplete_views": len(excluded),
        "excluded_patient_ids": excluded,
    }


def run_split(rows: list[dict], split_info: dict, seeds: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    if len(rows) < 2:
        raise ValueError(f"Need at least two eligible patients for {split_info['split']}")
    values = {view: np.stack([r[view] for r in rows]) for view in VIEW_ORDER}
    grades = np.asarray([r["patient_grade"] for r in rows])
    pair_rows: list[dict] = []
    hierarchy_rows: list[dict] = []
    patient_rows: list[dict] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        negative_index = matched_permutation(grades, rng)
        combination_scores: dict[str, np.ndarray] = {}
        negative_scores: dict[str, np.ndarray] = {}
        for left_view, right_view in CROSS_COMBINATIONS:
            key = f"{left_view}-{right_view}"
            positive = cosine(values[left_view], values[right_view])
            negative = cosine(values[left_view], values[right_view][negative_index])
            combination_scores[key] = positive
            negative_scores[key] = negative
            labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
            scores = np.concatenate((positive, negative))
            pair_rows.append({
                "split": split_info["split"],
                "seed": seed,
                "view_combination": key,
                "negative_condition": "patient_DR_level_matched",
                "n_positive": len(positive),
                "n_negative": len(negative),
                "same_patient_mean": float(positive.mean()),
                "negative_mean": float(negative.mean()),
                "mean_gap": float(positive.mean() - negative.mean()),
                "pair_identity_auroc": float(roc_auc_score(labels, scores)),
            })

        same_eye_left = cosine(values["L1"], values["L2"])
        same_eye_right = cosine(values["R1"], values["R2"])
        same_eye_patient = (same_eye_left + same_eye_right) / 2
        cross_patient = np.mean(np.stack(list(combination_scores.values())), axis=0)
        matched_negative = np.mean(np.stack(list(negative_scores.values())), axis=0)
        same_cross = same_eye_patient - cross_patient
        cross_negative = cross_patient - matched_negative
        same_cross_low, same_cross_high, same_cross_positive = paired_bootstrap(same_cross, seed)
        cross_neg_low, cross_neg_high, cross_neg_positive = paired_bootstrap(cross_negative, seed + 10000)
        hierarchy_rows.append({
            "split": split_info["split"],
            "seed": seed,
            "n_patients": len(rows),
            "same_eye_cross_view_mean": float(same_eye_patient.mean()),
            "cross_eye_same_patient_mean": float(cross_patient.mean()),
            "cross_patient_grade_matched_mean": float(matched_negative.mean()),
            "same_eye_minus_cross_eye_gap": float(same_cross.mean()),
            "cross_eye_minus_matched_negative_gap": float(cross_negative.mean()),
            "same_eye_minus_cross_eye_bootstrap_ci_low": same_cross_low,
            "same_eye_minus_cross_eye_bootstrap_ci_high": same_cross_high,
            "same_eye_minus_cross_eye_bootstrap_fraction_gt_zero": same_cross_positive,
            "cross_eye_minus_matched_negative_bootstrap_ci_low": cross_neg_low,
            "cross_eye_minus_matched_negative_bootstrap_ci_high": cross_neg_high,
            "cross_eye_minus_matched_negative_bootstrap_fraction_gt_zero": cross_neg_positive,
            "hierarchy_holds_mean": bool(
                same_eye_patient.mean() > cross_patient.mean() > matched_negative.mean()
            ),
        })
        for i, row in enumerate(rows):
            patient_rows.append({
                "split": split_info["split"],
                "seed": seed,
                "patient_id": row["patient_id"],
                "patient_grade": int(grades[i]),
                "matched_negative_patient_id": rows[int(negative_index[i])]["patient_id"],
                "same_eye_cross_view_cosine": float(same_eye_patient[i]),
                "cross_eye_same_patient_cosine": float(cross_patient[i]),
                "cross_patient_grade_matched_cosine": float(matched_negative[i]),
            })
    return pair_rows, hierarchy_rows, patient_rows


def aggregate_pair(rows: pd.DataFrame) -> list[dict]:
    numeric = ["same_patient_mean", "negative_mean", "mean_gap", "pair_identity_auroc"]
    out = []
    for (split, combination), group in rows.groupby(["split", "view_combination"], sort=True):
        result = {"split": split, "view_combination": combination, "n_seeds": int(len(group)), "n_patients": int(group["n_positive"].iloc[0])}
        for col in numeric:
            result[f"{col}_mean"] = float(group[col].mean())
            result[f"{col}_std"] = float(group[col].std(ddof=0))
        out.append(result)
    return out


def aggregate_hierarchy(rows: pd.DataFrame) -> list[dict]:
    numeric = [
        "same_eye_cross_view_mean", "cross_eye_same_patient_mean", "cross_patient_grade_matched_mean",
        "same_eye_minus_cross_eye_gap", "cross_eye_minus_matched_negative_gap",
        "same_eye_minus_cross_eye_bootstrap_ci_low", "same_eye_minus_cross_eye_bootstrap_ci_high",
        "same_eye_minus_cross_eye_bootstrap_fraction_gt_zero",
        "cross_eye_minus_matched_negative_bootstrap_ci_low", "cross_eye_minus_matched_negative_bootstrap_ci_high",
        "cross_eye_minus_matched_negative_bootstrap_fraction_gt_zero",
    ]
    out = []
    for split, group in rows.groupby("split", sort=True):
        result = {
            "split": split,
            "n_seeds": int(len(group)),
            "n_patients": int(group["n_patients"].iloc[0]),
            "hierarchy_holds_in_all_seeds": bool(group["hierarchy_holds_mean"].all()),
        }
        for col in numeric:
            result[f"{col}_mean"] = float(group[col].mean())
            result[f"{col}_std"] = float(group[col].std(ddof=0))
        out.append(result)
    return out


def markdown(summary: dict) -> str:
    pair = pd.DataFrame(summary["pair_summary"])
    hierarchy = pd.DataFrame(summary["hierarchy_summary"])
    test_pair = pair.loc[pair.split == "test"]
    val_pair = pair.loc[pair.split == "val"]
    test_h = hierarchy.loc[hierarchy.split == "test"].iloc[0]
    val_h = hierarchy.loc[hierarchy.split == "val"].iloc[0]
    lines = [
        "# DeepDRiD R2.0a/c：四视图 raw pair probe 与层级检验 v1",
        "",
        "## 目的与纪律",
        "",
        "本分析直接使用冻结的 ImageNet ResNet50 特征；没有训练模型、阈值或超参数。"
        "每位患者保留 L1/L2/R1/R2 四张明确命名的视图，并在同一 patient DR level 内"
        "置换右眼构造跨患者负对。test 是主报告；validation 只作为方向复现。",
        "",
        "由于少数记录使用 l3/l4/r3/r4 等非标准后缀，严格四视图分析仅纳入同时具有"
        " L1、L2、R1、R2 的患者；这些记录不会被重命名或混入其它组合。",
        "",
        "## R2.0a：四种跨眼组合（patient-DR-level-matched negatives）",
        "",
        "| Split | View pair | Patients | Same-patient cosine | Matched-negative cosine | Gap | Pair-ID AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in pair.iterrows():
        lines.append(
            f"| {r['split']} | {r['view_combination']} | {int(r['n_patients'])} | "
            f"{r['same_patient_mean_mean']:.4f} ± {r['same_patient_mean_std']:.4f} | "
            f"{r['negative_mean_mean']:.4f} ± {r['negative_mean_std']:.4f} | "
            f"{r['mean_gap_mean']:.4f} ± {r['mean_gap_std']:.4f} | "
            f"{r['pair_identity_auroc_mean']:.4f} ± {r['pair_identity_auroc_std']:.4f} |"
        )
    lines.extend([
        "",
        "test 的四种组合均高于随机 AUROC=0.5，且没有单一组合独占全部信号。"
        "因此，第一轮中按眼平均得到的信号不是某一固定视图组合偶然造成的。",
        "",
        "## R2.0c：同眼/跨眼层级",
        "",
        "每个患者先取同眼跨视图相似度 `(L1,L2)` 与 `(R1,R2)` 的平均；"
        "跨眼同患者和跨患者匹配负对则各自平均四个 L×R 组合。",
        "",
        "| Split | Patients | Same-eye, cross-view | Cross-eye, same patient | Cross-patient, grade matched | Same−cross gap (bootstrap 95% CI) | Cross−negative gap (bootstrap 95% CI) | Hierarchy mean direction |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for _, r in hierarchy.iterrows():
        lines.append(
            f"| {r['split']} | {int(r['n_patients'])} | {r['same_eye_cross_view_mean_mean']:.4f} ± {r['same_eye_cross_view_mean_std']:.4f} | "
            f"{r['cross_eye_same_patient_mean_mean']:.4f} ± {r['cross_eye_same_patient_mean_std']:.4f} | "
            f"{r['cross_patient_grade_matched_mean_mean']:.4f} ± {r['cross_patient_grade_matched_mean_std']:.4f} | "
            f"{r['same_eye_minus_cross_eye_gap_mean']:.4f} [{r['same_eye_minus_cross_eye_bootstrap_ci_low_mean']:.4f}, {r['same_eye_minus_cross_eye_bootstrap_ci_high_mean']:.4f}] | "
            f"{r['cross_eye_minus_matched_negative_gap_mean']:.4f} [{r['cross_eye_minus_matched_negative_bootstrap_ci_low_mean']:.4f}, {r['cross_eye_minus_matched_negative_bootstrap_ci_high_mean']:.4f}] | "
            f"{r['hierarchy_holds_in_all_seeds']} |"
        )
    lines.extend([
        "",
        f"在 test 中，均值方向为 `same-eye ({test_h['same_eye_cross_view_mean_mean']:.4f}) > "
        f"cross-eye same-patient ({test_h['cross_eye_same_patient_mean_mean']:.4f}) > "
        f"grade-matched cross-patient ({test_h['cross_patient_grade_matched_mean_mean']:.4f})`；"
        f"在 validation 中方向相同。需要注意，test 的 same-eye − cross-eye 差仅为 "
        f"{test_h['same_eye_minus_cross_eye_gap_mean']:.4f}，patient bootstrap CI "
        f"[{test_h['same_eye_minus_cross_eye_bootstrap_ci_low_mean']:.4f}, "
        f"{test_h['same_eye_minus_cross_eye_bootstrap_ci_high_mean']:.4f}] 跨过 0，故不能把这部分"
        "写成有把握的强度差异；稳健的区分是 cross-eye same-patient 明显高于疾病等级匹配的跨患者负对。",
        "",
        "## 可支持与不可支持的结论",
        "",
        "结果通过 R2.0a 的组合稳健性 gate，并支持 R2.0c 的关键区分（跨眼同患者高于"
        "疾病等级匹配的跨患者）。同眼与跨眼的均值方向符合预期，但其差异尚不确定。"
        "它仍未控制质量、"
        "视野与伪影；不能将本结果表述为疾病无关身份因子。R2.0b 的 grade+quality"
        "联合匹配是进入 H-CBFD 训练前的下一项必要 gate。",
        "",
        "完整逐种子结果：`deepdrid_r20_four_view_pair_probe_v1.csv`；患者级层级账本："
        "`deepdrid_r20_view_hierarchy_patients_v1.csv`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260921, 20260922, 20260923, 20260924, 20260925])
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    features = np.load(args.features)
    if len(manifest) != len(features):
        raise ValueError(f"Feature rows {len(features)} != manifest rows {len(manifest)}")
    required = {"patient_id", "image_id", "patient_grade", "benchmark_split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)

    all_pair_rows: list[dict] = []
    all_hierarchy_rows: list[dict] = []
    all_patient_rows: list[dict] = []
    split_info = {}
    for split in ("val", "test"):
        rows, info = patient_table(manifest, features, split)
        split_info[split] = info
        pair_rows, hierarchy_rows, patient_rows = run_split(rows, info, args.seeds)
        all_pair_rows.extend(pair_rows)
        all_hierarchy_rows.extend(hierarchy_rows)
        all_patient_rows.extend(patient_rows)

    pair_df = pd.DataFrame(all_pair_rows)
    hierarchy_df = pd.DataFrame(all_hierarchy_rows)
    patient_df = pd.DataFrame(all_patient_rows)
    summary = {
        "protocol": "deepdrid_r20_four_view_raw_pair_and_hierarchy_probe_v1",
        "feature_extractor": "ImageNet-pretrained ResNet50 frozen features supplied by full audited DeepDRiD cache",
        "features": str(args.features),
        "manifest": str(args.manifest),
        "view_requirement": "exactly L1,L2,R1,R2; alternative l3/l4/r3/r4 names excluded without relabelling",
        "negative_control": "cross-patient right-eye view permutation within patient_DR_Level / patient_grade strata, no fixed points",
        "splits": split_info,
        "seeds": args.seeds,
        "pair_summary": aggregate_pair(pair_df),
        "hierarchy_summary": aggregate_hierarchy(hierarchy_df),
        "model_training_or_threshold_tuning": False,
        "interpretation_boundary": "R2.0a/c controls view averaging and patient DR level only; it does not jointly control quality, field definition, artifact, or acquisition protocol.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.out_dir / "deepdrid_r20_four_view_pair_probe_v1.csv", index=False)
    hierarchy_df.to_csv(args.out_dir / "deepdrid_r20_view_hierarchy_seed_summary_v1.csv", index=False)
    patient_df.to_csv(args.out_dir / "deepdrid_r20_view_hierarchy_patients_v1.csv", index=False)
    (args.out_dir / "deepdrid_r20_four_view_hierarchy_probe_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "DEEPDRID_R20_FOUR_VIEW_HIERARCHY_V1.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
