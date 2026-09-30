#!/usr/bin/env python3
"""R2.0b/R2.0d conditional controls for DeepDRiD frozen features.

This script deliberately does *not* train a new bilateral model.  It measures
whether the raw-image representation separates same-patient cross-eye pairs
after negative pairs are constrained to have both the same patient DR grade and
a coarse, auditable image-quality stratum.  It also records simple linear
metadata probes on the existing patient-disjoint split.

Quality coarsening is specified before running the comparison:
  q = mean(OverallQuality, Clarity/10, FieldDefinition/10, 1-Artifact/10)
  low: q <= .625, mid: .625 < q <= .75, high: q > .75.
The cut points leave every (grade, quality-tier) stratum with >= 6 patients in
the full 300-patient cohort; exact matching on all four raw metadata fields
would leave many singleton strata and is intentionally not used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment


QUALITY_COLUMNS = ["Overall quality", "Clarity", "Field definition", "Artifact"]
QUALITY_BINS = [-np.inf, 0.625, 0.75, np.inf]
QUALITY_LABELS = ["low", "mid", "high"]


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return np.sum(left * right, axis=1)


def derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a random no-fixed-point permutation of a non-singleton group."""
    if len(indices) < 2:
        raise ValueError("Cannot construct a matched negative in a singleton stratum")
    for _ in range(200):
        candidate = rng.permutation(indices)
        if not np.any(candidate == indices):
            return candidate
    return np.roll(indices, 1)


def patient_table(manifest: pd.DataFrame, features: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create one row/patient with mean-pooled L and R representations."""
    required = {"patient_id", "eye", "patient_grade", "benchmark_split", *QUALITY_COLUMNS}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    records: list[dict] = []
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    for patient_id, group in manifest.groupby("patient_id", sort=True):
        group = group.copy()
        group["eye"] = group["eye"].astype(str).str.upper()
        if set(group.eye) != {"L", "R"}:
            continue
        rec = {
            "patient_id": str(patient_id),
            "patient_grade": int(group["patient_grade"].mode().iloc[0]),
            "benchmark_split": str(group["benchmark_split"].mode().iloc[0]),
        }
        for col in QUALITY_COLUMNS:
            rec[col] = float(group[col].mean())
        # A transparent equal-weight composite; higher means visually cleaner.
        rec["quality_composite"] = float(
            0.25 * (
                rec["Overall quality"]
                + rec["Clarity"] / 10.0
                + rec["Field definition"] / 10.0
                + (10.0 - rec["Artifact"]) / 10.0
            )
        )
        records.append(rec)
        left_rows.append(features[group.index[group.eye == "L"].to_numpy()].mean(axis=0))
        right_rows.append(features[group.index[group.eye == "R"].to_numpy()].mean(axis=0))
    table = pd.DataFrame.from_records(records)
    table["quality_tier"] = pd.cut(
        table["quality_composite"], bins=QUALITY_BINS, labels=QUALITY_LABELS, include_lowest=True
    ).astype(str)
    return table, np.stack(left_rows), np.stack(right_rows)


def match_permutation(table: pd.DataFrame, match_columns: list[str], seed: int) -> np.ndarray:
    """Create one matched right-eye negative for every patient.

    All reported matched strata are audited to have >=2 samples.  This avoids a
    hidden fallback to unmatched negatives, which would invalidate the control.
    """
    rng = np.random.default_rng(seed)
    perm = np.arange(len(table))
    for _, subgroup in table.groupby(match_columns, sort=True, observed=False):
        idx = subgroup.index.to_numpy()
        if len(idx) < 2:
            raise ValueError(f"singleton matching stratum for {match_columns}: {subgroup.iloc[0][match_columns].to_dict()}")
        perm[idx] = derangement(idx, rng)
    if np.any(perm == np.arange(len(table))):
        raise AssertionError("negative matching has fixed points")
    return perm


def nearest_quality_permutation(table: pd.DataFrame, seed: int) -> np.ndarray:
    """Grade-exact, all-quality-fields-nearest one-to-one assignment.

    Unlike an exact match on raw metadata (which is mostly singleton), this
    solves a deranged minimum-cost assignment within each DR grade.  The cost
    is Euclidean distance over four equally scaled, direction-aligned fields:
    Overall quality, Clarity/10, Field definition/10, and 1-Artifact/10.
    A tiny seeded jitter only breaks exact cost ties; it does not relax grade
    matching or change the quality metric.
    """
    rng = np.random.default_rng(seed)
    vectors = np.column_stack(
        [
            table["Overall quality"].to_numpy(),
            table["Clarity"].to_numpy() / 10.0,
            table["Field definition"].to_numpy() / 10.0,
            1.0 - table["Artifact"].to_numpy() / 10.0,
        ]
    )
    perm = np.arange(len(table))
    for _, subgroup in table.groupby(["patient_grade"], sort=True, observed=False):
        idx = subgroup.index.to_numpy()
        if len(idx) < 2:
            raise ValueError("grade matching stratum is singleton")
        v = vectors[idx]
        cost = np.sqrt(np.maximum(((v[:, None, :] - v[None, :, :]) ** 2).sum(axis=2), 0.0))
        cost += rng.uniform(0.0, 1e-8, size=cost.shape)
        np.fill_diagonal(cost, 1e6)
        source, target = linear_sum_assignment(cost)
        if not np.array_equal(source, np.arange(len(idx))):
            raise AssertionError("assignment source index mismatch")
        perm[idx] = idx[target]
    if np.any(perm == np.arange(len(table))):
        raise AssertionError("nearest-quality matching has fixed points")
    return perm


def metrics(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    y = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    scores = np.concatenate([positive, negative])
    return {
        "same_patient_cosine_mean": float(positive.mean()),
        "negative_cosine_mean": float(negative.mean()),
        "mean_gap": float(positive.mean() - negative.mean()),
        "pair_identity_auroc": float(roc_auc_score(y, scores)),
    }


def bootstrap_pair_metrics(
    positive: np.ndarray, negative: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> dict[str, float | list[float]]:
    aucs, gaps = [], []
    n = len(positive)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        aucs.append(roc_auc_score(np.r_[np.ones(n), np.zeros(n)], np.r_[positive[idx], negative[idx]]))
        gaps.append(float(positive[idx].mean() - negative[idx].mean()))
    return {
        "bootstrap_pair_id_auroc_mean": float(np.mean(aucs)),
        "bootstrap_pair_id_auroc_95ci": [float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))],
        "bootstrap_cosine_gap_mean": float(np.mean(gaps)),
        "bootstrap_cosine_gap_95ci": [float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))],
    }


def standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / denom) if denom > 1e-12 else 0.0


def balance_record(table: pd.DataFrame, neg_index: np.ndarray, condition: str, seed: int) -> dict:
    rec: dict[str, float | int | str] = {"condition": condition, "seed": seed}
    rec["mean_abs_quality_composite_difference"] = float(
        np.mean(np.abs(table["quality_composite"].to_numpy() - table["quality_composite"].to_numpy()[neg_index]))
    )
    for col in QUALITY_COLUMNS:
        source = table[col].to_numpy()
        target = source[neg_index]
        slug = col.lower().replace(" ", "_")
        rec[f"mean_abs_{slug}_difference"] = float(np.mean(np.abs(source - target)))
        rec[f"smd_{slug}"] = standardized_mean_difference(source, target)
    return rec


def probe_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        out["macro_ovr_auroc"] = float(roc_auc_score(y, prob, labels=labels, multi_class="ovr", average="macro"))
    except ValueError:
        out["macro_ovr_auroc"] = float("nan")
    return out


def run_linear_probes(table: pd.DataFrame, left: np.ndarray, right: np.ndarray) -> list[dict]:
    """Fit only on train; val/test are separately reported, no tuning loop."""
    x = (left + right) / 2.0  # patient representation, invariant to eye order.
    split = table["benchmark_split"].to_numpy()
    train = np.flatnonzero(split == "train")
    scaler = StandardScaler().fit(x[train])
    xz = scaler.transform(x)
    outputs: list[dict] = []
    for task, col in (("patient_DR_grade", "patient_grade"), ("quality_composite_tier", "quality_tier")):
        y = table[col].to_numpy()
        model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, multi_class="auto", random_state=20260922
        ).fit(xz[train], y[train])
        labels = model.classes_
        for eval_split in ("train", "val", "test"):
            idx = np.flatnonzero(split == eval_split)
            pred = model.predict(xz[idx])
            prob = model.predict_proba(xz[idx])
            row = {"task": task, "label": col, "fit_split": "train", "evaluation_split": eval_split, "n_patients": int(len(idx))}
            row.update(probe_metrics(y[idx], pred, prob, labels))
            outputs.append(row)
    return outputs


def markdown_report(out_json: Path, payload: dict) -> str:
    def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
        """Small dependency-free Markdown table formatter."""
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines = [header, divider]
        for _, row in frame[columns].iterrows():
            cells = []
            for col in columns:
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    cells.append("nan" if np.isnan(value) else f"{value:.{digits}f}")
                else:
                    cells.append(str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    audit = pd.DataFrame(payload["matching_strata_audit"])
    condition = pd.DataFrame(payload["pair_id_results"])
    probe = pd.DataFrame(payload["linear_metadata_probes"])
    aggregate = pd.DataFrame(payload["aggregate_results"])
    balance = pd.DataFrame(payload["matching_balance"])
    qcols = ["patient_grade", "quality_tier", "n_patients"]
    audit_table = markdown_table(audit, qcols, digits=0)
    cond_cols = ["condition", "seed", "pair_identity_auroc", "mean_gap", "negative_cosine_mean"]
    cond_table = markdown_table(condition, cond_cols)
    aggregate_cols = [
        "condition",
        "pair_identity_auroc_mean_over_seeds",
        "pair_identity_auroc_sd_over_seeds",
        "mean_gap_mean_over_seeds",
        "bootstrap_auroc_ci_lower_min_over_seeds",
        "bootstrap_auroc_ci_upper_max_over_seeds",
    ]
    aggregate_table = markdown_table(aggregate, aggregate_cols)
    balance_summary = (
        balance.groupby("condition", as_index=False)[
            [
                "mean_abs_quality_composite_difference",
                "mean_abs_overall_quality_difference",
                "mean_abs_clarity_difference",
                "mean_abs_field_definition_difference",
                "mean_abs_artifact_difference",
            ]
        ]
        .mean()
    )
    balance_table = markdown_table(balance_summary, list(balance_summary.columns))
    probe_cols = ["task", "evaluation_split", "n_patients", "balanced_accuracy", "macro_f1", "macro_ovr_auroc"]
    probe_table = markdown_table(probe, probe_cols)
    agg = payload["aggregate_results"]
    return f"""# DeepDRiD R2.0b/R2.0d：疾病+质量条件控制（v1）

## 结论边界

这是对**冻结 ImageNet 特征**的测量控制，不是新模型训练，也没有使用 CFP–OCT
选择任何参数。它测试：同患者 L/R 对是否仍比“疾病等级和可审计质量指标均匹配”的
跨患者负对更相似。结果只能支持 operational bilateral correspondence，不能证明
法律/临床意义的患者身份或疾病无关生物标志物。

## 质量匹配规则与可行性审计

每位患者的四张图先在患者层面均值汇总。质量复合分数为
`0.25 * (Overall quality + Clarity/10 + Field definition/10 + (10-Artifact)/10)`；
quality tier 为 low (`<=0.625`)、mid (`(0.625,0.75]`)、high (`>0.75`)。
使用 **exact patient-grade × quality-tier derangement** 构造负对。四个原始质量字段
均进入复合量；直接在四个原始值上 exact match 会产生大量 singleton/双样本 strata，
故不采用。下面每个联合 bucket 都至少有 {payload['minimum_joint_stratum_size']} 位患者，
没有 fallback 到未匹配负对。

{audit_table}

## 五个随机匹配种子

{cond_table}

跨 seed 汇总（CI 列是 5 个 seed 的 95% bootstrap CI 中最保守的下界和最宽的上界；
不是把 seeds 当独立患者叠加）：

{aggregate_table}

质量近邻程度的审计（source patient 与被指派 negative right-eye patient 的原始
metadata 绝对差，越小表示匹配越严格）：

{balance_table}

主 gate 使用每个 seed 的 paired-patient bootstrap CI。
`grade_only` 是第一轮的疾病控制复现；`grade_plus_quality_tier` 是可审计的 coarsened
主控制；`grade_plus_quality_nearest` 是更严格的近邻敏感性分析，在同 grade 内对四个
方向对齐的原始质量字段做一对一最小距离 deranged assignment。

## R2.0d：raw patient representation 的线性 metadata probe

patient representation 为 `(mean(L views)+mean(R views))/2`，标准化器和
multinomial logistic regression 只在 train 拟合（C=1、class-balanced、无调参），随后
独立报告 val/test。quality 的标签是上面的固定复合 tier；probe 表示表征中可线性读取
的 metadata，不表示因果泄漏。

{probe_table}

## 可复现性

- cohort: {payload['n_complete_bilateral_patients']} complete bilateral DeepDRiD patients / {payload['n_feature_rows']} image rows
- seeds: {payload['seeds']}
- bootstrap: {payload['n_bootstrap']} paired-patient resamples per seed and condition
- feature input: `{payload['features']}`
- manifest: `{payload['manifest']}`

## 解释纪律

如果 grade+quality 的 bootstrap AUROC CI 下界仍高于 0.5，R2.0b 的“仍存在条件共享
结构”gate 通过；否则第二轮停止 patient-factor narrative，转为 bilateral
correspondence measurement benchmark。无论结果如何，后续 factorization 必须把这些
metadata probes 作为消混杂对照，而不能只报告疾病分类分数。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260922, 20260923, 20260924, 20260925, 20260926])
    parser.add_argument("--n-bootstrap", type=int, default=3000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).reset_index(drop=True)
    features = np.load(args.features)
    if len(manifest) != len(features):
        raise ValueError(f"features ({len(features)}) and manifest ({len(manifest)}) differ")
    table, left, right = patient_table(manifest, features)

    audit = (
        table.groupby(["patient_grade", "quality_tier"], observed=False)
        .size()
        .reset_index(name="n_patients")
        .sort_values(["patient_grade", "quality_tier"])
    )
    minimum = int(audit["n_patients"].min())
    if minimum < 2:
        raise ValueError("joint matching has singleton strata; revise coarsening before running")

    positive = cosine_rows(left, right)
    pair_rows, balance_rows, bootstrap_rows = [], [], []
    conditions = {
        "grade_only": ("exact_stratum", ["patient_grade"]),
        "grade_plus_quality_tier": ("exact_stratum", ["patient_grade", "quality_tier"]),
        "grade_plus_quality_nearest": ("nearest_quality", ["patient_grade"]),
    }
    for seed in args.seeds:
        for condition, (match_mode, cols) in conditions.items():
            perm = match_permutation(table, cols, seed) if match_mode == "exact_stratum" else nearest_quality_permutation(table, seed)
            negative = cosine_rows(left, right[perm])
            row = {"condition": condition, "seed": int(seed), "n_positive": int(len(positive)), "n_negative": int(len(negative))}
            row.update(metrics(positive, negative))
            pair_rows.append(row)
            balance_rows.append(balance_record(table, perm, condition, seed))
            boot = {"condition": condition, "seed": int(seed)}
            boot.update(bootstrap_pair_metrics(positive, negative, np.random.default_rng(seed + 100000), args.n_bootstrap))
            bootstrap_rows.append(boot)

    aggregate = []
    for condition in conditions:
        subset = pd.DataFrame(pair_rows).query("condition == @condition")
        boot = pd.DataFrame(bootstrap_rows).query("condition == @condition")
        aggregate.append({
            "condition": condition,
            "pair_identity_auroc_mean_over_seeds": float(subset.pair_identity_auroc.mean()),
            "pair_identity_auroc_sd_over_seeds": float(subset.pair_identity_auroc.std(ddof=1)),
            "mean_gap_mean_over_seeds": float(subset.mean_gap.mean()),
            "bootstrap_auroc_ci_lower_min_over_seeds": float(min(x[0] for x in boot.bootstrap_pair_id_auroc_95ci)),
            "bootstrap_auroc_ci_upper_max_over_seeds": float(max(x[1] for x in boot.bootstrap_pair_id_auroc_95ci)),
            "r2_0b_gate_ci_lower_above_chance": bool(min(x[0] for x in boot.bootstrap_pair_id_auroc_95ci) > 0.5),
        })

    payload = {
        "protocol": "deepdrid_r2_0b_grade_quality_conditioned_raw_feature_pair_id_and_r2_0d_linear_metadata_probe_v1",
        "features": str(args.features),
        "manifest": str(args.manifest),
        "n_feature_rows": int(len(manifest)),
        "n_complete_bilateral_patients": int(len(table)),
        "seeds": [int(x) for x in args.seeds],
        "n_bootstrap": int(args.n_bootstrap),
        "quality_composite": "0.25*(Overall quality + Clarity/10 + Field definition/10 + (10-Artifact)/10)",
        "quality_tier_bins": {"low": "<=0.625", "mid": "(0.625,0.75]", "high": ">0.75"},
        "matching": "exact patient_grade x quality_tier derangement; no unmatched fallback",
        "matching_strata_audit": audit.to_dict(orient="records"),
        "minimum_joint_stratum_size": minimum,
        "pair_id_results": pair_rows,
        "matching_balance": balance_rows,
        "bootstrap_results": bootstrap_rows,
        "aggregate_results": aggregate,
        "linear_metadata_probes": run_linear_probes(table, left, right),
        "split_counts": table.groupby("benchmark_split").size().to_dict(),
        "external_data_used_for_selection": False,
        "model_training_or_threshold_tuning": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(pair_rows).to_csv(args.out.with_suffix(".pair_id.csv"), index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.out.with_suffix(".bootstrap.csv"), index=False)
    pd.DataFrame(balance_rows).to_csv(args.out.with_suffix(".balance.csv"), index=False)
    pd.DataFrame(payload["linear_metadata_probes"]).to_csv(args.out.with_suffix(".metadata_probe.csv"), index=False)
    args.out.with_suffix(".md").write_text(markdown_report(args.out, payload), encoding="utf-8")
    print(json.dumps({"aggregate_results": aggregate, "out": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
