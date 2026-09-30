#!/usr/bin/env python3
"""Create deterministic patient-disjoint train/validation/test manifests.

This is a metadata-only utility.  It reads CSV/TSV/JSON manifests and writes
new derived manifests; input files are never modified.  A patient identifier
and an explicitly provided laterality field are required for an auditable
bilateral split.  In particular, laterality is *never* inferred from an image
or file name.  Datasets without an eye field are still split at patient level,
but their output ``canonical_eye`` is ``unknown``.

Example
-------
python3 patient_disjoint_split.py \
  --spec DeepDRiD=../retinal_application_discovery/data/deepdrid/.../regular-fundus-training.csv \
  --spec safe_oct_referral=../safe_oct_referral/data/processed/paired_manifest.csv \
  --out-dir audit_outputs/patient_disjoint_split --seed 20260921
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from manifest_audit_runner import (
    FIELD_ALIASES,
    _clean,
    _normalize_eye,
    _read_records,
    _resolve,
    _parse_spec,
)


def _patient_pair_status(rows: list[dict[str, Any]], patient_field: str, eye_field: str | None) -> dict[str, str]:
    eyes: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        patient = _clean(row.get(patient_field))
        if patient:
            eyes[patient].append(_normalize_eye(_clean(row.get(eye_field))) if eye_field else "unknown")

    status: dict[str, str] = {}
    for patient, values in eyes.items():
        counts = Counter(values)
        recognized = set(values) & {"L", "R"}
        if not eye_field or not recognized:
            status[patient] = "unknown"
        elif "unknown" in values:
            status[patient] = "conflict"
        elif recognized == {"L", "R"} and counts["L"] == counts["R"] == 1:
            status[patient] = "complete_pair"
        elif len(recognized) == 1 and counts[next(iter(recognized))] == 1:
            status[patient] = "single_eye"
        else:
            status[patient] = "multiple_same_eye"
    return status


def _allocate_counts(n: int, fractions: tuple[float, float, float]) -> list[int]:
    """Allocate a stratum without dropping a partition when possible."""
    raw = [n * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[: n - sum(counts)]:
        counts[index] += 1
    if n >= 3:
        # A tiny stratum should not be forced to contain all partitions; this
        # repair only applies to the aggregate cohort below.  For strata with
        # at least three patients, preserve one patient per partition.
        for target in (0, 1, 2):
            if counts[target] == 0:
                donor = max((index for index in range(3) if counts[index] > 1), key=lambda index: counts[index], default=None)
                if donor is not None:
                    counts[donor] -= 1
                    counts[target] += 1
    return counts


def _assign_patients(
    patients: list[str], seed: int, fractions: tuple[float, float, float], groups: dict[str, str] | None = None
) -> dict[str, str]:
    """Assign patients, optionally stratified by a patient-level group."""
    if not groups:
        groups = {patient: "__unstratified__" for patient in patients}
    grouped: dict[str, list[str]] = defaultdict(list)
    for patient in sorted(patients):
        grouped[groups.get(patient, "__missing__")].append(patient)

    assignments: dict[str, str] = {}
    split_names = ("train", "val", "test")
    for group_name in sorted(grouped):
        members = list(grouped[group_name])
        group_seed = int.from_bytes(hashlib.sha256(f"{seed}|{group_name}".encode("utf-8")).digest()[:8], "big")
        random.Random(group_seed).shuffle(members)
        counts = _allocate_counts(len(members), fractions)
        cursor = 0
        for split, count in zip(split_names, counts):
            for patient in members[cursor : cursor + count]:
                assignments[patient] = split
            cursor += count

    # With many one-patient strata it is possible for rounding to leave a
    # global partition empty.  Repair deterministically while preserving all
    # patient-disjoint and patient-level-label constraints.
    if len(patients) >= 3:
        for missing in split_names:
            if missing in assignments.values():
                continue
            donors = [split for split in split_names if split != missing and sum(value == split for value in assignments.values()) > 1]
            if not donors:
                continue
            donor = max(donors, key=lambda split: sum(value == split for value in assignments.values()))
            patient = sorted(patient for patient, split in assignments.items() if split == donor)[-1]
            assignments[patient] = missing
    return assignments


def _resolve_stratification(
    fields: list[str], rows: list[dict[str, Any]], patient_field: str, requested: str
) -> tuple[str | None, dict[str, str], dict[str, Any]]:
    auto_aliases = ("patient_DR_Level", "patient_label", "label", "diagnosis", "patient_diagnosis", "dme_oct")
    resolved = _resolve(fields, auto_aliases if requested == "auto" else (requested,))
    info: dict[str, Any] = {"requested_field": requested, "resolved_field": resolved, "status": "unstratified_fallback", "fallback_reason": None}
    if not resolved:
        info["fallback_reason"] = "no_matching_label_field"
        return None, {}, info

    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient = _clean(row.get(patient_field))
        if patient:
            values[patient].add(_clean(row.get(resolved)) or "__missing__")
    groups: dict[str, str] = {}
    conflict_count = 0
    missing_count = 0
    for patient, labels in values.items():
        if labels == {"__missing__"}:
            groups[patient] = "__missing__"
            missing_count += 1
        elif len(labels) > 1:
            groups[patient] = "__conflict__"
            conflict_count += 1
        else:
            groups[patient] = next(iter(labels))
    if not groups or len(set(groups.values())) == 1 and "__missing__" in groups.values():
        info["fallback_reason"] = "label_values_all_missing"
        return None, {}, info
    info["status"] = "stratified"
    info["group_counts"] = dict(Counter(groups.values()))
    info["patients_with_missing_label"] = missing_count
    info["patients_with_conflicting_labels"] = conflict_count
    return resolved, groups, info


def _split_dataset(name: str, path: Path, seed: int, train_frac: float, val_frac: float, test_frac: float, out_dir: Path, stratify_field: str) -> dict[str, Any]:
    rows, fields, source_format = _read_records(path)
    resolved = {name_: _resolve(fields, aliases) for name_, aliases in FIELD_ALIASES.items()}
    patient_field = resolved["patient_id"]
    if not patient_field:
        raise ValueError(f"{name}: no patient identifier field resolved; refusing an unauditable split")

    missing_patient_rows = [index for index, row in enumerate(rows) if not _clean(row.get(patient_field))]
    if missing_patient_rows:
        raise ValueError(f"{name}: {len(missing_patient_rows)} rows have missing patient IDs; refusing an unauditable split")

    patients = sorted({_clean(row.get(patient_field)) for row in rows})
    resolved_stratify, stratification_groups, stratification = _resolve_stratification(fields, rows, patient_field, stratify_field)
    assignments = _assign_patients(patients, seed, (train_frac, val_frac, test_frac), stratification_groups or None)
    pair_status = _patient_pair_status(rows, patient_field, resolved["eye"])
    output_fields = list(fields)
    for extra in ("dataset", "canonical_eye", "pair_status", "stratification_group", "benchmark_split"):
        if extra not in output_fields:
            output_fields.append(extra)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        patient = _clean(row.get(patient_field))
        output = dict(row)
        output["dataset"] = name
        output["canonical_eye"] = _normalize_eye(_clean(row.get(resolved["eye"]))) if resolved["eye"] else "unknown"
        output["pair_status"] = pair_status[patient]
        output["stratification_group"] = stratification_groups.get(patient, "__unstratified__")
        output["benchmark_split"] = assignments[patient]
        output_rows.append(output)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    output_path = out_dir / f"{safe_name}_patient_disjoint_manifest.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)

    row_counts = Counter(row["benchmark_split"] for row in output_rows)
    patient_counts = Counter(assignments.values())
    split_sets = {split: {p for p, value in assignments.items() if value == split} for split in ("train", "val", "test")}
    overlaps = {
        f"{left}_{right}": len(split_sets[left] & split_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    source_split_field = resolved["split"]
    source_agreement = None
    if source_split_field:
        comparable = 0
        agreed = 0
        for row in rows:
            source = _clean(row.get(source_split_field)).lower()
            patient = _clean(row.get(patient_field))
            if source in {"train", "val", "test"}:
                comparable += 1
                agreed += source == assignments[patient]
        source_agreement = {"rows_with_standard_split": comparable, "rows_agreeing": agreed}

    report = {
        "dataset": name,
        "source_path": str(path),
        "source_format": source_format,
        "output_manifest": str(output_path),
        "seed": seed,
        "ratios_requested": {"train": train_frac, "val": val_frac, "test": test_frac},
        "rows": len(rows),
        "patients": len(patients),
        "patient_counts": dict(patient_counts),
        "row_counts": dict(row_counts),
        "pair_status_patient_counts": dict(Counter(pair_status.values())),
        "eye_field": resolved["eye"],
        "stratification": stratification,
        "canonical_eye_counts": dict(Counter(row["canonical_eye"] for row in output_rows)),
        "split_patient_overlaps": overlaps,
        "source_split_agreement": source_agreement,
        "checks": {
            "patient_disjoint": all(value == 0 for value in overlaps.values()),
            "all_input_rows_emitted_once": len(output_rows) == len(rows),
            "filename_eye_inference_used": False,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", action="append", required=True, type=_parse_spec, metavar="DATASET=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument(
        "--stratify-field",
        default="auto",
        help="Patient-level label field. 'auto' tries patient_DR_Level, label, diagnosis, then dme_oct; missing fields safely fall back to unstratified.",
    )
    args = parser.parse_args()
    fractions = (args.train_frac, args.val_frac, args.test_frac)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
        raise SystemExit("--train-frac, --val-frac and --test-frac must be positive and sum to 1")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for name, path in args.spec:
        if not path.exists():
            raise SystemExit(f"Input manifest does not exist: {path}")
        reports.append(_split_dataset(name, path, args.seed, *fractions, args.out_dir, args.stratify_field))

    summary = {
        "schema_version": "1.0",
        "runner": "patient_disjoint_split.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "filename_eye_inference_used": False,
        "seed": args.seed,
        "stratify_field": args.stratify_field,
        "datasets": reports,
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["dataset", "seed", "rows", "patients", "train_patients", "val_patients", "test_patients", "train_rows", "val_rows", "test_rows", "patient_disjoint", "filename_eye_inference_used"]
    with (args.out_dir / "split_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            writer.writerow({
                "dataset": report["dataset"], "seed": report["seed"], "rows": report["rows"], "patients": report["patients"],
                "train_patients": report["patient_counts"].get("train", 0), "val_patients": report["patient_counts"].get("val", 0), "test_patients": report["patient_counts"].get("test", 0),
                "train_rows": report["row_counts"].get("train", 0), "val_rows": report["row_counts"].get("val", 0), "test_rows": report["row_counts"].get("test", 0),
                "patient_disjoint": report["checks"]["patient_disjoint"], "filename_eye_inference_used": report["checks"]["filename_eye_inference_used"],
            })
    print(json.dumps({"datasets": [report["dataset"] for report in reports], "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
