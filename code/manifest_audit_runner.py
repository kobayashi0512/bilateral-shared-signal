#!/usr/bin/env python3
"""Run a read-only, patient-level audit over CSV/TSV/JSON manifests.

The runner intentionally treats missing patient/eye fields as unknown.  It
never derives patient IDs or laterality from a file name.  This makes the
summary safe to use before constructing patient-disjoint benchmark splits.

Example
-------
python3 manifest_audit_runner.py \
  --spec DeepDRiD=../retinal_application_discovery/data/deepdrid/.../regular-fundus-training.csv \
  --spec safe_oct_referral=../safe_oct_referral/data/processed/paired_manifest.csv \
  --out-dir audit_outputs/manifest_runner
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "patient_id": ("patient_id", "patient", "subject_id", "subject", "patientid"),
    "image_id": ("image_id", "image", "imageid", "fundus_name", "filename", "file"),
    "eye": ("eye", "laterality", "exam_eye", "eye_side", "side"),
    "visit_id": ("visit_id", "visit", "encounter_id", "exam_id"),
    "center": ("center", "centre", "site", "hospital", "institution"),
    "camera": ("camera", "device", "camera_model", "device_model"),
    "quality": ("quality", "overall_quality", "final_quality", "focus"),
    "source_path": (
        "source_path",
        "image_path",
        "fundus_path",
        "filepath",
        "file_path",
        "path",
    ),
    "split": ("split", "partition", "subset"),
}

EYE_L = {"l", "left", "os", "oi", "2"}
EYE_R = {"r", "right", "od", "1"}
PAIR_STATUSES = ("complete_pair", "single_eye", "multiple_same_eye", "conflict", "unknown")
MAX_DISTINCT = 50


def _norm_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _resolve(fields: list[str], aliases: Iterable[str]) -> str | None:
    normalized = {_norm_field(field): field for field in fields}
    for alias in aliases:
        found = normalized.get(_norm_field(alias))
        if found is not None:
            return found
    return None


def _read_records(path: Path) -> tuple[list[dict[str, Any]], list[str], str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        dialect = "excel-tab" if suffix == ".tsv" else "excel"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, dialect=dialect)
            fields = list(reader.fieldnames or [])
            return [dict(row) for row in reader], fields, suffix[1:]
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            with path.open(encoding="utf-8-sig") as handle:
                payload: Any = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict):
            for key in ("rows", "records", "data", "items"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"JSON manifest must be a list of objects: {path}")
        fields: list[str] = []
        for row in payload:
            for key in row:
                if key not in fields:
                    fields.append(key)
        return [dict(row) for row in payload], fields, suffix[1:]
    raise ValueError(f"Unsupported manifest format: {path}")


def _normalize_eye(value: str) -> str:
    token = _norm_field(value)
    if token in EYE_L:
        return "L"
    if token in EYE_R:
        return "R"
    return "unknown"


def _pair_statuses(
    rows: list[dict[str, Any]], patient_field: str | None, eye_field: str | None
) -> tuple[Counter[str], dict[str, int]]:
    if not patient_field:
        return Counter({"unknown": len(rows)}), {"missing_patient_id_rows": len(rows)}

    patients: dict[str, list[str]] = defaultdict(list)
    missing_patient_rows = 0
    for row in rows:
        patient = _clean(row.get(patient_field))
        if not patient:
            missing_patient_rows += 1
            continue
        patients[patient].append(_normalize_eye(_clean(row.get(eye_field))) if eye_field else "unknown")

    statuses: Counter[str] = Counter()
    for eyes in patients.values():
        unique = set(eyes)
        recognized = unique & {"L", "R"}
        has_unknown = "unknown" in unique
        counts = Counter(eyes)
        if not eye_field:
            status = "unknown"
        elif has_unknown and recognized:
            status = "conflict"
        elif not recognized:
            status = "unknown"
        elif recognized == {"L", "R"} and all(counts[eye] == 1 for eye in ("L", "R")):
            status = "complete_pair"
        elif recognized == {"L", "R"} or any(counts[eye] > 1 for eye in recognized):
            status = "multiple_same_eye"
        else:
            status = "single_eye"
        statuses[status] += 1
    if missing_patient_rows:
        statuses["unknown"] += missing_patient_rows
    for status in PAIR_STATUSES:
        statuses.setdefault(status, 0)
    return statuses, {"missing_patient_id_rows": missing_patient_rows}


def _nonempty_count(rows: list[dict[str, Any]], field: str) -> int:
    return sum(bool(_clean(row.get(field))) for row in rows)


def _distinct(rows: list[dict[str, Any]], field: str | None) -> list[str]:
    if not field:
        return []
    values = sorted({_clean(row.get(field)) for row in rows if _clean(row.get(field))})
    return values[:MAX_DISTINCT]


def _audit_dataset(name: str, path: Path, check_paths: bool = False, path_root: Path | None = None) -> dict[str, Any]:
    rows, fields, source_format = _read_records(path)
    resolved = {name_: _resolve(fields, aliases) for name_, aliases in FIELD_ALIASES.items()}
    patient_field = resolved["patient_id"]
    image_field = resolved["image_id"]
    eye_field = resolved["eye"]
    pair_status, pair_extra = _pair_statuses(rows, patient_field, eye_field)

    patient_values = {_clean(row.get(patient_field)) for row in rows} if patient_field else set()
    patient_values.discard("")
    image_values = {_clean(row.get(image_field)) for row in rows} if image_field else set()
    image_values.discard("")

    missingness = {
        field: {"nonempty": _nonempty_count(rows, field), "missing": len(rows) - _nonempty_count(rows, field)}
        for field in fields
    }
    warnings: list[str] = []
    if not patient_field:
        warnings.append("No patient identifier field resolved; patient-level split is not auditable.")
    if not eye_field:
        warnings.append("No eye/laterality field resolved; bilateral pairing is unknown; no filename inference was used.")
    if patient_field and len(patient_values) < 2:
        warnings.append("Fewer than two non-empty patient identifiers were found.")

    duplicate_image_rows = 0
    if image_field:
        counts = Counter(_clean(row.get(image_field)) for row in rows if _clean(row.get(image_field)))
        duplicate_image_rows = sum(count - 1 for count in counts.values() if count > 1)
        if duplicate_image_rows:
            warnings.append(f"{duplicate_image_rows} duplicate image-id rows detected.")

    path_audit: dict[str, Any] | None = None
    if check_paths and resolved["source_path"]:
        field = resolved["source_path"]
        existing = 0
        checked = 0
        for row in rows:
            raw = _clean(row.get(field))
            if not raw:
                continue
            checked += 1
            candidate = Path(raw)
            if not candidate.is_absolute() and path_root:
                candidate = path_root / candidate
            if candidate.exists():
                existing += 1
        path_audit = {"field": field, "checked": checked, "existing": existing, "missing": checked - existing}
        if path_audit["missing"]:
            warnings.append(f"{path_audit['missing']} source paths do not exist under the selected path root.")

    categorical_fields = ("eye", "center", "camera", "quality", "split")
    distinct = {
        key: _distinct(rows, resolved[key])
        for key in categorical_fields
        if resolved[key]
    }
    split_counts = Counter(_clean(row.get(resolved["split"])) for row in rows if resolved["split"] and _clean(row.get(resolved["split"])))
    eye_counts = Counter(_normalize_eye(_clean(row.get(eye_field))) for row in rows if eye_field and _clean(row.get(eye_field)))

    label_fields = [
        field for field in fields
        if any(token in _norm_field(field) for token in ("label", "diagnos", "dme", "dr_level", "cimt", "disease", "hypertension", "diabet"))
    ]
    label_summary = {
        field: {"nonempty": _nonempty_count(rows, field), "distinct": _distinct(rows, field)}
        for field in label_fields
    }

    return {
        "dataset": name,
        "source_path": str(path),
        "source_format": source_format,
        "rows": len(rows),
        "columns": fields,
        "resolved_fields": resolved,
        "patient_count": len(patient_values) if patient_field else None,
        "image_count": len(image_values) if image_field else None,
        "pair_status": dict(pair_status),
        "eye_counts": dict(eye_counts),
        "split_counts": dict(split_counts),
        "missingness": missingness,
        "distinct_values": distinct,
        "label_fields": label_summary,
        "duplicate_image_rows": duplicate_image_rows,
        "path_audit": path_audit,
        "warnings": warnings,
        "audit_notes": pair_extra,
    }


def _parse_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Each --spec must be DATASET=PATH")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Each --spec must contain a non-empty DATASET and PATH")
    return name.strip(), Path(path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", action="append", required=True, type=_parse_spec, metavar="DATASET=PATH")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--check-paths", action="store_true", help="Check resolved source paths without opening image files.")
    parser.add_argument("--path-root", type=Path, default=None, help="Root for relative source paths when --check-paths is used.")
    args = parser.parse_args()

    reports = []
    for name, path in args.spec:
        if not path.exists():
            raise SystemExit(f"Input manifest does not exist: {path}")
        reports.append(_audit_dataset(name, path, args.check_paths, args.path_root))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", report["dataset"])
        (out_dir / f"{safe_name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    combined = {
        "schema_version": "1.0",
        "runner": "manifest_audit_runner.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "filename_or_path_inference_for_patient_or_eye": False,
        "datasets": reports,
    }
    (out_dir / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_fields = [
        "dataset", "source_format", "rows", "patient_count", "image_count",
        "complete_pair", "single_eye", "multiple_same_eye", "conflict", "unknown",
        "missing_patient_id_rows", "duplicate_image_rows", "warning_count",
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for report in reports:
            row = {field: "" for field in summary_fields}
            row.update({key: report.get(key, "") for key in ("dataset", "source_format", "rows", "patient_count", "image_count", "duplicate_image_rows")})
            row.update({key: report["pair_status"].get(key, 0) for key in PAIR_STATUSES})
            row["missing_patient_id_rows"] = report["audit_notes"].get("missing_patient_id_rows", 0)
            row["warning_count"] = len(report["warnings"])
            writer.writerow(row)

    print(json.dumps({"datasets": [r["dataset"] for r in reports], "out_dir": str(out_dir), "summary_csv": str(out_dir / "summary.csv")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
