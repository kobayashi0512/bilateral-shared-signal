#!/usr/bin/env python3
"""Normalize an extracted official PAPILA release into a bilateral manifest.

The PAPILA descriptor specifies two clinical XLSX files (OD and OS) plus
``RET###OD.jpg``/``RET###OS.jpg`` images.  This adapter uses those documented
structures only, joins each image to the clinical row with the same official
patient ID, and aborts on any unmatched or duplicate record.  It deliberately
does not remove ``Diagnosis == 2`` suspect cases: cohort selection belongs in
a pre-registered downstream protocol, not data ingestion.

No third-party dataset mirror is needed.  The XLSX reader below uses only the
Python standard library, avoiding an implicit dependency on pandas/openpyxl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
IMAGE_PATTERN = re.compile(r"^RET(?P<patient>[0-9]{3})(?P<eye>OD|OS)\.jpg$", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _column_index(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha())
    number = 0
    for character in letters:
        number = number * 26 + ord(character.upper()) - ord("A") + 1
    return number - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(node.itertext()) for node in root.findall(f"{{{SPREADSHEET_NS}}}si")]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    sheet = workbook.find(f".//{{{SPREADSHEET_NS}}}sheet")
    if sheet is None:
        raise ValueError("Workbook has no sheets")
    relationship_id = sheet.attrib.get(f"{{{REL_NS}}}id")
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    for relationship in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib["Target"].lstrip("/")
            return target if target.startswith("xl/") else f"xl/{target}"
    raise ValueError("Could not resolve first worksheet relationship")


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(cell.itertext()).strip()
    value = cell.find(f"{{{SPREADSHEET_NS}}}v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        index = int(value.text)
        return strings[index].strip()
    return value.text.strip()


def _read_xlsx(path: Path) -> list[list[str]]:
    with zipfile.ZipFile(path) as archive:
        strings = _shared_strings(archive)
        root = ET.fromstring(archive.read(_first_sheet_path(archive)))
    table: list[list[str]] = []
    for row in root.findall(f".//{{{SPREADSHEET_NS}}}sheetData/{{{SPREADSHEET_NS}}}row"):
        cells: dict[int, str] = {}
        for cell in row.findall(f"{{{SPREADSHEET_NS}}}c"):
            reference = cell.attrib.get("r")
            if reference:
                cells[_column_index(reference)] = _cell_text(cell, strings)
        width = max(cells, default=-1) + 1
        table.append([cells.get(index, "") for index in range(width)])
    if len(table) < 3:
        raise ValueError(f"Expected header, helper row, and records in {path}")
    return table


def _normalise_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _clinical_records(path: Path, eye: str) -> dict[str, dict[str, str]]:
    table = _read_xlsx(path)
    header = list(table[0])
    if header:
        header[0] = "ID"  # Official spreadsheets expose the first ID header as blank in some readers.
    wanted = {"id": "ID", "age": "Age", "gender": "Gender", "diagnosis": "Diagnosis"}
    positions: dict[str, int] = {}
    for index, name in enumerate(header):
        token = _normalise_header(name)
        if token in wanted:
            positions[wanted[token]] = index
    missing = sorted(set(wanted.values()) - set(positions))
    if missing:
        raise ValueError(f"{path.name} does not expose expected PAPILA columns: {', '.join(missing)}")

    records: dict[str, dict[str, str]] = {}
    # These workbooks contain two header rows followed by a one-cell ``ID``
    # helper row.  Data begin on worksheet row 4.  Treating that helper row as
    # a patient record would silently corrupt the OD/OS join, so skip it
    # explicitly and retain the worksheet row number in any error.
    for spreadsheet_row, values in enumerate(table[3:], start=4):
        def get(field: str) -> str:
            index = positions[field]
            return values[index].strip() if index < len(values) else ""
        raw_id = get("ID")
        patient_id = raw_id.removeprefix("#").strip()
        if not patient_id:
            continue
        if not patient_id.isdigit() or len(patient_id) != 3:
            raise ValueError(f"Unexpected PAPILA patient ID at {path.name}:{spreadsheet_row}: {raw_id!r}")
        if patient_id in records:
            raise ValueError(f"Duplicate PAPILA {eye} clinical ID {patient_id} in {path.name}")
        records[patient_id] = {
            "age": get("Age"),
            "gender": get("Gender"),
            "diagnosis": get("Diagnosis"),
            "clinical_source": f"PAPILA_official_patient_data_{eye.lower()}_xlsx",
        }
    if not records:
        raise ValueError(f"No clinical records read from {path}")
    return records


def _images(image_root: Path) -> list[tuple[Path, str, str]]:
    if not image_root.is_dir():
        raise ValueError(f"Missing official FundusImages directory: {image_root}")
    rows: list[tuple[Path, str, str]] = []
    unmatched: list[str] = []
    for path in sorted(image_root.iterdir()):
        if not path.is_file():
            continue
        match = IMAGE_PATTERN.fullmatch(path.name)
        if match:
            rows.append((path, match.group("patient"), match.group("eye").upper()))
        else:
            unmatched.append(path.name)
    if unmatched:
        raise ValueError(f"Unexpected files in official FundusImages (first 10): {unmatched[:10]}")
    if not rows:
        raise ValueError("No documented PAPILA RET###OD/OS JPEGs found")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Extracted official PAPILA_DBv1 directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    od_path = root / "ClinicalData" / "patient_data_od.xlsx"
    os_path = root / "ClinicalData" / "patient_data_os.xlsx"
    for required in (od_path, os_path):
        if not required.is_file():
            raise SystemExit(f"Missing official PAPILA spreadsheet: {required}")
    clinical = {"OD": _clinical_records(od_path, "OD"), "OS": _clinical_records(os_path, "OS")}
    image_rows = _images(root / "FundusImages")

    output: list[dict[str, str]] = []
    unmatched: list[str] = []
    for path, patient_id, source_eye in image_rows:
        record = clinical[source_eye].get(patient_id)
        if record is None:
            unmatched.append(path.name)
            continue
        output.append({
            "dataset": "PAPILA_v2_official",
            "patient_id": patient_id,
            "eye": "R" if source_eye == "OD" else "L",
            "image_id": path.stem,
            "image_path": str(path),
            "eye_source": "PAPILA_documented_RET_OD_OS_naming",
            "source_eye": source_eye,
            "age": record["age"],
            "gender": record["gender"],
            "diagnosis": record["diagnosis"],
            "clinical_source": record["clinical_source"],
        })
    if unmatched:
        raise SystemExit(f"Images without matching official clinical rows (first 10): {unmatched[:10]}")

    image_keys = {(row["patient_id"], row["source_eye"]) for row in output}
    clinical_keys = {(patient, eye) for eye, records in clinical.items() for patient in records}
    missing_images = sorted(clinical_keys - image_keys)
    if missing_images:
        raise SystemExit(f"Official clinical records without matching image rows (first 10): {missing_images[:10]}")
    if len(image_keys) != len(output):
        raise SystemExit("Duplicate image patient/eye records detected")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    complete_patients = sum(
        {row["eye"] for row in output if row["patient_id"] == patient} == {"L", "R"}
        for patient in {row["patient_id"] for row in output}
    )
    report = {
        "dataset": "PAPILA_v2_official",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "canonical_manifest": str(args.out.resolve()),
        "rows": len(output),
        "patients": len({row["patient_id"] for row in output}),
        "complete_bilateral_patients": complete_patients,
        "eye_counts": dict(Counter(row["eye"] for row in output)),
        "diagnosis_counts": dict(Counter(row["diagnosis"] for row in output)),
        "missing_diagnosis_rows": sum(not row["diagnosis"] for row in output),
        "clinical_workbooks_sha256": {str(od_path): _sha256(od_path), str(os_path): _sha256(os_path)},
        "provenance_rule": "image eye is parsed only from PAPILA's documented RET###OD/OS convention; patient/diagnosis/age/gender are read only from the matching official clinical workbook.",
        "selection_rule": "No diagnosis categories were removed during ingestion; freeze any downstream cohort selection before external evaluation.",
    }
    args.out.with_suffix(".preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
