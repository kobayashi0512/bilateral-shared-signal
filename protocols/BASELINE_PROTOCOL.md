# Bilateral shared-signal baseline protocol

`bilateral_baseline.py` freezes the patient-level input conditions before any
model is trained. It is manifest-only: it never opens an image path, downloads
files, or accesses credentialed datasets.

## Conditions

| CLI condition | Patient eligibility | Input order | Purpose |
|---|---|---|---|
| `single` | At least one recognized eye | right eye preferred, otherwise left | Single-eye baseline |
| `bilateral` | Exactly one selected left and one selected right row | `[left, right]` | Paired-eye baseline |
| `missing_eye` | Same bilateral subset | one retained eye; choose `--missing-side left/right` | Missing-view robustness |
| `swap_eyes` | Same bilateral subset | `[right, left]` | Eye-order invariance |

If a patient has duplicate rows for the same eye, the first row in manifest
order is selected and the patient is counted in
`duplicate_patients_resolved_first_row`. Filename-based eye inference is
disabled. Unknown eye values are not eligible for bilateral conditions.

All train/validation/test assignment must be performed by `patient_id` before
model fitting. The `source_path` field is copied to the output for traceability
only; it is never checked or opened by this script.

## Usage

```bash
python3 bilateral_baseline.py \
  --manifest path/to/patient_manifest.csv \
  --dataset DATASET_NAME \
  --condition all \
  --out audit_outputs/bilateral_baseline.json
```

For a one-sided missing-eye stress test:

```bash
python3 bilateral_baseline.py \
  --manifest path/to/patient_manifest.csv \
  --condition missing_eye \
  --missing-side left \
  --out audit_outputs/missing_left.json
```

## Dry-run result

The protocol was run on the existing local
`safe_oct_referral/data/processed/paired_manifest.csv` (207 patients, 345
manifest rows), without opening any image. The resulting report is
`audit_outputs/bilateral_baseline_dryrun.json`.

| Condition | Cases | Notes |
|---|---:|---|
| single | 207 | right preferred when both eyes exist |
| bilateral | 137 | one selected L/R pair |
| missing_eye | 137 | left removed by default; right retained |
| swap_eyes | 137 | order reversed |

The dry-run recorded `image_paths_opened=0`, `downloads_performed=0`, and
`patient_or_eye_inferred_from_filename=false` for every condition.
