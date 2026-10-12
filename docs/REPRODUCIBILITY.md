# Reproducibility guide

## Before running

1. Obtain DeepDRiD and, if reproducing the locked external analysis, PAPILA directly or through an authorized, provenance-verified distribution. Follow each source's terms in [`DATA_ACCESS_AND_LICENSES.md`](DATA_ACCESS_AND_LICENSES.md).
2. Keep data, manifests, feature caches, and checkpoints outside this repository. Never commit identifiers or credentials.
3. Install a compatible environment using [`../environment/README.md`](../environment/README.md). The historical analysis environment is recorded, but a complete lockfile and container image were not preserved.
4. Read the relevant protocol before running scripts. Protocols are intentionally explicit about roles and forbidden tuning.

## Script map

| Task | Script(s) | Protocol / output |
|---|---|---|
| DeepDRiD explicit laterality normalization | `../code/normalize_deepdrid_csv.py` | Writes a local normalized manifest; the eye field is parsed from the documented source schema rather than guessed for arbitrary files. |
| Patient-disjoint split | `../code/patient_disjoint_split.py` | Patient-level train/validation/test assignment; current full cohort split is 210/46/44. |
| Manifest audit | `../code/manifest_audit_runner.py` | Dataset admission, completeness, and schema audit. |
| Frozen ImageNet ResNet-50 feature extraction | `../code/extract_deepdrid_resnet50_features.py`, `../code/extract_manifest_resnet50_features.py` | `../protocols/REVISION_V6_PROTOCOL.md`; features remain local and are not released. |
| Conditional controls / paired uncertainty | `../code/conditional_shared_signal_controls.py`, `../code/bootstrap_shared_signal_full.py`, `../code/revision_v6_uncertainty_audit.py` | Disease/quality matching and paired-patient uncertainty audits. |
| H-CBFD feature screen | `../code/train_hcbfd_feature_screen.py` | Validation-only variant selection; CFP–OCT remains a stress readout. |
| Eye-level conditional controls and patient-node uncertainty audit | `../code/revision_v11_conditional_audit.py` | Reads local feature arrays and row-aligned manifests; the public report and aggregate JSON are in `../results/revision_v11/`. Patient-level inputs remain local. |
| PAPILA official structure and frozen probe | `../code/normalize_papila_official.py`, `../code/papila_frozen_external_probe.py` | `../protocols/PAPILA_EXTERNAL_FREEZE_PROTOCOL_V1.md`; PAPILA is not a tuning set. |
| Figure 1 | `../figures/figure1_framework/build_figure1_framework_v4.py` | Rebuild from the image assets documented in that directory; deliverable includes SVG and editable PPTX. |
| Figures 2–4 | `../figures/revision_v11/build_revision_v11_figures.py` | Uses the revision v11 aggregate audit and authorized local PAPILA example images for Figure 4; editable SVG/PDF outputs and captions are in `../figures/revision_v11/`. |

## Reproduction limits

The release omits raw images, clinical tables, patient manifests, cached features, trained checkpoints, and the full historical environment. Therefore a clean checkout alone cannot reproduce the numerical results. Users must acquire source data, pass the specified integrity/schema gates, supply local paths, and regenerate intermediate features. Some source analyses were run in a local research environment that is no longer captured by a fully pinned lockfile; see the environment note for the verified package versions and remaining uncertainty.

The reports under `../results/` are the concise numeric reference for this release. Do not substitute stale values from planning logs or the historical benchmark ledger where it conflicts with the current manuscript.
