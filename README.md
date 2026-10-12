# Bilateral Cross-Eye Correspondence

This repository accompanies **“Evaluating Same-Patient Matching Between Left- and Right-Eye Fundus Images Under Disease and Image-Quality Controls.”** The current manuscript file is version 1.9. The analyses and results are unchanged from version 1.8. The repository preserves analysis code, frozen protocols, figure sources, identity-free figure inputs, aggregate results, and the manuscript, subject to each source dataset's terms.

**Repository:** <https://github.com/kobayashi0512/bilateral-shared-signal>

## Study in brief

The study asks whether a generic frozen fundus representation contains a measurable same-patient, opposite-eye correspondence signal after cross-patient negatives are controlled for disease grade and image quality. It also separates eye/view hierarchy checks from the primary pair comparison and keeps datasets in distinct evidential roles.

The work is an evaluation study, not a patient-identification system, a disease-independent biomarker claim, or a clinical decision tool. PAPILA evaluates the frozen baseline only. The learned H-CBFD candidate did not preserve the raw baseline on the inspected CFP–OCT development stress set and is not presented as an externally validated method.

## Current manuscript and figures

- [Manuscript DOCX — version 1.9](manuscript/bilateral_1.9.docx)
- [Manuscript PDF — version 1.9](manuscript/bilateral_1.9.pdf)
- [Figure 1: study framework](figures/figure1_framework/package_v4/07_svg_export/figure1_framework.svg) and [editable PowerPoint](figures/figure1_framework/package_v4/06_editable_pptx/figure1_framework.pptx)
- [Figures 2–4: editable SVG/PDF sources and captions](figures/revision_v11/)
- [Revision v11 results and patient-node audit](results/revision_v11/REVISION_V11_CONDITIONAL_AUDIT.md)

Figure 1 includes four unmodified DeepDRiD examples (two views per eye) and an OD/OS PAPILA example pair. Figure 4 contains a separate illustrative PAPILA pair. The source image licenses and attribution are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); no CFP–OCT source photographs are redistributed because the source repository did not state an image-reuse license.

## Evidence snapshot

| Dataset | Role in the paper | Frozen result / boundary |
|---|---|---|
| DeepDRiD | Internal patient-disjoint development and conditional controls | 300 patients; current split 210/46/44. The strict four-view analysis, common-denominator raw control, and H-CBFD test use their stated eligible subsets and are not interchangeable. |
| PAPILA | Locked final external test of the frozen ImageNet ResNet-50 baseline | 244 complete bilateral patients. Pair-identity AUROC: 0.8669 ± 0.0041 with random negatives and 0.8585 ± 0.0095 with exact ordered bilateral glaucoma-diagnosis-signature matching. No model or threshold was selected on PAPILA. |
| CFP–OCT | Previously inspected development stress readout | 137 complete bilateral patients. H-CBFD was below the raw frozen baseline by 0.0529 AUROC under patient-max DME matching and 0.0408 under the legacy eye-mode schema. It is not a blind final external set. |

See [project decisions and claim boundaries](docs/PROJECT_STATUS_AND_DECISIONS.md) and the dataset-specific reports in [`results/`](results/).

## Repository map

- `code/` — dataset normalization, manifest/split checks, frozen feature extraction, conditional controls, uncertainty analysis, and frozen PAPILA probe.
- `protocols/` — analysis and freeze protocols. `PHASE2_PUBLIC_DATA_PROTOCOL.md` is retained as a historical planning record; see the current roles and decisions before interpreting it.
- `results/` — concise result reports and identity-free aggregate plotting inputs. Raw images, raw archives, patient manifests, model weights, and feature caches are not included.
- `figures/` — paper figures, figure-generation scripts, source image provenance, captions, and the editable Figure 1 package.
- `manuscript/` — current DOCX/PDF release.
- `docs/` and `environment/` — data-access/licensing, reproduction guidance, and runtime notes.

## Reproduction outline

1. Read [dataset access and rights](docs/DATA_ACCESS_AND_LICENSES.md) and obtain each source dataset directly under its terms.
2. Use the matching normalization/preflight script in `code/` to create local manifests; keep all patient splitting at the patient level.
3. Follow the frozen protocol for the dataset role. Do not use PAPILA to tune a model or threshold; do not use CFP–OCT to select a new H-CBFD variant.
4. Run the corresponding analysis script and compare with the concise report in `results/`.

The exact local data were not bundled, so the commands require user-supplied dataset paths and are not a one-click reproduction from a clean checkout. The revision v11 audit report records the assignment seeds and patient-node uncertainty procedure; the PAPILA source photographs used in Figure 4 are embedded in the released figure, while the generator expects authorized local image files at the paths noted in its manifest. Package versions and known constraints are documented in [environment/README.md](environment/README.md). No full dependency lockfile is claimed.

## Citation

If this repository is useful, cite the manuscript and the original dataset papers. A citation record is provided in [`CITATION.cff`](CITATION.cff). Dataset papers and source repositories remain the authoritative sources for dataset descriptions and access conditions.

## Reuse status

No code license has been selected by the authors yet; see [LICENSE_SELECTION_REQUIRED.md](LICENSE_SELECTION_REQUIRED.md). This repository is publicly viewable, but public visibility alone does not grant permission to reuse the code or third-party images. Dataset reuse remains subject to each dataset's terms.
