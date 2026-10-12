# Project status, evidence roles, and claim boundaries

## Status at manuscript release

The current release accompanies manuscript version 1.8, titled “Evaluating Same-Patient Matching Between Left- and Right-Eye Fundus Images Under Disease and Image-Quality Controls.” It updates dataset-source URLs and cites the source ethics/consent procedures for DeepDRiD, PAPILA, and CFP–OCT; the analyses and scientific results are unchanged from version 1.7. The release also includes the revision v11 aggregate audit and current editable vector figures. The study remains a conditional bilateral correspondence evaluation.

## Dataset roles

| Dataset | Role | Included evidence | Boundary |
|---|---|---|---|
| DeepDRiD | Internal development and controlled multi-view evaluation | 300 complete bilateral patients; patient-disjoint 210/46/44 split; separate strict four-view and matched-control eligible subsets | Different subset denominators are reported separately. No test patient is used to fit or choose the primary raw representation. |
| PAPILA | Locked final external evaluation | 244 complete bilateral patients; frozen raw ImageNet ResNet-50 pair-ID test under random and exact ordered diagnosis-signature-matched negatives | No H-CBFD evaluation, model selection, thresholding, or post-result redesign is allowed on this dataset. |
| CFP–OCT | Previously inspected development stress readout | 137 complete bilateral patients; raw baseline versus H-CBFD under two DME-matching schemas | Not blind final external validation. H-CBFD underperforms raw baseline and is not claimed to transfer. |
| Paraguay | Separate image-level transport diagnostic | A single-image AUROC is recorded in historical audit material | No verified patient/eye links; it cannot support bilateral patient-level validation. |

## Main reported readouts

- Frozen raw features on DeepDRiD: same-eye similarity 0.8308, cross-eye similarity 0.8241, and grade-matched negative similarity 0.7366 in the corresponding analysis.
- PAPILA: pair-identity AUROC 0.8669 ± 0.0041 with random cross-patient negatives and 0.8585 ± 0.0095 with exact ordered bilateral glaucoma-diagnosis-signature matching.
- CFP–OCT: the selected H-CBFD candidate is below the raw frozen baseline by 0.0529 AUROC under patient-max DME matching and 0.0408 under the legacy eye-mode schema.
- DeepDRiD H-CBFD screening selected `hierarchical_no_adversary` by its prespecified validation criterion, but its CFP–OCT stress performance failed the 0.02 preservation tolerance. It is retained as a negative/limited method result, not a validated domain-invariant method.

For exact denominators, uncertainty procedures, and per-analysis details, use the result reports in `../results/` and the captions attached to the main figures. Older planning files are context, not replacements for the current manuscript.

## Analysis decisions that must remain frozen

1. Keep the patient as the split and resampling unit.
2. Keep the ResNet-50 weights, transform, eye ordering, and negative-assignment protocol fixed for a reported evaluation.
3. Keep PAPILA as the consumed final external readout. Do not use its outputs for subsequent selection.
4. Keep CFP–OCT explicitly labeled as a previously inspected development stress readout.
5. Do not migrate DR, glaucoma, and DME labels across datasets or describe pair-identity AUROC as clinical disease accuracy.
6. Do not claim identity recovery, a disease-independent biomarker, domain invariance, or clinical readiness.
7. Report distinct eligibility denominators for full cohort, strict four-view, matched raw controls, and H-CBFD rather than implying a common sample size.

## Historical decisions retained in the repository

Candidate data access work considered public and controlled sources. mBRSET remained behind PhysioNet credentialed access and was not replaced by unauthorized mirrors; ODIR and China-Fundus-CIMT did not become primary datasets in this manuscript; Paraguay lacked patient/eye identifiers sufficient for bilateral claims. The old Phase-2 document is preserved as a planning record, while the frozen PAPILA protocol and current result reports describe the final external role.
