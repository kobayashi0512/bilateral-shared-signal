# Revision v11: eye-level conditional and patient-node uncertainty audit

This post-hoc audit uses the frozen ImageNet ResNet-50 feature arrays. It does not train a model or alter the locked PAPILA encoder, preprocessing, or pair score.

## Why the analysis was extended

The DeepDRiD manifest contains 134 patients with different left- and right-eye DR grades; 20 of 44 held-out test patients are discordant. Patient-level DR matching alone therefore does not balance the ordered bilateral grade signature.

The stricter control matches cross-patient negatives within the exact ordered bilateral DR signature. It additionally matches the right-eye quality-composite tier; because the anchor left-eye feature is identical in its positive and negative pair, this balances both the source eye exactly and the target eye at the recorded quality-tier resolution.

## Bootstrap definition

For each fixed one-to-one derangement, the interval uses a patient-node (pigeonhole) bootstrap. A positive pair receives its patient multiplicity; a cross-patient negative edge receives the product of its anchor and donor multiplicities. This retains the dependence created when a patient appears as an anchor and as another pair's donor. Intervals remain conditional on each original assignment and do not include rematching uncertainty. The five assignment-level estimates are summarized with sample SD (ddof=1).

## DeepDRiD results

### Existing protocol, recomputed with node bootstrap

| Scope | Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope |
|---|---|---:|---:|---:|---:|
| Full cohort | patient_DR_grade | 300 | 0.8827 | 0.0086 | [0.8363, 0.9240] |
| Full cohort | patient_grade_x_patient_quality_tier | 300 | 0.8660 | 0.0088 | [0.8140, 0.9094] |
| Full cohort | patient_DR_grade_nearest_patient_quality | 300 | 0.8500 | 0.0044 | [0.7989, 0.8952] |
| Held-out common subset | patient_DR_grade | 42 | 0.8549 | 0.0173 | [0.7323, 0.9555] |
| Held-out common subset | patient_grade_x_patient_quality_tier | 42 | 0.8348 | 0.0150 | [0.6853, 0.9392] |
| Held-out common subset | patient_DR_grade_nearest_patient_quality | 42 | 0.8731 | 0.0051 | [0.7763, 0.9524] |

### Exact bilateral disease and eye-quality controls

| Scope | Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope | Mean absolute right-eye quality difference |
|---|---|---:|---:|---:|---:|---:|
| full_cohort | exact_ordered_bilateral_DR_signature | 293 | 0.8847 | 0.0089 | [0.8342, 0.9242] | 0.2241 ± 0.0046 |
| full_cohort | bilateral_DR_signature_plus_right_eye_quality_tier | 293 | 0.8700 | 0.0052 | [0.8253, 0.9123] | 0.0808 ± 0.0030 |
| validation | exact_ordered_bilateral_DR_signature | 36 | 0.8946 | 0.0196 | [0.7459, 0.9793] | 0.2158 ± 0.0075 |
| validation | bilateral_DR_signature_plus_right_eye_quality_tier | 36 | 0.8776 | 0.0172 | [0.7277, 0.9712] | 0.0938 ± 0.0048 |
| test | exact_ordered_bilateral_DR_signature | 34 | 0.8675 | 0.0213 | [0.6577, 0.9966] | 0.1509 ± 0.0191 |
| test | bilateral_DR_signature_plus_right_eye_quality_tier | 34 | 0.8614 | 0.0068 | [0.6814, 0.9947] | 0.0757 ± 0.0041 |

### Strict four-view hierarchy

| Split | Eligible four-view patients | Same-eye mean | Cross-eye mean | Matched-negative mean | Same-eye − cross-eye interval envelope | Cross-eye − negative interval envelope |
|---|---:|---:|---:|---:|---:|---:|
| validation | 35 | 0.8447 | 0.8407 | 0.7540 | [-0.0130, 0.0210] | [0.0339, 0.1451] |
| test | 33 | 0.8331 | 0.8382 | 0.7721 | [-0.0245, 0.0164] | [0.0205, 0.1114] |

## PAPILA frozen external baseline

| Control | n | Mean AUROC | Across-seed sample SD | Node-bootstrap interval envelope |
|---|---:|---:|---:|---:|
| unmatched_random | 244 | 0.8669 | 0.0041 | [0.8205, 0.9056] |
| bilateral_diagnosis_signature_matched | 244 | 0.8585 | 0.0095 | [0.8039, 0.9066] |

The archived per-seed PAPILA AUROCs were asserted to reproduce exactly before uncertainty was recomputed. This remains an external evaluation of the frozen raw baseline only; it does not validate H-CBFD.

## H-CBFD exploratory candidate: corrected across-seed SD

| Comparison | n | Raw frozen mean ± sample SD | H-CBFD mean ± sample SD |
|---|---:|---:|---:|
| DeepDRiD validation (patient-grade × patient-quality tier) | 42 | 0.8425 ± 0.0132 | 0.8816 ± 0.0195 |
| DeepDRiD held-out test (patient-grade × patient-quality tier) | 41 | 0.7842 ± 0.0118 | 0.8579 ± 0.0364 |
| CFP–OCT development stress set, patient-max DME | 137 | 0.9015 ± 0.0054 | 0.8485 ± 0.0193 |
| CFP–OCT development stress set, legacy eye-mode DME | 137 | 0.9108 ± 0.0054 | 0.8701 ± 0.0226 |

The selected candidate is hierarchical_no_adversary. The archived seedwise AUROCs were not changed; only SDs were recomputed from population SD to sample SD. A 0.10 branch mean-feature-SD floor was added to the rerunnable selector. Retrospective checks show all three archived validation variants pass and the winning variant is unchanged. The archived candidate has not been reevaluated under the stricter eye-specific disease/quality control above; CFP–OCT remains an already-inspected development stress set, not blind external validation.

## Interpretation boundary

All results are conditional on the available matching strata and the specified fixed assignment seeds. Full-cohort results are descriptive; the held-out analysis is a post-hoc sensitivity analysis. The stricter DeepDRiD control does not match continuous per-image quality, camera, site, age, pigmentation, or other unrecorded factors. No causal decomposition or identity biomarker claim follows from these AUROCs.
