# Results at a glance — current manuscript

This concise index follows the manuscript accompanying this release. Use the full manuscript captions and dataset-specific reports for protocol details and uncertainty interpretation. Historical logs and the archived `DEEPDRID_FULL300_VALIDATED_RESULT_V1.md` are not the source of current split denominators.

| Evidence | Current manuscript denominator | Result |
|---|---:|---|
| DeepDRiD patient-disjoint source cohort | 300 patients; split 210/46/44 | Full cohort is used for the descriptive conditional analysis; the split components sum to 300. |
| DeepDRiD strict four-view held-out hierarchy | 43 eligible test patients | Same-eye similarity 0.8308; same-patient cross-eye similarity 0.8241; grade-matched cross-patient similarity 0.7366. Paired uncertainty does not establish a reliable same-eye-over-cross-eye ordering; cross-eye similarity exceeds the grade-matched negative in the reported comparison. |
| DeepDRiD raw conditional-control audit | common eligible denominator n=42 | Disease-grade, grade × quality-tier, and nearest-quality controls are evaluated on the same eligible patients; do not equate this denominator with the full cohort or four-view subset. |
| DeepDRiD H-CBFD selected candidate | joint-matched test n=41 | Validation selects `hierarchical_no_adversary` (joint pair-ID AUROC 0.8816 ± 0.0174). Its CFP–OCT stress-set readout is below the raw frozen baseline, so it does not support a cross-domain superiority claim. |
| PAPILA locked final external test | 244 complete bilateral patients | Frozen raw ImageNet ResNet-50 pair-identity AUROC: 0.8669 ± 0.0041 under random cross-patient negatives; 0.8585 ± 0.0095 under exact ordered bilateral glaucoma-diagnosis-signature matching. PAPILA is not used for model/threshold selection and does not validate H-CBFD. |
| CFP–OCT development stress readout | 137 eligible bilateral patients | H-CBFD minus raw AUROC: −0.0529 under patient-max DME matching and −0.0408 under the legacy eye-mode schema. This is a previously inspected stress readout, not blind final external validation. |

## Interpretation boundary

These results support measurable bilateral correspondence under the observed dataset and matching protocols. They do not prove real-world identity, a disease-independent biomarker, clinical utility, or complete control of unmeasured acquisition factors. The PAPILA result applies only to the frozen raw baseline.
