# Results at a glance — current manuscript v1.8

This index summarizes the results used in manuscript v1.8. The numerical source of truth is the [revision v11 conditional and patient-node uncertainty audit](revision_v11/REVISION_V11_CONDITIONAL_AUDIT.md); historical logs and earlier figure inputs may use superseded denominators or protocols. Consult the full report and manuscript captions for matching rules and uncertainty interpretation.

| Evidence | Current denominator | Current result |
|---|---:|---|
| DeepDRiD patient-disjoint source cohort | 300 patients; split 210/46/44 | The split components sum to 300. Full-cohort conditional analyses are descriptive; held-out analyses use their own eligible subsets. |
| Strict four-view hierarchy | Validation n=35; test n=33 | Same-eye / cross-eye / matched-negative mean similarities are 0.8447 / 0.8407 / 0.7540 in validation and 0.8331 / 0.8382 / 0.7721 in test. The same-eye-minus-cross-eye interval envelopes include zero in both splits; cross-eye exceeds the matched negative in both. |
| Existing conditional controls, full cohort | n=300 | Pair-identity AUROC: DR-grade matching 0.8827 ± 0.0086; DR-grade × patient-quality-tier matching 0.8660 ± 0.0088; nearest-quality matching 0.8500 ± 0.0044. |
| Existing conditional controls, held-out common subset | n=42 | In the same control order: 0.8549 ± 0.0173, 0.8348 ± 0.0150, and 0.8731 ± 0.0051. These denominators are not interchangeable with the four-view subset. |
| Post-hoc exact bilateral controls | Full cohort n=293; test n=34 | Exact ordered bilateral DR-signature matching: 0.8847 ± 0.0089 full-cohort and 0.8675 ± 0.0213 test. Adding right-eye quality-tier matching: 0.8700 ± 0.0052 and 0.8614 ± 0.0068, respectively. |
| H-CBFD exploratory candidate | DeepDRiD validation n=42; test n=41 | Validation selects `hierarchical_no_adversary`: raw 0.8425 ± 0.0132 vs H-CBFD 0.8816 ± 0.0195. Held-out test: raw 0.7842 ± 0.0118 vs H-CBFD 0.8579 ± 0.0364. The candidate was not reevaluated under the stricter post-hoc eye-specific controls. |
| PAPILA locked external evaluation | 244 complete bilateral patients | Frozen raw ImageNet ResNet-50 pair-identity AUROC: 0.8669 ± 0.0041 with random cross-patient negatives and 0.8585 ± 0.0095 with exact ordered bilateral diagnosis-signature matching. PAPILA evaluates the frozen baseline only; it does not validate H-CBFD. |
| CFP–OCT development stress readout | 137 eligible bilateral patients | Raw vs H-CBFD AUROC is 0.9015 ± 0.0054 vs 0.8485 ± 0.0193 under patient-max DME matching, and 0.9108 ± 0.0054 vs 0.8701 ± 0.0226 under the legacy eye-mode schema. This previously inspected development-stress readout is not blind external validation. |

All ± values above are across-seed means and sample standard deviations. Patient-node bootstrap interval envelopes are reported in the linked audit and manuscript tables/figures; they are conditional on fixed assignments and do not include rematching uncertainty.

## Interpretation boundary

These results support measurable bilateral correspondence under the observed datasets and specified matching protocols. They do not establish real-world identity, a disease-independent biomarker, clinical utility, or complete control of unmeasured acquisition factors. PAPILA supports external evaluation of the frozen raw baseline only. CFP–OCT was already inspected during development, and its readout is a stress test rather than blind external validation.
