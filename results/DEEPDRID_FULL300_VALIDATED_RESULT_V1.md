# DeepDRiD full-300 validated result v1

> **Historical report; not the current manuscript source of truth.** This archived
> report lists a 210/45/44 split, which sums to 299 despite its 300-patient
> heading. The current manuscript uses the audited 210/46/44 split and reports
> distinct eligible subsets for the strict four-view, matched-control, and
> H-CBFD analyses. Do not reuse the split or test values below as current paper
> results. See [`RESULTS_AT_A_GLANCE.md`](RESULTS_AT_A_GLANCE.md).

The complete audited split was re-extracted with cached ImageNet ResNet50
weights: 1,200 explicit images, 300 patients, and 210/45/44 train/validation/
test patients. The matched-negative coefficient was selected on validation
only from `{0, 0.05, 0.1, 0.2, 0.4}`. The selected coefficient is **0.0**:
validation balanced accuracy was 0.6399 at 0.0 versus 0.6177 at 0.05/0.1,
0.5910 at 0.2, and 0.6257 at 0.4.

Using the selected classification-only shared head on the held-out 44 patients:

| Condition | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Full pair | 0.7227 | **0.7259** | 0.6974 |
| Missing left | 0.7227 | 0.6982 | 0.6929 |
| Missing right | 0.6636 | 0.6584 | 0.6313 |
| Same-patient swap | 0.7227 | **0.7259** | 0.6974 |

The exploratory `w=0.1` run reached 0.7365 full balanced accuracy on this test,
but it was not selected by the complete validation split and must not be used
as the primary result. The defensible conclusion is that the shared symmetric
head is validated; matched-negative separation remains an ablation requiring
more patients or a better calibrated objective.
