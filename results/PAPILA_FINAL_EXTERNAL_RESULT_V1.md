# PAPILA frozen final-external result v1

冻结/执行日期：2026-09-22（Asia/Shanghai）  
Protocol: `PAPILA_EXTERNAL_FREEZE_PROTOCOL_V1.md`  
Role: previously unseen final external **bilateral correspondence** test; not a systemic-risk endpoint.

## Data-provenance gate — passed

The official Figshare delivery path returned HTTP 403 in this CLI environment.
The accepted distribution candidate was Kaggle
`orvile/papila-retinal-fundus-images`, version 1, whose page declares GPL-3
and PAPILA provenance. It is documented and cited as a distribution copy of
PAPILA—not as an independent dataset. The paper must cite the official PAPILA
descriptor/Figshare item and disclose the distribution source.

| Gate | Observed value | Verdict |
|---|---:|---|
| Archive byte count | 591,089,413 | pass |
| GCS MD5 | `ZfKbYTi5JjuDIdhZo3N9jA==` | pass |
| Archive SHA-256 | `7eaf0cacf82d3ed500a7935d0141ac78bc5278e7cef624c43f0d577fe58a9a71` | recorded |
| ZIP CRC / safe paths | 2,965 file members; no CRC or unsafe-path failure | pass |
| Official structure | `ClinicalData` OD/OS XLSX + `FundusImages/RET###OD/OS.jpg` | pass |
| Clinical/image linkage | 488 images, 244 patients; every path exists; no duplicate image IDs | pass |
| Complete bilateral patients | 244 / 244 | pass |
| Missing diagnosis | 0 / 488 | pass |

The complete machine-readable provenance is
`papila_external_frozen_manifest.distribution_provenance.json`; the generic
manifest audit reports zero warnings in `manifest_audit/PAPILA.json`.

## Frozen evaluation

Before opening PAPILA results, the protocol fixed: ImageNet ResNet-50 V2
weights and transform, cosine similarity, one image per explicit eye, five
fixed negative-pair seeds, no fine-tuning, and no threshold/model selection on
PAPILA. All 488 image paths yielded finite 2,048-d features.

| Negative control | Complete patients / seed | Pair-identity AUROC, mean ± SD | Cosine gap, range over seeds |
|---|---:|---:|---:|
| Random cross-patient right eye | 244 | **0.8669 ± 0.0041** | 0.0873–0.0906 |
| Exact bilateral glaucoma-diagnosis signature matched | 244 | **0.8585 ± 0.0095** | 0.0791–0.0915 |

The controlled result remains well above chance. Because cross-patient pairs
are matched on the ordered `(left diagnosis, right diagnosis)` signature, it
cannot be explained solely by the coarse glaucoma healthy/suspect label
combination. It is evidence that the frozen generic fundus representation
retains a measurable same-patient bilateral signal under a new disease cohort,
site and camera.

## Allowed and prohibited claims

Allowed: the **frozen ImageNet baseline** replicates bilateral pair
correspondence on an independent, complete-pair glaucoma cohort; the
diagnosis-matched sensitivity control remains positive.

Not allowed: claim that a newly learned H-CBFD/shared-specific method
generalizes externally (none was selected using PAPILA); claim systemic-risk
prediction; or select another model/threshold/negative construction after
seeing these numbers. PAPILA is now consumed as the registered final external
readout.

## Artifacts

- Manifest: `papila_external_frozen_manifest.csv`
- Preflight: `papila_external_frozen_manifest.preflight.json`
- Distribution provenance: `papila_external_frozen_manifest.distribution_provenance.json`
- Feature cache: `papila_resnet50_features.npy`
- Scores: `papila_frozen_external_probe_v1.json` and `.csv`
