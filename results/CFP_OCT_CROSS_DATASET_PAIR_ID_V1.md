# CFP–OCT cross-dataset bilateral pair-identity validation v1

The local CFP–OCT manifest contains 207 patients, 137 complete bilateral
patients, and explicit OD/OI laterality. All 345 fundus files exist and the
provided train/validation/test patient split has zero overlap. Because its
endpoint is DME from OCT rather than DeepDRiD DR, no disease label is migrated;
this track tests only whether a generic frozen visual representation preserves
same-patient bilateral structure.

| Negative control | Same-patient cosine | Negative cosine | Gap | Pair-ID AUROC (5 permutations) |
|---|---:|---:|---:|---:|
| Random cross-patient | 0.8978 | 0.7274 | 0.1704 | 0.9102 ± 0.0072 |
| Patient-DME-level matched | 0.8978 | 0.7366 | 0.1612 | **0.9074 ± 0.0117** |

Patient bootstrap over 3,000 resamples gives pair-ID AUROC mean 0.8586 with
95% CI [0.7926, 0.9349] and cosine-gap 95% CI [0.0625, 0.1783]. The wider
interval reflects only 137 complete bilateral patients. This is the first
external bilateral validation of the shared-signal measurement protocol; it is
not a cross-dataset disease-classification result.
