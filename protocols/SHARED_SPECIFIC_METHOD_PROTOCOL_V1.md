# Shared/specific method protocol v1

## Default architecture

For each eye, an image encoder produces (z_L,z_R). The primary patient path
uses the symmetric shared input

\[
s = (z_L + z_R) / (m_L + m_R),
\]

where (m_e) is the eye-presence mask. A patient-label head reads only
`shared(s)` by default. The eye-specific path receives `abs(z_L-z_R)` and is
used as an auxiliary diagnostic branch, not as the main prediction shortcut.

## Training objectives

1. Patient endpoint loss on the shared path;
2. same-patient pair-consistency loss for the shared embedding;
3. patient-DR-level-matched negative-pair separation loss;
4. optional eye-specific auxiliary loss with a small coefficient;
5. orthogonality/decorrelation penalty between shared and specific embeddings;
6. explicit presence mask for missing-eye inference.

The initial image-model sweep should compare:

- shared-only;
- shared-only + pair consistency;
- shared + weak specific auxiliary branch;
- ordered concat control.

## Required evaluations

- patient-disjoint train/validation/test;
- full pair, missing left, missing right, same-patient swap;
- cross-patient eye replacement as a negative control;
- blur/occlusion/quality stratification;
- same-patient versus patient-grade-matched pair identity AUROC;
- source/camera/domain probe when a second dataset is available.

## Decision rule

The extra eye-specific branch is retained only if it improves the full endpoint
without reducing missing-eye robustness or same-patient pair separation. The
current frozen-feature screen favours shared-only as the default because it
reached mean balanced accuracy 0.7514 and was exactly swap-invariant, while
shared+specific reached 0.6660. A positive-only consistency term is not enough:
the pretrained-feature cross-patient screen showed that it can make
same-grade cross-patient pairs easier, so matched negatives are mandatory.
